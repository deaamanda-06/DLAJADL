"""
models/123450043/run.py  --  Jalur RNN (GRU atas baris citra), kasus tinggi air wadah B.

Contoh:
    python run.py --data-dir ../../data --seed 1 --out-dir ../../outputs/123450043

PENTING: bagian load_samples() adalah ADAPTER. Sesuaikan dengan format data
sesi meja / keluaran pemotong jendela (#4) / PROTOKOL.md di repo tim.
Skema metrics.json juga harus dicocokkan dengan yang diminta starter dosen.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# 0. Utilitas
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# 1. Data  (ADAPTER -- sesuaikan!)
# ----------------------------------------------------------------------------
from data_utils import load_samples, split_by_block, to_arrays  # noqa: E402


# ----------------------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------------------
class RowGRU(nn.Module):
    """Citra (H x W) dibaca sebagai urutan H baris, tiap baris W fitur."""
    def __init__(self, in_dim=224, hid=64, layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True,
                          bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hid, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                 # x: (B, H, W)
        out, _ = self.rnn(x)
        # rata-rata sepanjang baris lebih stabil daripada hanya langkah terakhir
        return self.head(out.mean(dim=1)).squeeze(-1)


# ----------------------------------------------------------------------------
# 3. Latih & evaluasi
# ----------------------------------------------------------------------------
def make_loader(x, y, bs, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, x, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def mae_mm(model, x, y_mm, y_mean, y_std, device):
    pred = predict(model, x, device) * y_std + y_mean
    return float(np.mean(np.abs(pred - y_mm)))


def fit(xtr, ytr_mm, xva, yva_mm, stats, args, device, epochs=None, patience=15):
    x_mean, x_std, y_mean, y_std = stats
    xtr_n = (xtr - x_mean) / x_std
    ytr_n = (ytr_mm - y_mean) / y_std
    xva_n = (xva - x_mean) / x_std if xva is not None else None

    model = RowGRU(hid=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    loader = make_loader(xtr_n.astype(np.float32), ytr_n.astype(np.float32),
                         args.batch_size, True)

    best, best_ep, best_state, bad = 1e9, 0, None, 0
    n_ep = epochs or args.epochs
    for ep in range(1, n_ep + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if xva_n is not None:
            va = mae_mm(model, xva_n.astype(np.float32), yva_mm, y_mean, y_std, device)
            if va < best:
                best, best_ep, bad = va, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0:
                print(f"ep {ep:03d} | val MAE {va:.2f} mm | best {best:.2f} @ {best_ep}")
            if bad >= patience:
                break
    if best_state is not None and epochs is None:
        model.load_state_dict(best_state)
    return model, best_ep, best


# ----------------------------------------------------------------------------
# 4. ONNX + latensi
# ----------------------------------------------------------------------------
def export_onnx(model, path, device):
    model.eval()
    dummy = torch.zeros(1, 224, 224, device=device)
    kw = dict(input_names=["image"], output_names=["mm"],
              dynamic_axes={"image": {0: "batch"}}, opset_version=17)
    try:                                   # torch baru: pakai exporter klasik
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:                      # torch lama tidak punya argumen dynamo
        torch.onnx.export(model, dummy, path, **kw)
    return os.path.getsize(path) / 1024.0


def latency_ms(onnx_path, n=100):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        x = np.zeros((1, 224, 224), dtype=np.float32)
        for _ in range(10):
            sess.run(None, {"image": x})
        t = time.perf_counter()
        for _ in range(n):
            sess.run(None, {"image": x})
        return (time.perf_counter() - t) / n * 1000.0
    except Exception as e:
        print("Latensi ONNX gagal diukur:", e)
        return None


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")          # run_all.py tidak mengirim argumen ini
    ap.add_argument("--out-dir", default="results/123450043")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--final", action="store_true",
                    help="evaluasi data uji (blok uji) SEKALI, hanya di sprint terakhir")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--test-block", type=int, default=3)
    ap.add_argument("--val-block", type=int, default=2)
    args = ap.parse_args()

    t0 = time.perf_counter()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_samples(args.data_dir)
    train, val, test = split_by_block(samples, args.test_block, args.val_block)
    print(f"frame: train={len(train)} val={len(val)} test={len(test)}")
    xtr, ytr = to_arrays(train); xva, yva = to_arrays(val)

    # normalisasi dari data LATIH saja
    stats = (float(xtr.mean()), float(xtr.std() + 1e-8), float(ytr.mean()), float(ytr.std() + 1e-8))

    # Latih di blok latih, pilih epoch terbaik di blok validasi
    model, best_ep, best_val = fit(xtr, ytr, xva, yva, stats, args, device)
    x_mean, x_std, y_mean, y_std = stats
    pva = predict(model, ((xva - x_mean) / x_std).astype(np.float32), device) * y_std + y_mean
    val_rmse = float(np.sqrt(np.mean((pva - yva) ** 2)))
    print(f"epoch terbaik={best_ep}, val MAE={best_val:.2f} mm, val RMSE={val_rmse:.2f} mm")

    test_mae = None
    n_fit = len(xtr)
    if args.final:
        # latih ulang di blok latih+validasi dengan epoch tsb, uji SEKALI di blok uji
        xte, yte = to_arrays(test)
        x_all = np.concatenate([xtr, xva]); y_all = np.concatenate([ytr, yva])
        stats = (float(x_all.mean()), float(x_all.std() + 1e-8),
                 float(y_all.mean()), float(y_all.std() + 1e-8))
        set_seed(args.seed)
        model, _, _ = fit(x_all, y_all, None, None, stats, args, device, epochs=best_ep)
        x_mean, x_std, y_mean, y_std = stats
        test_mae = mae_mm(model, ((xte - x_mean) / x_std).astype(np.float32), yte, y_mean, y_std, device)
        n_fit = len(x_all)
        print(f"TEST MAE (blok {args.test_block}) = {test_mae:.2f} mm")

    # ONNX: bungkus normalisasi agar model menerima citra mentah 0..1 dan keluar mm
    class Wrapped(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, x):
            return s.m((x - x_mean) / x_std) * y_std + y_mean
    onnx_path = os.path.join(args.out_dir, f"rnn_seed{args.seed}.onnx")
    size_kb = export_onnx(Wrapped(model).to(device), onnx_path, device)
    lat = latency_ms(onnx_path)

    # TODO: cocokkan kunci dengan skema tim (models/TEMPLATE/run.py) dan isi data_hash
    metrik = {"val_mae_mm": round(best_val, 4), "val_rmse_mm": round(val_rmse, 4)}
    if test_mae is not None:
        metrik["test_mae_mm"] = round(test_mae, 4)
    metrics = {
        "model": "rnn_gru_baris",
        "jalur": "rnn",
        "nim": "123450043",
        "seed": args.seed,
        "metrik": metrik,
        "onnx_kb": round(size_kb, 2),
        "latency_ms": None if lat is None else round(lat, 3),
        "durasi_detik": round(time.perf_counter() - t0, 1),
        "perangkat": device,
        "n_fit_frames": int(n_fit),
        "n_val_frames": int(len(xva)),
        "val_block": args.val_block,
        "best_epoch": best_ep,
        "final": bool(args.final),
        "data_hash": None,   # TODO: harus sama dengan MANIFEST/PROTOKOL tim
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
