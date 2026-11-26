"""Model Transfer Learning (ResNet-18) untuk regresi tinggi air (mm) dari citra 224x224.

    python models/123450038/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"


def _buat_backbone():
    from torchvision import models
    from torch import nn

    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        m = models.resnet18(weights=None)
    # Bekukan layer awal agar pelatihan cepat di CPU
    for name, param in m.named_parameters():
        if not name.startswith(("layer4", "fc")):
            param.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 1))
    return m


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]

    Xtr_t = torch.tensor((np.asarray(Xtr, dtype=np.float32) - mean) / std)
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.float32))
    Xva_t = torch.tensor((np.asarray(Xva, dtype=np.float32) - mean) / std)
    Xte_t = (
        torch.tensor((np.asarray(Xte, dtype=np.float32) - mean) / std)
        if final and Xte is not None
        else None
    )

    model = _buat_backbone()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    batch_size = 16
    epochs = 6

    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr_t))
        for i in range(0, len(idx), batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            pred = torch.nn.functional.softplus(model(Xtr_t[b]).squeeze(-1))
            loss = loss_fn(pred, ytr_t[b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = torch.nn.functional.softplus(model(Xva_t).squeeze(-1)).numpy()
        pred_test = (
            torch.nn.functional.softplus(model(Xte_t).squeeze(-1)).numpy()
            if Xte_t is not None
            else None
        )
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val, dtype=np.float32))
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test, dtype=np.float32))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
