"""Model Transformer Encoder atas patch 16x16 untuk regresi tinggi air (mm).

    python models/123450047/run.py --seed 1 [--final]
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
JALUR = "trf"


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.float32))
    Xva_t = torch.tensor(np.asarray(Xva, dtype=np.float32))
    Xte_t = torch.tensor(np.asarray(Xte, dtype=np.float32)) if final and Xte is not None else None

    class PatchTransformer(nn.Module):
        def __init__(self, d_model: int = 64, patch: int = 16):
            super().__init__()
            n_patches = (224 // patch) ** 2
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1), nn.Linear(d_model, 1))

        def forward(self, x):
            # (B, 3, 224, 224) -> (B, 196, d_model)
            tok = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
            out = self.encoder(tok).mean(dim=1)
            return torch.nn.functional.softplus(self.head(out).squeeze(-1))

    model = PatchTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    batch_size = 16
    epochs = 12

    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr_t))
        for i in range(0, len(idx), batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(Xva_t).numpy()
        pred_test = model(Xte_t).numpy() if Xte_t is not None else None
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
