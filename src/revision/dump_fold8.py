"""Dump fold-8 scores from the seed-0 fold8 checkpoints (inference only).

Needed to fit the logistic parent aggregator on fold 8, so that folds 9 and 10
are never used for fitting. Fold 8 selected the checkpoint, so its scores carry
mild optimism; that is stated where the aggregator is reported.
"""
import pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PTBXL_PATH, HELME_DATA, CHECKPOINT_DIR
from src.utils.data import load_ptbxl_metadata, apply_standardizer, load_standardizer
from src.models.resnet1d import ECGResNet

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
meta = load_ptbxl_metadata(PTBXL_PATH)
idx8 = np.where(meta["strat_fold"].values == 8)[0]
X = np.load(Path(CHECKPOINT_DIR)/"X_raw.npy")
sc = load_standardizer(Path(HELME_DATA)/"standard_scaler.pkl")
X = np.ascontiguousarray(apply_standardizer(X, sc).transpose(0,2,1))[idx8]

for name, nc in [("acuity35", 35), ("super5", 5)]:
    d = Path(CHECKPOINT_DIR)/"retrain"/f"{name}_seed0_fold8"
    m = ECGResNet(num_classes=nc).to(dev)
    m.load_state_dict(torch.load(d/"model_selected.pt", map_location=dev, weights_only=True))
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            b = torch.from_numpy(X[i:i+64].astype(np.float32)).to(dev)
            out.append(m(b).cpu().numpy())
    logits = np.concatenate(out)
    np.save(d/"logits_fold8.npy", logits)
    np.save(d/"scores_fold8.npy", 1/(1+np.exp(-logits)))
    print(f"  {name}: fold-8 scores {logits.shape} -> {d}")
np.save(Path(CHECKPOINT_DIR)/"retrain"/"fold8_index.npy", idx8)
print(f"  fold-8 records: {len(idx8)}")
