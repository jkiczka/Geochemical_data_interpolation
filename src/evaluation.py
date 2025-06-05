from pathlib import Path
import torch, xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import csv_to_xarray, download_csv

ds = csv_to_xarray(download_csv('temperature', 2))

arr = torch.from_numpy(ds.values.astype("float32")).unsqueeze(0)

# 3) potnij na małe kostki 16×16×8
patch = torch.nn.Unfold(kernel_size=(8,16,16), stride=(8,16,16))
vox = patch(arr)              # shape: (1, patch_vol, N_patches)
vox = vox.permute(2,0,1)      # (N_patches, 1, patch_vol)

loader = DataLoader(TensorDataset(vox), batch_size=64, shuffle=True)

# 4) minimalny auto-encoder 1-linear
ae = torch.nn.Sequential(
    torch.nn.Linear(vox.size(-1), 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, vox.size(-1))
)
opt = torch.optim.Adam(ae.parameters(), 1e-3)

for epoch in range(3):         # 3 × ~5 s na RTX3060
    for (x,) in loader:
        pred = ae(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        opt.zero_grad(); loss.backward(); opt.step()
print("loss", loss.item())