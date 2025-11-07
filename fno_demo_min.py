#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo FNO: train on evalV.mat, test on testV.mat (no trainingV.mat needed)

- Uses utilities3: MatReader, UnitGaussianNormalizer, LpLoss, count_params
- Inputs: coeff (inverted as 1/x), II
- Target: sol
"""

import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from timeit import default_timer
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params

# -----------------------------
# Paths (edit if needed)
# -----------------------------
EVAL_PATH = 'data/evalV.mat'   # used as "training" set
TEST_PATH = 'data/testV.mat'   # used as test set

# -----------------------------
# Hyperparameters
# -----------------------------
p_dropout   = 0.0        # dropout prob (set >0 if you want)
weight_decay= 9e-4
patience    = 20         # early stopping patience (epochs)

ntrain_cap  = 80000      # cap max train samples (will clamp to dataset size)
ntest_cap   = 100        # cap max test samples
batch_size  = 50
learning_rate = 1e-2
epochs      = 100

modes1 = 16
modes2 = 16
width  = 18
padding= 20
r      = 1               # downsample stride

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.empty_cache()

# ================================================================
# Fourier layer and model (your architecture, simplified blocks)
# ================================================================
class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> multiply low modes -> IFFT"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.dropout  = nn.Dropout(p=p_dropout)

    def compl_mul2d(self, a, w):
        # (B, Cin, X, Y) x (Cin, Cout, X, Y) -> (B, Cout, X, Y)
        return torch.einsum("bixy,ioxy->boxy", a, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],  self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=p_dropout)
    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1  = modes1
        self.modes2  = modes2
        self.width   = width
        self.padding = padding

        # NOTE: inputs are 2 channels (II, 1/coeff) + 2D grid = 4 -> width
        self.p = nn.Linear(4, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.mlp0  = MLP(width, width, width)
        self.mlp1  = MLP(width, width, width)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)

        # Head
        self.q  = MLP(width, 1, width * 4)

        self.drop_conv0 = nn.Dropout(p=p_dropout)
        self.drop_mlp0  = nn.Dropout(p=p_dropout)
        self.drop_conv1 = nn.Dropout(p=p_dropout)
        self.drop_mlp1  = nn.Dropout(p=p_dropout)

    def get_grid(self, shape, device):
        B, S1, S2 = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, S1, dtype=torch.float, device=device).view(1, S1, 1, 1).repeat(B, 1, S2, 1)
        gridy = torch.linspace(0, 1, S2, dtype=torch.float, device=device).view(1, 1, S2, 1).repeat(B, S1, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        # x: (B, H, W, 2) => append grid => 4 chans for linear
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)       # (B, H, W, 4)
        x = self.p(x)                          # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)              # (B, C, H, W)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x);    x1 = self.drop_conv0(x1)
        x1 = self.mlp0(x1);    x1 = self.drop_mlp0(x1)
        x  = F.gelu(x1 + self.w0(x))

        x1 = self.conv1(x);    x1 = self.drop_conv1(x1)
        x1 = self.mlp1(x1);    x1 = self.drop_mlp1(x1)
        x  = F.gelu(x1 + self.w1(x))

        x  = x[..., :-self.padding, :-self.padding]
        x  = self.q(x)
        x  = x.permute(0, 2, 3, 1)             # (B, H, W, 1)
        return x

# =========================
# Load & prep data
# =========================
def load_eval_as_train_and_test(eval_path, test_path, r=1):
    reader = MatReader(eval_path)
    # Training from evalV
    x_tr_coeff = reader.read_field('coeff')[:, ::r, ::r]
    x_tr_II    = reader.read_field('II')    [:, ::r, ::r]
    y_tr       = reader.read_field('sol')   [:, ::r, ::r]

    # Apply 1/x on coeff
    x_tr_coeff = 1.0 / x_tr_coeff

    # Test from testV
    reader.load_file(test_path)
    x_te_coeff = reader.read_field('coeff')[:, ::r, ::r]
    x_te_II    = reader.read_field('II')    [:, ::r, ::r]
    y_te       = reader.read_field('sol')   [:, ::r, ::r]
    x_te_coeff = 1.0 / x_te_coeff

    # Normalization (fit on train)
    x_norm1 = UnitGaussianNormalizer(x_tr_coeff); x_tr_coeff = x_norm1.encode(x_tr_coeff); x_te_coeff = x_norm1.encode(x_te_coeff)
    x_norm2 = UnitGaussianNormalizer(x_tr_II);    x_tr_II    = x_norm2.encode(x_tr_II);    x_te_II    = x_norm2.encode(x_te_II)
    y_norm  = UnitGaussianNormalizer(y_tr);       y_tr       = y_norm.encode(y_tr)

    # Shapes
    s1, s2 = x_te_coeff.numpy().shape[1], x_te_coeff.numpy().shape[2]

    # Pack (B,H,W,2)
    ntrain = min(ntrain_cap, x_tr_coeff.shape[0])
    ntest  = min(ntest_cap,  x_te_coeff.shape[0])

    x_train = torch.zeros(ntrain, s1, s2, 2)
    x_test  = torch.zeros(ntest,  s1, s2, 2)

    for i in range(ntrain):
        x_train[i, :, :, 0] = x_tr_II[i, :, :]
        x_train[i, :, :, 1] = x_tr_coeff[i, :, :]
    for i in range(ntest):
        x_test[i, :, :, 0]  = x_te_II[i, :, :]
        x_test[i, :, :, 1]  = x_te_coeff[i, :, :]

    y_train = y_tr[:ntrain, ::]
    y_test  = y_te[:ntest,  ::]

    return x_train, y_train, x_test, y_test, y_norm, s1, s2, ntrain, ntest

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, x_test, y_test, y_normalizer, s1, s2, ntrain, ntest = \
        load_eval_as_train_and_test(EVAL_PATH, TEST_PATH, r=r)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test,  y_test),  batch_size=batch_size, shuffle=False
    )

    model = FNO2d(modes1, modes2, width).to(device)
    print("Parameters:", count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Use a cosine scheduler over total training steps, but guard T_max >= 1
    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    myloss = LpLoss(size_average=False)
    y_normalizer.to(device)

    best_loss = float('inf')
    counter   = 0

    train_losses = []
    test_losses  = []

    step_idx = 0
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x).reshape(x.size(0), s1, s2)
            out = y_normalizer.decode(out)
            y   = y_normalizer.decode(y)

            loss = myloss(out.view(x.size(0), -1), y.view(x.size(0), -1))
            loss.backward()
            optimizer.step()

            train_l2_sum += loss.item()
            scheduler.step()
            step_idx += 1

        model.eval()
        test_l2_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).reshape(x.size(0), s1, s2)
                out = y_normalizer.decode(out)
                test_l2_sum += myloss(out.view(x.size(0), -1), y.view(x.size(0), -1)).item()

        # average by number of samples (not batches)
        train_l2 = train_l2_sum / max(1, ntrain)
        test_l2  = test_l2_sum  / max(1, ntest)

        train_losses.append(train_l2)
        test_losses.append(test_l2)

        t2 = default_timer()
        print(f"epoch {ep:03d} | {t2-t1:.2f}s | train_l2 {train_l2:.6e} | test_l2 {test_l2:.6e}")

        # Early stopping heuristic (your original ratio check)
        if test_l2 < 1.5 * train_l2:
            if test_l2 < best_loss:
                best_loss = test_l2
                counter = 0
                torch.save(model.state_dict(), 'best_model_eval_test.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping after epoch {ep}.")
            break

if __name__ == "__main__":
    main()
