from dataclasses import dataclass
import math

import torch
import pytorch_lightning as pl


def generate_naca4_airfoil(code: str, num_points: int = 100):
    """Return x, y coordinates of a NACA 4 digit airfoil using cosine spacing."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must be 4 digits")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    x = [
        0.5 * (1 - math.cos(math.pi * i / (num_points - 1))) for i in range(num_points)
    ]

    y_t = [
        5
        * t
        * (
            0.2969 * math.sqrt(xi)
            - 0.1260 * xi
            - 0.3516 * xi**2
            + 0.2843 * xi**3
            - 0.1015 * xi**4
        )
        for xi in x
    ]

    y_c = []
    dyc_dx = []
    for xi in x:
        if xi < p:
            y_c.append(m / p**2 * (2 * p * xi - xi**2))
            dyc_dx.append(2 * m / p**2 * (p - xi))
        else:
            y_c.append(m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xi - xi**2))
            dyc_dx.append(2 * m / (1 - p) ** 2 * (p - xi))

    theta = [math.atan(dy) for dy in dyc_dx]

    xu = [xi - yt * math.sin(th) for xi, yt, th in zip(x, y_t, theta)]
    yu = [yc + yt * math.cos(th) for yc, yt, th in zip(y_c, y_t, theta)]
    xl = [xi + yt * math.sin(th) for xi, yt, th in zip(x, y_t, theta)]
    yl = [yc - yt * math.cos(th) for yc, yt, th in zip(y_c, y_t, theta)]

    x_coords = xu + xl[::-1]
    y_coords = yu + yl[::-1]
    return x_coords, y_coords


def generate_rect_mesh(num: int = 32, x_range=(-0.5, 1.5), y_range=(-0.5, 0.5)):
    """Generate a simple rectangular mesh as a list of (x, y) points."""
    xs = [x_range[0] + (x_range[1] - x_range[0]) * i / (num - 1) for i in range(num)]
    ys = [y_range[0] + (y_range[1] - y_range[0]) * j / (num - 1) for j in range(num)]
    return [[x, y] for y in ys for x in xs]


class NACA4Dataset(torch.utils.data.Dataset):
    """Dataset generating meshes for random NACA 4 digit airfoils."""

    def __init__(self, codes):
        self.codes = codes
        self.input_dim = 2
        self.output_dim = 2

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        _ = generate_naca4_airfoil(code)
        mesh = generate_rect_mesh()
        mesh = torch.tensor(mesh, dtype=torch.float32)
        target = torch.zeros(mesh.shape[0], self.output_dim)
        return mesh, target


class GraphNeuralOperatorModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4


def main(cfg: TrainConfig) -> None:
    codes = ["2412", "0012", "4412", "2424"]
    train_ds = NACA4Dataset(codes)
    val_ds = NACA4Dataset(codes)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GraphNeuralOperatorModel(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.output_dim,
        learning_rate=cfg.learning_rate,
    )

    trainer = pl.Trainer(max_epochs=cfg.epochs)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
