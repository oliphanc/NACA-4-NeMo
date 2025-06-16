import os
from dataclasses import dataclass

import torch
import pytorch_lightning as pl

from nemo.collections.pde.datasets.navier_stokes_dataset import NavierStokesNACADataset
from nemo.collections.pde.models.graph_neural_operator import GraphNeuralOperatorModel


@dataclass
class TrainConfig:
    data_dir: str = "data"
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4


def main(cfg: TrainConfig) -> None:
    train_ds = NavierStokesNACADataset(cfg.data_dir, split="train")
    val_ds = NavierStokesNACADataset(cfg.data_dir, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
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
