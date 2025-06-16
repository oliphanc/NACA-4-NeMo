# Example training script using NVIDIA PhysicsNeMo
from dataclasses import dataclass

import torch

# Import the PhysicsNemo framework and PDEs
import physicsnemo as nemo
from physicsnemo.pdes import UnsteadyNavierStokes
from physicsnemo.models import GraphNeuralOperator
from physicsnemo.solvers import PhysicsSolver


def generate_naca4_airfoil(code: str, num_points: int = 100):
    """Return x, y coordinates of a NACA 4 digit airfoil using cosine spacing."""
    import math

    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must be 4 digits")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    x = [0.5 * (1 - math.cos(math.pi * i / (num_points - 1))) for i in range(num_points)]

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


class NACA4Problem(nemo.PDEProblem):
    """Define the unsteady Navier-Stokes problem around a NACA4 airfoil."""

    def __init__(self, code: str):
        self.code = code
        x_coords, y_coords = generate_naca4_airfoil(code)
        mesh = generate_rect_mesh()
        geometry = nemo.geometry.Mesh(mesh, boundaries={"airfoil": list(zip(x_coords, y_coords))})
        equations = UnsteadyNavierStokes(nu=1e-3, rho=1.0)
        super().__init__(geometry=geometry, equations=equations)


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4


def main(cfg: TrainConfig) -> None:
    codes = ["2412", "0012"]
    problems = [NACA4Problem(c) for c in codes]
    dataset = nemo.data.PDEDataset(problems)
    dataloader = nemo.data.PDEDataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = GraphNeuralOperator(in_features=2, out_features=2)
    solver = PhysicsSolver(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=cfg.learning_rate))

    ptrainer = nemo.framework.PhysicsTrainer(max_epochs=cfg.epochs)
    ptrainer.fit(solver=solver, train_dataloaders=dataloader)


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
