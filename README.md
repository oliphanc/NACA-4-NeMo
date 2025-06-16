# NACA-4-NeMo

This repository demonstrates a minimal workflow for training a graph neural operator (GNO).
The example generates NACA 4-series airfoil geometry procedurally and builds a simple mesh
around the airfoil using cosine spacing.

## Training

Install dependencies such as `torch` and `pytorch_lightning` and run the training script:

```bash
python train_gno_naca4.py
```
