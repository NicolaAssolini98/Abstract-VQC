# Abstract VQC

Lightweight utilities and experiments that use interval arithmetic with PennyLane to verify the robustness of Variational Quantum Classifiers (VQCs).

This repository accompanies a paper presenting a formal framework for robustness verification of VQCs; the paper is available on arXiv: https://arxiv.org/abs/2507.10635

### Quick structure
- Top-level scripts: `complex_interval.py`, `intervalVQC.py`, `vvqc_utils.py`.
- Model folders: `1_iris_QCL/`, `2_irs_CCQC/`, `3_digits_CCQC/`, `4_QCNN/` (each contains training/verification scripts and notebooks).
- `req.txt` — conda environment package list used by the author.

### Setup
Create a conda environment from `req.txt` (recommended):

```cmd
conda create --name interval_env --file req.txt
conda activate interval_env
```

Or install core packages with pip (adjust versions as needed):

```cmd
pip install numpy pandas scipy seaborn matplotlib pennylane
```

Wrong versioning may cause issues with `pyinterval`. We report the versions we used for development and testing:

```cmd
pip install setuptools==69.2.0
pip install crlibm==1.0.3
pip install pyinterval
```

### Run verification
Each model folder contains a `verification_main.py` that runs verification for that model. To run verification for a model, run the corresponding script.

