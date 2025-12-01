# cluster_maker
`cluster_maker` is a small educational Python package for simulating clustered
datasets and running clustering analyses with a simple, user-friendly interface.

It is designed for practicals and exams where students are given an incomplete
or faulty version of the package and asked to debug or extend it.

## Allowed libraries
The package only uses:

- Python standard library  
- NumPy  
- pandas  
- matplotlib  
- SciPy  
- scikit-learn  

No other third-party libraries are required.

## Main features
- Define a **seed DataFrame** describing cluster centres  
- Simulate clustered data around these centres  
- Compute basic **descriptive statistics** and **correlations**  
- Preprocess data: feature selection and standardisation  
- Run clustering with:
  - a simple **manual K-means** implementation  
  - a scikit-learn **KMeans** wrapper  
- Evaluate clustering with:
  - **inertia** (within-cluster sum of squares)  
  - **silhouette score**  
  - **elbow curve** for K selection  
- Plot:
  - 2D cluster scatter with optional centroids  
  - elbow curve  
- High-level **`run_clustering`** interface  
- Demo scripts and unit tests

## Package root directory structure
- `cluster_maker/`
  - `dataframe_builder.py` – build seed DataFrame and simulate clustered data  
  - `data_analyser.py` – descriptive statistics and correlation  
  - `data_exporter.py` – CSV and formatted text export  
  - `preprocessing.py` – feature selection and standardisation  
  - `algorithms.py` – manual K-means and scikit-learn KMeans wrapper  
  - `evaluation.py` – inertia, silhouette, elbow curve  
  - `plotting_clustered.py` – 2D cluster plots and elbow plots  
  - `interface.py` – high-level `run_clustering` function  
- `demo/` – example scripts  
- `data/` - csv data file used by the example scripts
- `tests/` – basic unit tests using the standard library `unittest`

## Installation (local use)
From the root directory of the project, run:

```bash
pip install -e .
```

This installs the package in editable mode, meaning you can modify the files
and re-run tests or demos without reinstalling.

## Notes on pyproject.toml and the *.egg-info directory
This project includes a small file named pyproject.toml.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
pip install -e . would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited.
