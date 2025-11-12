# scSAGA -- single-cell SAmpled Gromov wAsserstein framework

Scalable optimal transport based method for multimodal integration of RNA-seq and ATAC-seq data

## Installation

To install locally, we recommend to create a virtual environment using
conda ([miniforge](https://github.com/conda-forge/miniforge) for a faster
installation) as below:

```sh
conda config --add channels conda-forge
conda create -n scmit python=3.13
conda activate cerebrum
```

To install the software into the environment created above,
the repo can be cloned as follows.

```sh
git clone https://github.com/Swethasree-Bhattaram/scSAGA.git
```

## Usage

Commandline to run the analysis is as follows

```sh
    python -m scmint.scsaga config/input.yml
```

Input yaml should have the following format:
More datasets can be added in the same format as needed. 

```yaml
anchor: "rna1"

datasets:
  - name: "rna1"
    modality: "rna"
    counts: "/PATH/TO/rna_normalized_counts.mtx"
    barcodes: "/PATH/TO/rna_barcodes.txt"
    features: "/PATH/TO/rna_features.txt"
    pca: "/PATH/TO/rna_pca_50.txt"

  - name: "atac1"
    modality: "atac1"
    counts: "/PATH/TO/atac_normalized_counts.mtx"
    barcodes: "/PATH/TO/atac_barcodes.txt"
    features: "/PATH/TO/atac_features.txt"
    pca: "/PATH/TO/atac_pca_50.txt"
```

## Development Environment

Development environment requires python3.13+ environment.
After cloning the git repository, we can create a virtual environment in conda
using the `environment.yml` file.

```sh
    conda create env -n scmint -f environment.yml
    conda activate scmint
```

`environment.yml` includes the dependency packages.
If the path to the code is in directory `$HOME/scmint`, the code from the
directory can be run as follows:

```bash
env PYTHONPATH=$PYTHONPATH:$HOME/scmint python -m scmint.cli --help
```
