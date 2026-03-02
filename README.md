# scSAGA — single-cell SAmpled Gromov-Wasserstein Alignment framework

Scalable optimal transport-based method for multimodal integration of RNA-seq and ATAC-seq data.

## Installation

### Option 1: pip (recommended)

```sh
pip install scSAGA
```

### Option 2: from source

We recommend creating a conda environment first:

```sh
conda env create -n scmint -f environment.yml
conda activate scmint
pip install -e .
```

> **Note on PyTorch:** `pip install scSAGA` will install a CPU-only version of PyTorch by default. If you need GPU support, install PyTorch manually with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before installing scSAGA.

## Usage

Once installed, run the analysis from the command line:

```sh
scsaga config/input.yml
```

### Input YAML format

Create a YAML config file specifying your datasets and parameters. A template is provided in `config/input.yml`. More datasets can be added in the same format as needed.

```yaml
anchor: "rna1"

datasets:
  - name: "rna1"
    modality: "rna"
    counts: "/path/to/rna_normalized_counts.mtx"
    barcodes: "/path/to/rna_barcodes.txt"
    features: "/path/to/rna_features.txt"
    pca: "/path/to/rna_pca_50.txt"

  - name: "atac"
    modality: "atac"
    counts: "/path/to/atac_normalized_counts.mtx"
    barcodes: "/path/to/atac_barcodes.txt"
    features: "/path/to/atac_features.txt"
    pca: "/path/to/atac_pca_50.txt"

  # Add more datasets as needed:
  # - name: "rna2"
  #   modality: "rna"
  #   ...

output_dir: "/path/to/output_directory"

# sketch_size:   # Optional: downsample cells via geometric sketching

# --- Hyperparameters ---
s_shared_cells:   # Estimated number of shared cells across modalities
M_samples:        # Anchor pairs sampled per OT iteration
alpha:             # Update step size (0 to 1)
S_iterations:        # Number of SAGA iterations
gw_epsilon:      # Convergence threshold
gw_reg:            # Sinkhorn regularization strength
```

### Outputs

Results are saved to the directory specified by `output_dir`:

- `T_<dataset>_to_<anchor>.npy` — transport plan for each dataset pair
- `joint_embedding_2d.png` — PCA plot of the joint embedding
- `joint_embedding_2d.csv` — 2D coordinates for the joint embedding
- `saga_runtimes.txt` — timing breakdown and alignment scores

## Development

After cloning the repo:

```sh
git clone https://github.com/Swethasree-Bhattaram/scSAGA.git
cd scSAGA
conda env create -n scmint -f environment.yml
conda activate scmint
pip install -e .
```

The `-e` flag installs in editable mode, so changes to the source code take effect immediately without reinstalling.