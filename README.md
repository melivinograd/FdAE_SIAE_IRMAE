# Reduced Representations of Turbulent Rayleigh-Bénard Flows via Autoencoders

This repository contains the models used for dimensionality reduction implemented in TensorFlow, applied to a Rayleigh-Bénard system. This repository accompanies the work outlined in the following paper: [Reduced Representations of Turbulent Rayleigh-Bénard Flows via Autoencoders](link). It features a model for a CNN Autoencoder with a fixed bottleneck size (FdAE, Fixed-d Autoencoder) and two regularizations: L1 regularization for the bottleneck layer (SIAE, Sparsity-inducing Autoencoder) and implicit regularization (IRMAE, Implicit Rank-Minimizing Autoencoder).

## Outline

### Files and Directories

- **model.py**:Contains the implementation of the dimensionality reduction models: FdAE, IRMAE, and SIAE.
- **train.py**: Includes the training scripts for the models.
- **args.json**: A configuration file containing the arguments and hyperparameters used for training and model configuration.
- **singular.py**: Handles the computation of the spectrum of singular values for IRMAE.
- **iqr.py**: Contains functions for computing the relevant dimensions of the SIAE.
- **utils.py**: Utility functions used throughout the project for data loading.
- **data/**: Directory containing the data used for the experiments. For more details, see the [data/README.md](data/README.md).

## Getting Started

### Prerequisites

To run the code in this repository, you will need:

- Python 3.x
- TensorFlow
- NumPy
- SciPy
- Matplotlib
- Additional dependencies listed in `requirements.txt`

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/melivinograd/FdAE_SIAE_IRMAE
cd FdAE_SIAE_IRMAE
pip install -r requirements.txt
```

### Usage
1. **Configure the model**: Modify the `args.json` file to set the desired parameters and hyperparameters for the model and training.

    - `d`: The dimensionality of the reduced space in the dimensionality reduction model, the bottleneck layer size.
    - `L1_lambda`: Parameter for SIAE. The regularization parameter for L1 regularization, used to encourage sparsity in the model weights. If set to zero, it is a regular FdAE.
    - `l`: Parameter for IRMAE. Number of linear layers to add at the end of the encoder. If set to zero, it is a regular FdAE.

2. **Train the model**: Run the `train.py` script to start the training process.
    ```bash
    python3 train.py
    ```

3. **Analyze the results**: Use `singular.py` and `iqr.py` to analyze the results and extract insights from the trained model.

### Data
The data used for this project is located in the `data/` directory. For more information on the data, refer to the `data/README.md` file.

### Citation
If you use this code or data in your research, please cite our paper.



