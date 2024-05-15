import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import model
import utils

def load_data(args):
    """Load the dataset based on the provided arguments."""
    print(f"Loading data with {args['rolls']} rolls")
    _, test_images = utils.load_data(
        args['data_path'], args['train_size'], args['test_size'], args['rolls'], args['ra']
    )
    test_dataset = test_images.batch(args['batch_size'])
    return test_dataset

def get_last_epoch():
    """Retrieve the last epoch number from the performances.csv file."""
    epochs = pd.read_csv("performances.csv")["Epoch"]
    return epochs.iloc[-1]

def encode(args, old_epoch=None):
    """Encode the test dataset using the autoencoder model."""
    test_dataset = load_data(args)

    if old_epoch is None:
        old_epoch = get_last_epoch()
        print(f"Running for last epoch {old_epoch}")

    # Load model
    net = model.AE(args)
    input_shape = [args["batch_size"], args["imgs_shape"], args["imgs_shape"], 1]
    net(tf.zeros(input_shape))

    print(f"Loading weights from epoch {old_epoch}")
    net.load_weights(os.path.join(args['checkpoint'], f'{old_epoch}.weights.h5'))

    z = [net.encode(yi) for yi in test_dataset]
    z = tf.concat(z, axis=0).numpy()
    return z

def compute_iqr(z):
    """Compute the Interquartile Range (IQR) and its normalized values for the encoded data."""
    z_T = np.transpose(z)
    Q1 = np.percentile(z_T, 25, axis=1)
    Q3 = np.percentile(z_T, 75, axis=1)
    IQR = Q3 - Q1
    normalized_IQR = IQR / np.max(IQR)
    return normalized_IQR

def plot_iqr(normalized_IQR, th, n):
    """Plot the IQR values and their distribution."""
    count = np.sum(normalized_IQR < th)
    print(f"There are {count} values less than {th}")
    print(f"There are {n - count} values greater than {th}")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    axes[0].hist(normalized_IQR)
    axes[0].axvline(th)
    axes[0].set_xlabel(r'Value of $z_i$')
    axes[0].set_ylabel(r'Counts')

    axes[1].plot(normalized_IQR, marker="o", ls="")
    axes[1].axhline(th, ls="--", label=f"Threshold: {th}")
    axes[1].set_xlabel('Dimension of Latent Layer')
    axes[1].set_ylabel(r'IQR / IQR$_{max}$')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def main(th=0.1, plot=True):
    with open('args.json', 'r') as f:
        args = json.load(f)

    z = encode(args)
    normalized_IQR = compute_iqr(z)
    n = args['d']

    if plot:
        plot_iqr(normalized_IQR, th, n)

if __name__ == '__main__':
    main(th=0.1)

