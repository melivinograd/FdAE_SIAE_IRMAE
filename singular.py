import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import model

def main(old_epoch=0):
    with open('args.json', 'r') as f:
        args = json.load(f)

    # Load data
    train_images, test_images = utils.load_data(
        args['data_path'],
        args['train_size'],
        args['test_size'],
        args['rolls'],
        args['ra']
    )
    test_dataset = test_images.batch(args['batch_size'])

    if not old_epoch:
        old_epoch = pd.read_csv("performances.csv")["Epoch"].iloc[-1]

    # Load model
    net = model.AE(args)
    input_shape = [args["batch_size"], args["imgs_shape"], args["imgs_shape"], 1]
    net(tf.zeros(input_shape))

    net.load_weights(f"{args['checkpoint']}{old_epoch}.weights.h5")

    # Encoding
    z = [net.encode(yi) for yi in test_dataset]
    z = tf.concat(z, axis=0)

    # Singular Value Decomposition
    c = np.cov(z, rowvar=False)
    s, _, _ = tf.linalg.svd(c)
    s /= s[0]

    # Plot the spectrum
    fig, ax = plt.subplots()
    ax.semilogy(range(args['n']), s, marker="o")
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel(r"$\sigma_i$", fontsize=14)
    plt.show()

if __name__ == '__main__':
    main()



