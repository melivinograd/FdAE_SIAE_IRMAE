import csv
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import model
import utils

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

def print_model_details(model):
    for layer in model.layers:
        print(f"Layer: {layer.name}")
        print(f"    Type: {type(layer)}")
        print(f"    Output Shape: {layer.output_shape}")
        print(f"    Trainable Weights: {len(layer.trainable_weights)}")
        for weight in layer.trainable_weights:
            print(f"        Weight Shape: {weight.shape}")
        print("")

def load_datasets(args):
    train_images, test_images = utils.load_data(
        args['data_path'], args['train_size'], args['test_size'], args['rolls'], args['ra']
    )
    return train_images.batch(args['batch_size']), test_images.batch(args['batch_size'])

def calculate_dataset_lengths(args, train_dataset, test_dataset):
    total_train = args["train_size"] * max(args["rolls"], 1)
    total_test = args["test_size"] * max(args["rolls"], 1)
    len_train = (total_train + args["batch_size"] - 1) // args["batch_size"]
    len_test = (total_test + args["batch_size"] - 1) // args["batch_size"]
    return len_train, len_test

def initialize_network(args):
    net = model.AE(args)
    optimizer = get_optimizer(args)
    input_shape = [args["batch_size"], args["imgs_shape"], args["imgs_shape"], 1]
    net(tf.zeros(input_shape))

    # For Encoder
    print("Encoder Details:")
    print_model_details(net.enc)

    # For Decoder
    print("Decoder Details:")
    print_model_details(net.dec)

    return net, optimizer

def get_optimizer(args):
    optimizers = {
        "adam": tf.keras.optimizers.Adam(args['lr']),
    }
    return optimizers.get(args["optimizer"], ValueError("Unsupported optimizer"))

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_csv_file():
    with open('performances.csv', 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)
        if out_file.tell() == 0:
            header = ['Epoch', 'L2 Loss', 'L1 Loss', 'Total Train Loss', 'Eval mse Loss', 'Eval L1 Loss', 'Total Eval Loss']
            csv_writer.writerow(header)

def update_csv_file(epoch, losses):
    with open('performances.csv', 'a', newline='') as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow([epoch] + list(losses))

def load_previous_model_if_needed(args, net, save_path):
    print('Loading old weights')
    if args['restart']:
        try:
            epochs = pd.read_csv("performances.csv")["Epoch"]
            old_epoch = epochs.iloc[-1]
            net.load_weights(save_path + f'{old_epoch}.weights.h5')
            print('Previous models loaded.')
            return old_epoch
        except Exception as e:
            print(f"Error loading previous model: {e}")
            return 0
    return 0

def compute_losses(net, dataset, len_dataset, training=False, optimizer=None):
    mse_loss, l1_loss, total_loss = 0.0, 0.0, 0.0
    for yi in tqdm(dataset):
        if training:
            with tf.GradientTape() as tape:
                mse, l1_z, total = net(yi)
        else:
            mse, l1_z, total = net(yi)

        mse_loss += mse.numpy()
        l1_loss += l1_z.numpy()
        total_loss += total.numpy()

        if training:
            gradients = tape.gradient(total, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))

    return mse_loss / len_dataset, l1_loss / len_dataset, total_loss / len_dataset

def train_epoch(net, train_dataset, optimizer, len_train):
    return compute_losses(net, train_dataset, len_train, training=True, optimizer=optimizer)

def validate_epoch(net, test_dataset, len_test, epoch, old_epoch, save_path):
    eval_losses = compute_losses(net, test_dataset, len_test)

    if epoch % 10 == 0 and epoch != 0:
        net.save_weights(os.path.join(save_path, f"{old_epoch + epoch}.weights.h5"))

    return eval_losses

def print_duration(start):
    duration = time.time() - start
    print(f"Duration: {duration / 3600:.2f} hours")
    print(f"Duration: {duration / 60:.2f} min")

def main(args):
    start = time.time()

    # Load data ################################################
    train_dataset, test_dataset = load_datasets(args)
    len_train, len_test = calculate_dataset_lengths(args, train_dataset, test_dataset)

    # Initialize network ######################################
    net, optimizer = initialize_network(args)

    save_path = args['checkpoint']
    create_directory(save_path)
    initialize_csv_file()

    old_epoch = load_previous_model_if_needed(args, net, save_path)
    print('Old epoch', old_epoch)

    if old_epoch == 0:
        net.save_weights(os.path.join(save_path, "initial.weights.h5"))
        print("Initial weights saved.")

    for e in range(1, args['epochs'] + 1):
        start_e = time.time()
        tf.print(f"Epoch {e}/{args['epochs']}")

        train_losses = train_epoch(net, train_dataset, optimizer, len_train)
        eval_losses = validate_epoch(net, test_dataset, len_test, e, old_epoch, save_path)

        update_csv_file(e + old_epoch, train_losses + eval_losses)

        tf.print(f'Time for training epoch {e} is {time.time() - start_e:.2f} sec')

    # Save last epoch
    net.save_weights(os.path.join(save_path, f"{old_epoch + args['epochs']}.weights.h5"))
    print_duration(start)

if __name__ == '__main__':
    with open('args.json', 'r') as f:
        args = json.load(f)

    main(args)

