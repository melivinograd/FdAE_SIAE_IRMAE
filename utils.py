import os
import tensorflow as tf

# Load data fuctions
def _parse_function(tfrecord):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    x = tf.io.parse_single_example(tfrecord, features)
    img = tf.io.decode_raw(x['image'], tf.float32)  # Decode as float32
    img = tf.reshape(img, [512, 512, 1])  # Reshape to the original dimensions
    return img

def load_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def load_data(data_path, train_size, test_size, rolls, ra):
    train_images = load_dataset(data_path + f'train_{train_size}_{rolls}_{ra}.tfrecords')
    test_images =  load_dataset(data_path +   f'test_{test_size}_{rolls}_{ra}.tfrecords')
    return train_images, test_images
