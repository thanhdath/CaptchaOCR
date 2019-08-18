"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
import numpy as np
import PIL.Image

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import datasets
import os
import data_provider
import re
import json

FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')
flags.DEFINE_string('labels', '',
                    'Path to labels file.')


def get_dataset_image_size(dataset_name):
    # Ideally this info should be exposed through the dataset interface itself.
    # But currently it is not available by other means.
    ds_module = getattr(datasets, dataset_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
    return width, height


def load_images(file_pattern, batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    paths = list(sorted([os.path.join(file_pattern, i) for i in os.listdir(file_pattern)]))
    images_actual_data = np.ndarray(shape=(len(paths), height, width, 3),
                                    dtype='uint8')
    # images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
    #                                 dtype='uint8')
    # for i in range(batch_size):
    for i, path in enumerate(paths):
        # path = file_pattern % i
        # print("Reading %s" % path)
        pil_image = PIL.Image.open(tf.gfile.GFile(path))
        # print(pil_image.size)
        images_actual_data[i, ...] = np.asarray(pil_image)
    return images_actual_data, paths


def create_model(batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)
    raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
    mask = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
    images = tf.map_fn(data_provider.preprocess_image, raw_images,
                       dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints


def run(checkpoint, batch_size, dataset_name, image_path_pattern):
    images_placeholder, endpoints = create_model(batch_size,
                                                 dataset_name)
    images_data, paths = load_images(image_path_pattern, batch_size,
                              dataset_name)
    session_creator = monitored_session.ChiefSessionCreator(
        checkpoint_filename_with_path=checkpoint)
    
    import time
    stime = time.time()
    with monitored_session.MonitoredSession(
            session_creator=session_creator) as sess:
        predictions = sess.run(endpoints.predicted_text,
                               feed_dict={images_placeholder: images_data})
    print('Running time: ', time.time()-stime)
    return predictions.tolist(), paths

def get_filename(path):
    elms = re.split("[\\\/]", path)
    filename = elms[-1]
    return filename

def main(_):
    labels = None 
    if FLAGS.labels:
        labels = json.load(open(FLAGS.labels))

    print("Predicted strings:")
    predictions, paths = run(FLAGS.checkpoint, FLAGS.batch_size, FLAGS.dataset_name,
                      FLAGS.image_path_pattern)

    n_correct = 0

    for i in range(len(paths)):
        actual = None
        if labels is not None:
            try:
                actual = labels[get_filename(paths[i])].encode('utf-8')
            except:
                actual = None

        if actual == predictions[i].replace("\xe2\x96\x91", ""):
            n_correct += 1
            print("predict: {}\tactual: {}\tx\t{}".format(predictions[i], actual, paths[i]))
        else:
            print("predict: {}\tactual: {}\t \t{}".format(predictions[i], actual, paths[i]))
    print('Overall sequence accuracy: {:.3f}%'.format(n_correct*100.0/len(predictions)))


if __name__ == '__main__':
    tf.app.run()
