
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import data_provider
import datasets
import sys
import numpy as np
import PIL.Image

FLAGS = flags.FLAGS
common_flags.define()
FLAGS.dataset_name = 'namedataset'

def get_dataset_image_shape(dataset_name):
    ds_module = getattr(datasets, dataset_name)
    height, width, channel = ds_module.DEFAULT_CONFIG['image_shape']
    return height, width, channel


def create_model(dataset_name):
    height, width, channel = get_dataset_image_shape(dataset_name)
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)

    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)

    raw_images = tf.placeholder(tf.uint8, shape=[1, height, width, channel])
    images = tf.map_fn(data_provider.preprocess_image, raw_images, dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints

import timeit
import logging
logging.disable(logging.CRITICAL) 

def main():
    
    height, width, channel = get_dataset_image_shape(FLAGS.dataset_name)
    images_placeholder, endpoints = create_model(FLAGS.dataset_name)
    session_creator = monitored_session.ChiefSessionCreator(checkpoint_filename_with_path='/media/dont/data/dont/datasets/crnn/tfname/logs/model.ckpt-110665')
    image = PIL.Image.open(sys.argv[1]).convert('RGB')
    #image = image.resize((width, height), PIL.Image.ANTIALIAS)
    images_data = np.expand_dims(np.asarray(image),axis=0)
    sess = monitored_session.MonitoredSession(session_creator=session_creator)
    start = timeit.default_timer()
    predictions = sess.run(endpoints.predicted_text,
                           feed_dict={images_placeholder: images_data})
    print(predictions[0].decode('utf8'))
    stop = timeit.default_timer()
    print('Time : ', stop - start) 

if __name__ == '__main__':
    main()
