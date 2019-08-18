# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to evaluate a trained Attention OCR model.

python convert_to_pb.py --split_name=train --dataset_name passport_mrz --checkpoint  checkpoint_passport_mrz/model.ckpt-281998

A simple usage example:
python eval.py
"""
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
import common_flags
import data_provider

import numpy as np
import sys
import subprocess

FLAGS = flags.FLAGS
common_flags.define()

def main(_):
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    raw_images = tf.placeholder(dtype=tf.uint8, shape=[1, 50, 750, 3])  # fix here (batch_size, height, width, channel)
    images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint)
        print(raw_images, endpoints.predicted_chars)
        tf.train.write_graph(sess.graph_def, '.', 'train.pbtxt')
        saver.save(sess, 'ckpt/model')

if __name__ == '__main__':
    app.run()
    ### Get freeze_graph.py from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
    # python freeze_graph.py --input_graph=train.pbtxt --input_checkpoint=checkpoint_passport_mrz/model.ckpt-281998 --input_binary=false --output_graph=frozen_graph.pb --output_node_names="AttentionOcr_v1/predicted_chars"
