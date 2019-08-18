
# coding: utf-8

import random
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import numpy as np
from tqdm import tqdm
import json
import copy
import tensorflow as tf
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data from png to TFRecord format.")
    parser.add_argument('--output_dir', default='datasets/data/gmo',
                        help="Directory which will contains data in tfrecord format.")
    parser.add_argument('--raw_data_dir', default='train_data/',
                        help="Raw data directory which contains image in png format. Requires name of image file is its label.")
    parser.add_argument('--w', default=200, type=int, help="Image width")
    parser.add_argument('--h', default=50, type=int, help="Image height")
    parser.add_argument('--step', default=10000, type=int,
                        help="Number of data in one tfrecord data file.")
    return parser.parse_args()


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def encode_utf8_string(text, charset, length, null_char_id):
    char_ids_unpadded = []
    for i in text:
        char_ids_unpadded.append(charset[i])
    char_ids_padded = copy.copy(char_ids_unpadded)
    while len(char_ids_padded) < length:
        char_ids_padded.append(null_char_id)
    return char_ids_padded, char_ids_unpadded


def write_examples(image_data, output_path, charset, length, null_char_id, num_of_views):
    """
    image_data : list of [img, text]
    """
    writer = tf.python_io.TFRecordWriter(output_path)

    for img, text, real_image_width in tqdm(image_data):
        char_ids_padded, char_ids_unpadded = encode_utf8_string(
            text, charset, length, null_char_id)
        _, encoded_image = cv2.imencode('.png', img)
        encoded_image = encoded_image.tobytes()

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask[:,:real_image_width] = 1
        _, mask = cv2.imencode('.png', mask)
        mask = mask.tobytes()

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image/format': _bytes_feature(["png".encode('utf8')]),
                'image/encoded': _bytes_feature([encoded_image]),
                'image/class': _int64_feature(char_ids_padded),
                'image/unpadded_class': _int64_feature(char_ids_unpadded),
                'height': _int64_feature([img.shape[0]]),
                'width': _int64_feature([img.shape[1]]),
                'num_of_views': _int64_feature([num_of_views]),
                'orig_width': _int64_feature([img.shape[1] // num_of_views]),
                'image/text': _bytes_feature([text.encode('utf8')]),
                'mask': _bytes_feature([mask])
            }
        ))

        writer.write(example.SerializeToString())

    writer.close()
    return example.SerializeToString()


def build_char2idx(data_paths, output_dir):
    charset = set()
    for path, name in data_paths:
        for char in name:
            charset.add(char)
    print("Size of charset: {}".format(len(charset) + 1))  # add one for null value

    char2idx = {}
    with open(output_dir + '/charset_size={}.txt'.format(len(charset) + 1), 'w+') as fp:
        char2idx['<nul>'] = 0
        fp.write('0\t<nul>\n')
        for idx, char in enumerate(charset):
            char2idx[char] = idx + 1
            fp.write('{}\t{}\n'.format(idx + 1, char.encode('utf8')))
    return char2idx


def find_max_length(data_paths):
    return max([len(x) for x in data_paths])


def build_tfrecord_data(data_paths, char2idx, args):
    output_dir, width, height, step = args.output_dir, args.w, args.h, args.step
    max_length = find_max_length(data_paths)
    print("Max sequence length: {}".format(max_length))
    image_data = []

    from_idx = 0
    for path, name in tqdm(data_paths):
        img = cv2.imread(path)
        h, w, _ = img.shape
        new_width = min(int(height * w / h), width)
        img = cv2.resize(img, (new_width, height))
        new_img = np.zeros((height, width, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img
        image_data.append([new_img, name])

        if np.random.randint(0, 100) == 1:
            # write some sample images to test
            cv2.imwrite("{}/sample_data/{}.png".format(args.output_dir,
                                                       name.encode('utf8')), new_img)

        if len(image_data) == args.step:
            write_examples(image_data=image_data,
                           output_path=args.output_dir +
                           '/gmo_train_name-{}-to-{}'.format(from_idx,
                                                             from_idx + len(image_data)),
                           charset=char2idx,
                           length=max_length,
                           null_char_id=char2idx['<nul>'],
                           num_of_views=1,
                           real_image_width=new_width)
            from_idx += len(image_data)
            image_data = []

    if len(image_data) > 0:
        write_examples(image_data=image_data,
                       output_path=args.output_dir +
                       '/gmo_train_name-{}-to-{}'.format(from_idx,
                                                         from_idx + len(image_data)),
                       charset=char2idx,
                       length=max_length,
                       null_char_id=char2idx['<nul>'],
                       num_of_views=1,
                       real_image_width=new_width)


def process_data(args):
    data_paths = []
    for r, _, f in os.walk(args.raw_data_dir):
        for f_ in f:
            if 'train' in r and f_.endswith('.jpg'):
                name = f_.split('_')[0].decode('utf8').lower()
                data_paths.append([os.path.join(r, f_), name])

    print("Size of data: {}".format(len(data_paths)))

    char2idx = build_char2idx(data_paths, args.output_dir)
    build_tfrecord_data(data_paths, char2idx, args)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if not os.path.isdir(args.output_dir + '/sample_data'):
        os.makedirs(args.output_dir + '/sample_data')

    process_data(args)
