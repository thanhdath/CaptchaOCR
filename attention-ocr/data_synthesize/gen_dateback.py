#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from progressbar import progressbar
from synthesize_data import find_max_length, write_examples
import argparse
import pdb


def get_valid_coord(x, size):
    if x < 0:
        return 0
    if x > size:
        return size
    return x


def get_image(bg, bg_type, fontpath, text):
    if np.random.randint(2) == 1:
        text = ' '.join(list(text))
    img = bg.copy()
    font_size = random.choice(range(80, 105))
    fnt = ImageFont.truetype(fontpath, font_size, encoding="utf-8")
    text_size = fnt.getsize(text.replace('y', 'a').replace(
        'g', 'a').replace('p', 'a').replace('q', 'a'))
    if bg_type == "old":
        bot_left_y = random.choice(range(100, 250))
    else:
        bot_left_y = random.choice(range(100, 180))
    bot_left_x = random.choice(range(img.size[0] - text_size[0]))
    d = ImageDraw.Draw(img)
    d.text((bot_left_x, bot_left_y - text_size[1]), text, font=fnt, fill=(0, 0, 0))
    img = np.asarray(img)[:, :, ::-1]
    from_valid_coor_h = get_valid_coord(bot_left_y - text_size[1] - 20, img.shape[0])
    to_valid_coor_h = get_valid_coord(
        bot_left_y - text_size[1] + fnt.getsize(text)[1] + 20, img.shape[0])
    from_valid_coor_w = get_valid_coord(bot_left_x - 20, img.shape[1])
    to_valid_coor_w = get_valid_coord(
        bot_left_x + fnt.getsize(text)[0] + 20, img.shape[1])
    img = img[from_valid_coor_h:to_valid_coor_h, from_valid_coor_w:to_valid_coor_w]
    # plt.imshow(img)
    # plt.show()
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data from png to TFRecord format.")
    parser.add_argument('--output_dir', default='../datasets/data/dateback',
                        help="Directory which will contains data in tfrecord format.")
    parser.add_argument('--w', default=225, type=int, help="Image width")
    parser.add_argument('--h', default=75, type=int, help="Image height")
    parser.add_argument('--step', default=10000, type=int,
                        help="Number of data in one tfrecord data file.")
    parser.add_argument('--n_sample', type=int, default=10000,
                        help='Number of samples to be generated')
    return parser.parse_args()

from random import randrange
from datetime import datetime, timedelta


def random_date():
    # days 01-09 or 1-9 + 10-31
    # months 01-09 or 1-9 + 10-12
    # years 1900-2050 or 00-99
    kind = np.random.randint(3)
    if kind == 0:
        # random days
        day = np.random.randint(1, 32)
        if day < 10:
            if np.random.randint(2) == 1:
                day = "0" + str(day)
        return str(day), "day"
    elif kind == 1:
        # random months
        month = np.random.randint(1, 13)
        if month < 10:
            if np.random.randint(2) == 1:
                month = "0" + str(month)
        return str(month), "month"
    else:
        # random years
        year = np.random.randint(1900, 2051)
        if np.random.randint(2) == 1:
            year = str(year)[2:]
        return str(year), "year"


def build_char2idx(output_dir):
    charset = []
    for name in "0 1 2 3 4 5 6 7 8 9".split():
        charset.append(name)
    # add zero for null value
    print("Size of charset: {}".format(len(charset) + 1))

    char2idx = {}
    with open(output_dir + '/charset_size={}.txt'.format(len(charset) + 1), 'w+') as fp:
        char2idx['<nul>'] = 0
        fp.write('0\t<nul>\n')
        for idx, char in enumerate(charset):
            char2idx[char] = idx + 1
            fp.write('{}\t{}\n'.format(idx + 1, char.encode('utf8')))
    return char2idx


def read_names(file):
    names = []
    probs = []
    with open(file) as fp:
        for line in fp:
            if line.strip() != "":
                name, freq = line.split()
                freq = int(freq)
                names.append(name.decode('utf-8'))
                probs.append(freq)
    probs = np.array(probs, dtype=np.float32)
    probs = probs / probs.sum()
    return names, probs


def read_fonts(file):
    fonts = []
    probs = []
    with open(file) as fp:
        for line in fp:
            name, freq = line.split('\t')
            freq = int(freq)
            fonts.append('data/font/' + name)
            probs.append(freq)
    probs = np.array(probs, dtype=np.float32)
    probs = probs / probs.sum()
    return fonts, probs


if __name__ == '__main__':

    args = parse_args()
    print(args)

    if not os.path.exists(args.output_dir + '/sample_data'):
        os.makedirs(args.output_dir + '/sample_data')

    max_length = 4
    print("Max sequence length: {}".format(max_length))

    bg_dir = './data/bg-dateback/'
    fontpaths, font_probs = read_fonts('data/font.txt')

    bgs = {
        "old": {
            "day": [Image.open(bg_dir + '/dayback.jpg')],
            "month": [Image.open(bg_dir + '/monthback.jpg')],
            "year": [Image.open(bg_dir + '/yearback.jpg'), Image.open(bg_dir + '/yearback2.jpg')]
        },
        "new": {
            "day": [Image.open(bg_dir + '/dayback-new.jpg')],
            "month": [Image.open(bg_dir + '/monthback-new.jpg')],
            "year": [Image.open(bg_dir + '/yearback-new.jpg')]
        }
    }

    char2idx = build_char2idx(args.output_dir)

    image_data = []
    idx = 0
    from_idx = 0

    for _ in progressbar(range(args.n_sample)):
        fontpath = np.random.choice(fontpaths, size=1, p=font_probs)[0]
        name, kind = random_date()

        bg_type = random.choice(["old", "new"])
        bg = random.choice(bgs[bg_type][kind])

        img = get_image(bg, bg_type, fontpath, name)
        h, w, _ = img.shape
        new_width = min(int(args.h * w / h), args.w)
        img = cv2.resize(img, (new_width, args.h))
        # img = aug.augment_image(img)
        new_img = np.zeros((args.h, args.w, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img
        image_data.append([new_img, name.lower(), new_width])

        if np.random.randint(0, 1000) >= 1:
            # write some sample images to test
            cv2.imwrite("{}/sample_data/{}.png".format(args.output_dir,
                                                       name), new_img)

        if len(image_data) == args.step:
            write_examples(image_data=image_data,
                           output_path=args.output_dir +
                           '/gmo_train_name-{}-to-{}'.format(from_idx,
                                                             from_idx + len(image_data)),
                           charset=char2idx,
                           length=max_length,
                           null_char_id=char2idx['<nul>'],
                           num_of_views=1)
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
                       num_of_views=1)
