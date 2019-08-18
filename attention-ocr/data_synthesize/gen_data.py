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


def get_image(bg, fontpath, text):
    img = bg.copy()

    append_commas = random.choice(range(10))
    if append_commas >= 8:
            text += ','

    upper_case = random.choice(range(2))
    if upper_case == 0:
        text = text[0].upper() + text[1:]

    font_size = random.choice(range(85, 97))
    fnt = ImageFont.truetype(fontpath, font_size, encoding="utf-8")
    text_size = fnt.getsize(text.replace('y', 'a').replace('g', 'a').replace('p', 'a').replace('q', 'a'))
    bot_left_y = random.choice(range(100, 130))
    bot_left_x = random.choice(range(img.size[0]-text_size[0]))
    d = ImageDraw.Draw(img)
    d.text((bot_left_x, bot_left_y - text_size[1]), text, font=fnt, fill=(0, 0, 0))
    img = np.asarray(img)[:,:,::-1]
    from_valid_coor_h = get_valid_coord(bot_left_y-text_size[1] - 20, img.shape[0])
    to_valid_coor_h = get_valid_coord(bot_left_y - text_size[1]+fnt.getsize(text)[1] + 20, img.shape[0])
    from_valid_coor_w = get_valid_coord(bot_left_x - 20, img.shape[1])
    to_valid_coor_w = get_valid_coord(bot_left_x+fnt.getsize(text)[0] + 20, img.shape[1])
    img = img[from_valid_coor_h:to_valid_coor_h,from_valid_coor_w:to_valid_coor_w]
    # plt.imshow(img)
    # plt.show()
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data from png to TFRecord format.")
    parser.add_argument('--output_dir', default='../datasets/data/gmo',
                        help="Directory which will contains data in tfrecord format.")
    parser.add_argument('--w', default=225, type=int, help="Image width")
    parser.add_argument('--h', default=75, type=int, help="Image height")
    parser.add_argument('--step', default=10000, type=int,
                        help="Number of data in one tfrecord data file.")
    parser.add_argument('--n_sample', type=int, default=10000,
                        help='Number of samples to be generated')
    return parser.parse_args()


def build_char2idx(names, output_dir):
    charset = set()
    for name in names:
        name.encode('utf8')
        for char in name.lower():
            charset.add(char)
    # add one for null value
    print("Size of charset: {}".format(len(charset) + 1))

    char2idx = {}
    with open(output_dir + '/charset_size={}.txt'.format(len(charset) + 1), 'w+') as fp:
        char2idx['<nul>'] = 0
        fp.write('0\t<nul>\n')
        for idx, char in enumerate(charset):
            char2idx[char] = idx + 1
            fp.write('{}\t{}\n'.format(idx + 1, char.encode('utf8')))
    return char2idx


def gen_house_number(n_sample=300):
    names = []
    probs = []
    chars = list(u"AĂÂBCDĐEÊFGHIJKLMNOÔƠPQRSTUƯVXYZ/////-----")
    nums = list(range(10))
    for i in range(n_sample):
        length = random.choice(range(1,10))
        address = ""
        while len(address) < length:
            if np.random.randint(2) == 0:
                address += random.choice(chars)
            else:
                address += str(random.choice(nums))
        names.append(address)
        probs.append(1)
    return names, probs

def gen_abrreviated_names(words, n_sample=50):
    names =[]
    probs =[]
    prefixs = "TP Q P T H TX TT X tp q p t h tx tt x".split()
    for i in range(n_sample):
        name = random.choice(prefixs) + "." + random.choice(words)
        names.append(name)
        probs.append(.5)
    return names, probs

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

    names2, probs2 = gen_house_number(n_sample=1000)
    names3, probs3 = gen_abrreviated_names(names, n_sample=1000)
    names += names2
    probs += probs2
    names += names3
    probs += probs3
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
            fonts.append('data/font/'+name)
            probs.append(freq)
    probs = np.array(probs, dtype=np.float32)
    probs = probs / probs.sum()
    return fonts, probs


if __name__ == '__main__':

    args = parse_args()
    print(args)

    if not os.path.exists(args.output_dir+'/sample_data'):
        os.makedirs(args.output_dir+'/sample_data')

    names, probs = read_names('./data/dict.txt')
    max_length = find_max_length(names)
    print("Max sequence length: {}".format(max_length))

    font_dir = './data/font'
    bg_dir = './data/bg/'
    fontpaths, font_probs = read_fonts('data/font.txt')
    bgs = [Image.open(os.path.join(bg_dir, i)) for i in os.listdir(bg_dir)]

    char2idx = build_char2idx(names, args.output_dir)

    image_data = []
    idx = 0
    from_idx = 0


    for _ in progressbar(range(args.n_sample)):
        bg = random.choice(bgs)
        fontpath = np.random.choice(fontpaths, size=1, p=font_probs)[0]
        name = np.random.choice(names, size=1, p=probs)[0]
        # name = names[idx]
        # idx = (idx+1) % len(names)
        img = get_image(bg, fontpath, name)

        h, w, _ = img.shape
        new_width = min(int(args.h * w / h), args.w)
        img = cv2.resize(img, (new_width, args.h))
        # img = aug.augment_image(img)
        new_img = np.zeros((args.h, args.w, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img
        image_data.append([new_img, name.lower(), new_width])

        if np.random.randint(0, 100) >= 0:
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
