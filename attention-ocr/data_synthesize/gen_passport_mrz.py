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
import rstr

augment_bg = iaa.Sequential([
    iaa.SomeOf((0, 1), [
      iaa.GammaContrast(per_channel=True, gamma=(0.5, 1.75)),
      iaa.SigmoidContrast(per_channel=True, gain=(5,15), cutoff=(0.0, 1.0))
    ]),
    iaa.SaltAndPepper(p=(0, 0.05)),
    iaa.Dropout(p=(0, 0.1)),
    iaa.SomeOf((1,3),[
        iaa.GaussianBlur(sigma=(1.2, 5)),
        iaa.AverageBlur(k=(2, 7)),
        iaa.MotionBlur(angle=(72, 288), k=(3,13))
    ]),
    
    iaa.AdditiveGaussianNoise(scale=0.01*255),
    iaa.Add((-30, 30)),
    iaa.AddElementwise((-20, 20)),
    iaa.MultiplyElementwise((0.9, 1.2), per_channel=True),
    iaa.ContrastNormalization((0.5, 1.5))
])

def get_valid_coord(x, size):
    if x < 0:
        return 0
    if x > size:
        return size
    return x


def get_image(bg, fontpath, text):
    img = bg.copy()
    img = np.asarray(img)[:,:,1:]
    if np.random.randint(0, 10) >= 3:
        img = augment_bg.augment_image(img)
    img = img[:,:,::-1]
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    font_size = random.choice(range(20, 25))
    fnt = ImageFont.truetype(fontpath, font_size, encoding="utf-8")
    text_size = fnt.getsize(text.replace('y', 'a').replace(
        'g', 'a').replace('p', 'a').replace('q', 'a'))
    bot_left_y = random.choice(range(80, img.size[1]-40))
    bot_left_x = random.choice(range(img.size[0] - text_size[0]))
    d = ImageDraw.Draw(img)

    font_color = np.random.randint(0,100)
    d.text((bot_left_x, bot_left_y - text_size[1]), text, font=fnt, fill=(font_color, font_color, font_color))
    img = np.asarray(img)[:, :, ::-1]
    from_valid_coor_h = get_valid_coord(bot_left_y - text_size[1] - 10, img.shape[0])
    to_valid_coor_h = get_valid_coord(
        bot_left_y - text_size[1] + fnt.getsize(text)[1] + 10, img.shape[0])
    from_valid_coor_w = get_valid_coord(bot_left_x - np.random.randint(10,25), img.shape[1])
    to_valid_coor_w = get_valid_coord(
        bot_left_x + fnt.getsize(text)[0] + np.random.randint(10,25), img.shape[1])
    img = img[from_valid_coor_h:to_valid_coor_h, from_valid_coor_w:to_valid_coor_w]
    # plt.imshow(img)
    # plt.show()
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data from png to TFRecord format.")
    parser.add_argument('--output_dir', default='../datasets/data/passport_mrz',
                        help="Directory which will contains data in tfrecord format.")
    parser.add_argument('--w', default=1125, type=int, help="Image width")
    parser.add_argument('--h', default=75, type=int, help="Image height")
    parser.add_argument('--step', default=5000, type=int,
                        help="Number of data in one tfrecord data file.")
    parser.add_argument('--n_sample', type=int, default=200000,
                        help='Number of samples to be generated')
    return parser.parse_args()


def build_char2idx(output_dir):
    charset = []
    for name in list("0123456789<abcdefghijklmnopqrstuvwxyz"):
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

from random import randrange
from datetime import datetime, timedelta


def random_firstline(personal_names, country_code_dict):
    chars = list("abcdefghijklmnopqrstuvwxyz".upper())
    firstline = "P"
    firstline += random.choice(chars + ["<"]*50)
    # choose which country will be used
    nations = [(nation_name, len(names)) for nation_name, names in personal_names.items()]
    # random with probability of the number of names
    ps = np.array([x[1] for x in nations], dtype=np.float32)**0.75
    ps = ps/ps.sum()
    nation = np.random.choice([x[0] for x in nations], p=ps)
    # random name
    names = personal_names[nation]
    length_name = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.44, 0.45, 0.08, 0.01])
    names = [np.random.choice(names) for _ in range(length_name)]
    first_name = names[0]
    # gen random name
    first_name = ''.join([random.choice(chars) for _ in names[0]])
    last_name = '<'.join([''.join([random.choice(chars) for _ in name]) for name in names[1:]])
    # last_name = ''.join([random.choice(chars) for _ in name for name in names[1:]])
    # FIXME
    # last_name = "<".join(names[1:])
    # country code, 95% keep, 5% change to other
    if np.random.randint(100) >= 95: # change 
        country_code = np.random.choice(country_code_dict['Other'])
    else:
        if type(country_code_dict[nation]) == list:
            country_code = np.random.choice(country_code_dict[nation])
        else:
            country_code = country_code_dict[nation]

    firstline += country_code + first_name + "<<" + last_name
    if len(firstline) <= 44:
        firstline += "<"*(44-len(firstline))
        return firstline
    else:
        return None

def random_secondline(personal_names, country_code_dict, passport_number_format):
    digits = list("0123456789")
    chars = list("abcdefghijklmnopqrstuvwxyz".upper())
    nations = [(nation_name, len(names)) for nation_name, names in personal_names.items()]
    # random with probability of the number of names
    ps = np.array([x[1] for x in nations], dtype=np.float32)**0.75
    ps = ps/ps.sum()
    nation = np.random.choice([x[0] for x in nations], p=ps)
    passport_number = rstr.xeger(np.random.choice(passport_number_format[nation]))
    
    # 1% change rule
    if np.random.randint(100) >= 99:
        passport_number = rstr.xeger(r'[a-z0-9<]{9}')
    # 4% remove last character
    if np.random.randint(100) >= 96:
        passport_number = passport_number[:-1] + "<"
    if len(passport_number) > 9:
        print('err number')
    secondline = passport_number
    secondline += random.choice(digits) # check gigit

    if type(country_code_dict[nation]) == list:
        nationality = np.random.choice(country_code_dict[nation])
    else:
        nationality = country_code_dict[nation]
    # 2% change nation
    if np.random.randint(100) >= 98:
        nationality = np.random.choice(country_code_dict['Other'])
    if len(nationality) < 3:
        nationality += "<"*(3-len(nationality))

    secondline += nationality # 
    secondline += random_date(start='1/1/1950', end='1/1/2020') # birthdate

    secondline += random.choice(digits) # check digit
    secondline += random.choice(["M", "F", "<"]) # sex
    secondline += random_date(start="1/1/1990", end='1/1/2030') # expiry

    secondline += random.choice(digits) # check digit


    personal_number = rstr.xeger('[a-z0-9]{'+str(np.random.randint(5, 15))+'}') 
    if len(personal_number) < 14:
        personal_number += "<"*(14-len(personal_number))
    secondline += personal_number # optional data
    secondline += random.choice(digits) # check digit
    secondline += random.choice(digits) # check digit

    secondline = secondline.upper()
    if len(secondline) > 44:
        print(secondline)
        return None
    return secondline

def random_passport(personal_names, country_code_dict, passport_number_format):
    if np.random.randint(2) == 0:
        return random_firstline(personal_names, country_code_dict)
    else:
        return random_secondline(personal_names, country_code_dict, passport_number_format)

def random_date(start='1/1/1900', end='1/1/2050'):
    start = datetime.strptime(start, '%d/%m/%Y')
    end = datetime.strptime(end, '%d/%m/%Y')
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    date = start + timedelta(seconds=random_second)
    date = "{}{:02d}{:02d}".format(str(date.year)[2:], date.month, date.day)
    return date


def load_personal_names():
    personal_names = {}
    for nation in "Chinese English French Japanese Korean Vietnamese Russian".split():
        personal_names[nation] = []
        with open('./data/passport_mrz/names_{}.txt'.format(nation)) as fp:
            for line in fp:
                line = line.strip()
                line = line.replace(" ", "<")
                personal_names[nation].append(line.upper())
    return personal_names

def load_nationality_codes():
    codes = []
    with open('./data/passport_mrz/nationality_dict.txt') as fp:
        for line in fp:
            line = line.strip().split('\t')
            codes.append(line[1].upper())
    return codes


if __name__ == '__main__':

    args = parse_args()
    print(args)

    if not os.path.exists(args.output_dir + '/sample_data'):
        os.makedirs(args.output_dir + '/sample_data')

    max_length = 44
    print("Max sequence length: {}".format(max_length))

    font_dir = './data/font'
    bg_dir = './data/bg/'
    fontpaths, font_probs = read_fonts('data/passport_mrz/font.txt')

    # bgs = [Image.open(os.path.join(bg_dir, i)) for i in os.listdir(bg_dir)]
    bgs = [Image.open("./data/bg-mrz.png")]
    char2idx = build_char2idx(args.output_dir)

    image_data = []
    idx = 0
    from_idx = 400000

    # 
    personal_names = load_personal_names()
    country_code_dict = {
        'Vietnamese': 'VNM',
        'Korean': 'PRK',
        'Japanese': 'JPN',
        'Chinese': ['CHN', 'TWN'],
        'English': ['GBR', 'GBD','GBN','GBO','GBP','GBS', 'USA'],
        'French': 'FRA',
        'Russian': 'RUS',
        'Other': load_nationality_codes()
    }

    # rules in https://help.symantec.com/
    passport_number_format = {
        # "Chinese": [r'\d{9}', r'[a-z]\d{8}', r'[a-z]{2}\d{8}'],
        "Chinese": [r'\d{9}', r'[a-z]\d{8}', r'[a-z]{2}\d{7}'],
        "English": [r'\d{9}'],
        "French": [r'\d{2}[a-z0-9]{2}[a-z0-9]{5}'],
        "Japanese": [r'[a-z]{2}\d{7}', r'[a-z]\d{8}'],
        # "Japanese": [r'[a-z]{2}\d{3}[a-z]\d{2}[a-z]\d', r'[a-z]{2}\d{4}[a-z]\d[a-z]\d', r'[a-z]\d{4}[a-z]\d{2}[a-z]\d',
        #     r'[a-z]\d{4}[a-z]\d{2}[a-z]{2}\d', r'[a-z]{2}\d{3}[a-z]\d{2}[a-z]{2}\d', r'[a-z]{2}\d{8}', r'[a-z]{2}\d{7}', 
        #     r'[a-z]\d{8}']
        "Korean": [r'[a-z]{2}\d{7}', r'[a-z]\d{8}', r'\d{9}'],
        "Vietnamese": [r'[a-z]\d{8}', r'[a-z]{2}\d{7}', r'[a-z]{6}<'],
        "Russian": [r'\d{9}']
    }

    for _ in progressbar(range(args.n_sample)):
        fontpath = np.random.choice(fontpaths, size=1, p=font_probs)[0]
        # random augment background

        bg = random.choice(bgs)
        name = random_passport(personal_names=personal_names, 
            country_code_dict=country_code_dict,
            passport_number_format=passport_number_format)

        if name is None: 
            continue
        try:
            img = get_image(bg, fontpath, name)
            # img = aug1.augment_image(img)
            # img = aug2.augment_image(img)
        except:
            continue
        h, w, _ = img.shape
        new_width = min(int(args.h * w / h), args.w)
        img = cv2.resize(img, (new_width, args.h))
        # img = aug.augment_image(img)
        new_img = np.zeros((args.h, args.w, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img
        image_data.append([new_img, name.lower(), new_width])

        if np.random.randint(0, 1000) >= 0:
            # write some sample images to test
            cv2.imwrite("{}/sample_data/{}.png".format(args.output_dir, fontpath.split('/')[-1].split('.')[0]) , new_img)
            idx += 1
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
