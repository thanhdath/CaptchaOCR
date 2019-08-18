import sys
import os
import glob
from tqdm import tqdm
import cv2
import re 
import argparse
from data_synthesize.synthesize_data import find_max_length, write_examples
import numpy as np

def get_filename(path):
    elms = re.split(r"[\\\/]", path)
    name = elms[-1].split('.')[0]
    return name

def build_char2idx(output_dir):
    names = list("0123456789abcdefghijklmnopqrstuvwxyz")
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

def load_charset(charset_file):
    char2idx = {}
    with open(charset_file) as fp:
        for line in fp:
            idx, char = line.split()
            char2idx[char.decode('utf8')] = int(idx)
    return char2idx

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data from png to TFRecord format.")
    parser.add_argument('--input_dir',  default='real_data',
                        help='Directory which will contains real data and labels.')
    parser.add_argument('--output_dir', default='data',
                        help="Directory which will contains data in tfrecord format.")
    parser.add_argument('--charset', default=None, help="Charset file or will be rebuild based on labels.")
    parser.add_argument('--max_length', default=None, help="", type=int)
    parser.add_argument('--w', default=225, type=int, help="Image width")
    parser.add_argument('--h', default=75, type=int, help="Image height")
    parser.add_argument('--step', default=10000, type=int,
                        help="Number of data in one tfrecord data file.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    images_path = glob.glob(args.input_dir+'/*.jpg')


    if args.charset:
        char2idx = load_charset(args.charset)
    else:
        char2idx = build_char2idx(args.output_dir)

    labels = {}

    print("Read labels")
    for image_path in tqdm(images_path):
        label = image_path.split('/')[-1].split('-')[0].decode('utf-8').lower()
        # label dateback
        if len(label) > 6:
            continue
        filename = get_filename(image_path)
        labels[filename] = label

    if args.max_length:
        max_length = args.max_length
    else:
        max_length = find_max_length(list(labels.values()))
    # max_length = 7
    print('Max length: ', max_length)
    image_data = []
    from_idx = 0
    for image_path in tqdm(images_path):
        label_key = get_filename(image_path)
        if label_key not in labels:
            continue
        name = labels[label_key]

        img = cv2.imread(image_path)
        h, w, _ = img.shape
        new_width = min(int(args.h * w / h), args.w)
        img = cv2.resize(img, (new_width, args.h))
        # img = aug.augment_image(img)
        new_img = np.zeros((args.h, args.w, 3))
        new_img[:img.shape[0], :img.shape[1], :] = img

        
        image_data.append([new_img, name.lower(), new_width])

        if len(image_data) == args.step:
            write_examples(image_data=image_data,
                           output_path=args.output_dir +
                           '/data_train_name-{}-to-{}'.format(from_idx,
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
                       '/data_train_name-{}-to-{}'.format(from_idx,
                                                         from_idx + len(image_data)),
                       charset=char2idx,
                       length=max_length,
                       null_char_id=char2idx['<nul>'],
                       num_of_views=1)
        
