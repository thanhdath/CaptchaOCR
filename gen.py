from captcha.image import ImageCaptcha
import glob
import random
import numpy as np
import os 
from tqdm import tqdm
import sys

if not os.path.isdir('fakedata'):
    os.makedirs('fakedata')

font_paths = glob.glob('font/*')

image = ImageCaptcha(fonts=font_paths)
# data = image.generate('1234')
N_IMAGES = int(sys.argv[1])
for idx in tqdm(range(N_IMAGES)):
    length_name = np.random.randint(4, 7)
    name = ''.join([random.choice('0123456789abcdefghijklmnopqrstuvwxyz'.upper())
        for _ in range(length_name)])
    image.write(name, 'fakedata/{}-{}.png'.format(name, idx))
