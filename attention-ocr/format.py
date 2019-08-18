import cv2
import sys
import os
import numpy as np
from progressbar import progressbar

data_dir = sys.argv[1]

for i,f in progressbar(enumerate(os.listdir(data_dir))):
    path = os.path.join(data_dir, f)
    img = cv2.imread(path)
    # print(img.shape)
    # os.remove(path)

    # img = cv2.resize(img, (600, 150))
    # cv2.imwrite(path, img)

    h,w,_=img.shape
    new_w = int(150.0*w/h)
    if new_w >600:
        new_w = 600
    img = cv2.resize(img, (new_w,150))
    new_img = np.ones((150,600,3)) *255
    new_img[:img.shape[0],:img.shape[1],:] = img
    cv2.imwrite(os.path.join(data_dir,"{:02d}.png".format(i)), new_img)