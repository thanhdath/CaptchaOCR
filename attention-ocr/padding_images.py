import cv2
import numpy as np
import os

INPUT_DIR = 'test/name/ori'
OUTPUT_DIR = 'test/name/pad'

height, width = 75, 225

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for file in os.listdir(INPUT_DIR):
    img = cv2.imread(INPUT_DIR+'/'+file)
    h, w, _ = img.shape
    new_width = min(int(height * w / h), width)
    img = cv2.resize(img, (new_width, height))
    new_img = np.zeros((height, width, 3))
    new_img[:img.shape[0], :img.shape[1], :] = img
    
    cv2.imwrite(OUTPUT_DIR+'/'+file, new_img)

