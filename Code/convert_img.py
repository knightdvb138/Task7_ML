import cv2
import os
import numpy as np
from sys import path
path.append('..')

path_human = 'human data for classification/Human'
path_non_human = 'human data for classification/Non-Human'

path_human_write = 'human data for classification/_Human'
path_non_human_write = 'human data for classification/_Non-Human'

for file in os.listdir(path_human):
    path_img_human = path_human+'/'+file
    image_human = cv2.imread(path_img_human, 0)
    image_human = cv2.resize(image_human, (64, 64),
                             interpolation=cv2.INTER_AREA)

    path_img_human_write = path_human_write + '/' + file
    cv2.imwrite(path_img_human_write, image_human)


for file in os.listdir(path_non_human):
    path_img_non_human = path_non_human+'/'+file
    image_non_human = cv2.imread(path_img_non_human, 0)
    image_non_human = cv2.resize(
        image_non_human, (64, 64), interpolation=cv2.INTER_AREA)

    path_img_non_human_write = path_non_human_write + '/' + file
    cv2.imwrite(path_img_non_human_write, image_non_human)