import os
from utils import *
import random

paths = os.path.join('.', 'data')

img_paths = []

for p in os.listdir(paths):
    tmp = os.path.join(paths, p)
    imgs = os.listdir(tmp)
    for i in os.listdir(tmp):
        img_path = os.path.join(tmp, i)
        img_paths.append(img_path)


random.shuffle(img_paths)
train_length = int(len(img_paths)*0.8)
val_length = len(img_paths) - train_length

write_json('train.json', img_paths[:train_length])
write_json('val.json', img_paths[train_length:])

print("GEN DATA DONE !!")