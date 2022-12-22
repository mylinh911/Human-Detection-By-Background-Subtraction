from utils import *
import numpy as np
import cv2
from tqdm import tqdm

def get_data_cnn(path_name):
	data = read_json(path_name)

	imgs = []
	labels = []

	for d in tqdm(data):
		label = get_label(d)
		img = cv2.imread(d)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# img = img.reshape(224,224)
		img = cv2.resize(img, (224, 224))
		labels.append(label)
		imgs.append(img)
	
	imgs = np.array(imgs)
	labels = np.array(labels)
	
	print(imgs.shape)
	print(labels.shape)

	return [imgs, labels]

if __name__ == '__main__':
    train_data = get_data_cnn('train.json')
    print(train_data[0].shape)
    print(train_data[1].shape)