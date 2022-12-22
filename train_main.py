from cnn_model import *
from dataset import *
from train_cnn import *

CKTP_NAME = 'test'

if __name__ == '__main__':
    model = my_model((224,224,1),2)
    print("GET MODEL DONE!!!")

    train_data = get_data_cnn('train.json')
    val_data = get_data_cnn('val.json')
    print("GET DATA DONE!!")

    print(train_data[0].shape)
    print(train_data[1].shape)

    trainer(model, train_data, val_data, CKTP_NAME, epochs=100)