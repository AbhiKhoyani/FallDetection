import os
import wandb
import argparse
import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from wandb.keras import WandbCallback
from types import SimpleNamespace
from utils import reproducibility
from load_dataset import load_dataset
from models import MLP, CNN, Transformer
from evaluate import evaluate

default_config = SimpleNamespace(
    project_name = 'Fall-Detection',
    model = 'Transformer',
    mlp_units = [1024,512,256,64,32,16],
    mlp_dropout = 0.3,
    cnn_units = [300,200,100],
    cnn_kernels = [9,7,5],
    cnn_strides = [1,1,1],
    cnn_dropout = 0.3,
    cnn_pooling = True,
    tx_headSize = 256,
    tx_noHeads = 4,
    tx_ffDims = 2,
    tx_blocks = 2,
    tx_mlp = [64],
    tx_dropout = 0,
    tx_mlp_dropout = 0.2,
    batch_size = 16,
    epochs = 10,
    lr = 1e-3,
)

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument("--project_name", type=str, default = default_config.project_name, help='Project name to log data in W&B.')
    argparser.add_argument('--ms', type=t_or_f, help='Use this to enable Mu-Sigma augmentation')
    argparser.add_argument('--smote', type=t_or_f, help='Use this to enable SMOTE Oversampling')
    argparser.add_argument('--model', type=str, default = default_config.model, choices=['MLP','CNN','Transformer'],\
                                                    help="Choice of model out of three:['MLP','CNN','Transformer']")
    
    # MLP related arguments
    argparser.add_argument('--mlp_units', type=list, default = default_config.mlp_units,help="number of nodes in MLP per layer, eg: [1,2,3,4]")
    argparser.add_argument('--mlp_dropout', type=float, default = default_config.mlp_dropout, help="Dropout factor")

    #CNN related arguments
    argparser.add_argument('--cnn_units', type=list, default = default_config.cnn_units, help="number of nodes in CNN per layer, eg: [1,2,3,4]")
    argparser.add_argument('--cnn_kernels', type=list, default = default_config.cnn_kernels, help="number of nodes in CNN per layer, eg: [1,2,3,4]")
    argparser.add_argument('--cnn_strides', type=list, default = default_config.cnn_strides, help="number of nodes in CNN per layer, eg: [1,2,3,4]")
    argparser.add_argument('--cnn_dropout', type=float, default = default_config.cnn_dropout, help="Dropout factor")
    argparser.add_argument('--cnn_pooling', type=t_or_f, default = default_config.cnn_pooling, help='Use this to disable MaxPoolingLayer in CNN')

    #Transformer related arguments
    argparser.add_argument('--tx_headSize', type=int, default = default_config.tx_headSize, help="Head Size in Tx MultiHeadAttention layer")
    argparser.add_argument('--tx_noHeads', type=int, default = default_config.tx_noHeads, help="No. of heads in Tx MultiHeadAttention layer")
    argparser.add_argument('--tx_ffDims', type=int, default = default_config.tx_ffDims, help="No. of filters in Tx Conv1D layer")
    argparser.add_argument('--tx_blocks', type=int, default = default_config.tx_blocks, help="No. of Tx block ")
    argparser.add_argument('--tx_mlp', type=list, default = default_config.tx_mlp, help="No. of MLP units/layers in Tx after Tx head:[1,2,3,4]")
    argparser.add_argument('--tx_dropout', type=int, default = default_config.tx_dropout, help="Dropout rate in Tx head")
    argparser.add_argument('--tx_mlp_dropout', type=int, default = default_config.tx_mlp_dropout, help="Dropout rate in Tx MLP layers")
    
    #model training arguments
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--earlyStop', type=t_or_f, help='Use this to enable Early stopping')
    
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def train(config):
    os.environ['WANDB_API_KEY'] = "66eab73e48530bb4c50b2a1b04abaed644303514"
    os.environ['WANDB_ENTITY']= "abhi_khoyani"

    run = wandb.init(project = config.project_name, job_type='training', config = config)
    reproducibility()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(config.ms, config.smote)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train[:,0]),
                                                        y = y_train[:,0])
    class_weights = {i:j for i,j in enumerate(class_weights)}

    model = None
    if config.model == 'Transformer':
        print('Loading Transformer model.')
        model = Transformer(3000)       #3000 is number of features
    elif config.model == 'CNN':
        print('Loading CNN model.')
        model = CNN()
    elif config.model == 'MLP':
        print('Loading MLP model.')
        model = MLP()

    updateLr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.001, patience = 25,
                                                    verbose = 1, min_delta = 0.001, min_lr = 1e-6 )
    callbacks = [WandbCallback(save_model = False), updateLr, ]

    if config.earlyStop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True))


    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr),
                metrics=["sparse_categorical_accuracy"])

    history = model.fit(X_train, y_train,
                        validation_data = (X_val, y_val),
                        epochs = config.epochs,
                        batch_size = config.batch_size,
                        class_weight = class_weights,
                        callbacks = callbacks)

    evaluate(model, X_test, y_test)
    run.finish()

if __name__ == "__main__":
    parse_args()
    train(default_config)