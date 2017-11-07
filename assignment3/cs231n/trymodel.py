#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from assignment3.cs231n.data_augmentation import *
from assignment3.cs231n.classifiers.convnet import *
from assignment3.cs231n.classifier_trainer import *

# input_shape = (3, 28, 28)
#
# def augment_fn(X):
#     out = random_flips(random_crops(X, input_shape[1:]))
#     out = random_tint(random_contrast(out))
#     return out
#
# def predict_fn(X):
#     return fixed_crops(X, input_shape[1:], 'center')
#
# model = init_three_layer_convnet(filter_size=5, input_shape=input_shape, num_filters=(32, 128))
# trainer = ClassifierTrainer()

# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, three_layer_convnet,
#           reg=0.05, learning_rate=0.00005, learning_rate_decay=1.0,
#           batch_size=50, num_epochs=30, update='rmsprop', verbose=True, dropout=0.6,
#           augment_fn=augment_fn, predict_fn=predict_fn)


from assignment3.cs231n.data_utils import load_models

models_dir = 'datasets\\tiny-100-A-pretrained'

# models is a dictionary mappping model names to models.
# Like the previous assignment, each model is a dictionary mapping parameter
# names to parameter values.
models = load_models(models_dir)
# best_model = models[0]
# test_feat = five_layer_convnet(extract_features=True)

