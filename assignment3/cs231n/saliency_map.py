#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__coauthor__ = 'Daylock'

import pickle
import numpy as np
from matplotlib import pyplot as plt
from assignment3.cs231n.classifiers.convnet import *
from assignment3.cs231n.utils import get_CIFAR10
from assignment3.cs231n.data_utils import load_CIFAR10


from assignment3.cs231n.data_utils import load_tiny_imagenet, load_models

# tiny_imagenet_a = 'F:\\tinyImageNet\\tiny-imagenet-100-A'
#
# class_names, X_train, y_train, X_val, y_val, X_test, y_test = load_tiny_imagenet(tiny_imagenet_a)
#
# # Zero-mean the data
# mean_img = np.mean(X_train, axis=0)
# X_train -= mean_img
# X_val -= mean_img
# X_test -= mean_img

num_training = 3000
num_validation = 1000
num_test = 1000


cifar10_dir = 'datasets\\cifar10'
X_train, y_train, X_test, y_test = get_CIFAR10(cifar10_dir)

# Subsample the data
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Normalize the data: subtract the mean image
mean_images = np.mean(X_train, axis=0)
X_train -= mean_images
X_val -= mean_images
X_test -= mean_images

# Transpose so that channels come first
X_train = X_train.transpose(0, 3, 1, 2).copy()
X_val = X_val.transpose(0, 3, 1, 2).copy()
X_test = X_test.transpose(0, 3, 1, 2).copy()

mean_images = mean_images.transpose(2, 0, 1)


# Invoke the above function to get our data.
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)


# Load a pretrained model; it is a five layer convnet.
# models_dir = 'datasets\\tiny-100-A-pretrained'
# # model = load_models(models_dir)['model1']
# model = load_models(models_dir)
# print(model)

with open('best_model_2.pkl', 'rb') as f:
    model  = pickle.load(f, encoding='iso-8859-1')

# num_batches = 10
# N_val = X_val.shape[0]
# N_batches = int(N_val / num_batches)
# X_val_batches = np.array_split(X_val, num_batches)
# y_val_batches = np.array_split(y_val, num_batches)
#
# p = np.zeros((N_val, 10))
# for i in range(num_batches):
#     # probs = five_layer_convnet(X_val_batches[i], model, return_probs=True)
#     probs = supercool2_convnet(X_val_batches[i], model, return_probs=True)
#     p[i * N_batches : (i + 1) * N_batches] = probs
# y_val_pred = np.argmax(p, axis=1)
#
# correct_indices, = np.nonzero(y_val_pred == y_val)
#
# def show_image(img, rescale=False, add_mean=True):
#     img = img.copy()
#     if add_mean:
#         img += mean_images
#     img = img.squeeze()
#     if img.ndim == 3:
#         img = img.transpose(1, 2, 0)
#     if rescale:
#         low, high = np.min(img), np.max(img)
#         img = 255.0 * (img - low) / (high - low)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
#
# num_examples = 6
# class_idx = 5
# example_idxs = None
# class_indices, = np.nonzero(y_val == class_idx)
# example_idxs = np.intersect1d(class_indices, correct_indices)[:num_examples]
#
# dX = np.zeros((num_examples, 3, 32, 32))
#
# dX = supercool2_convnet(X_val[example_idxs], model, y_val[example_idxs], compute_dX=True)
#
# # Plot the images and their saliency maps.
# for i in range(num_examples):
#     # Visualize the image
#     plt.subplot(2, num_examples, i + 1)
#     show_image(X_val[example_idxs[i]])
#     # plt.title(class_names[y_val[example_idxs[i]]][0])
#     # Saliency map for the ith example image.
#     sal = np.zeros((32, 32))
#
#     sal = np.max(np.abs(dX[i]), axis=0)
#     # Visualize its saliency map.
#     plt.subplot(2, num_examples, num_examples + i + 1)
#     show_image(sal, rescale=True, add_mean=False)
#
# plt.show()







num_batches = 10
N_val = X_val.shape[0]
N_batches = int(N_val / num_batches)
X_val_sub = np.array_split(X_val, num_batches)
y_val_sub = np.array_split(y_val, num_batches)

X_val_feats = np.zeros((N_val, 512))
for i in range(num_batches):
    feats = supercool2_convnet(X_val_sub[0], model, extract_features=True)
    X_val_feats[i * N_batches : (i+1) * N_batches] = feats

N_train = X_train.shape[0]
N_batches = int(N_train / num_batches)
X_train_sub = np.array_split(X_train, num_batches)
y_train_sub = np.array_split(y_train, num_batches)

X_train_feats = np.zeros((N_train, 512))
for i in range(num_batches):
    feats = supercool2_convnet(X_train_sub[i], model, extract_features=True)
    X_train_feats[i * N_batches : (i+1) * N_batches] = feats

from assignment3.cs231n.classifiers.linear_classifier import *
from assignment3.cs231n.classifiers.k_nearest_neighbor import *

# knn = KNearestNeighbor()
# knn.train(X_train_feats, y_train)
# kk_y_val_pred = knn.predict(X_val_feats, k=50)
# print('KNN acuracy : %f' % np.mean(kk_y_val_pred == y_val))

sof = Softmax()
sof.train(X_train_feats.T, y_train, learning_rate=1e-2, reg=1e-3, num_iters=10000, batch_size=100, verbose=True)
y_train_pred = sof.predict(X_train_feats.T)
y_val_pred = sof.predict(X_val_feats.T)

# train_acc = np.mean(y_train == y_train_pred)
# val_acc = np.mean(y_val_pred == y_val)
# print (train_acc, val_acc)

from assignment3.cs231n.classifier_trainer import *

model_copy = {k: v.copy() for k, v in model.items()}

model_copy['W4'] = sof.W.T.copy().astype(model_copy['W4'].dtype)
model_copy['b4'] = np.zeros_like(model_copy['b4'])

trainer = ClassifierTrainer()
learning_rate = 1e-4
reg = 1e-7
dropout = 0.5
num_epochs = 5
finetuned_model = trainer.train(X_train, y_train, X_val, y_val,
                                model_copy, supercool2_convnet,
                                learning_rate=learning_rate, reg=reg, update='rmsprop',
                                dropout=dropout, num_epochs=num_epochs, verbose=True)[0]
