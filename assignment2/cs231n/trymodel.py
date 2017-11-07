#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from assignment2.cs231n.classifier_trainer import ClassifierTrainer
from assignment2.cs231n.classifiers.convnet import *
from assignment2.cs231n.utils import get_CIFAR10
from assignment2.cs231n.data_utils import load_CIFAR10
import pickle


def get_CIFAR10_data(num_training=3000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets\\cifar10\\'
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
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    x_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)


# model = init_five_layer_convnet(input_shape=(3, 32, 32), num_classes=10,
#                                 filter_sizes=(5, 5, 3), num_filters=(32, 32, 64, 128))
#
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#                             X_train, y_train, X_val, y_val, model, five_layer_convnet,
#                             reg=5e-1, momentum=0.9, learning_rate=5e-4, batch_size=100,
#                             num_epochs=3, verbose=True, acc_frequency=50)


model = init_supercool_convnet(weight_scale=3e-2, bias_scale=0, filter_size=3)


trainer = ClassifierTrainer()
best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          X_train, y_train, X_val, y_val, model, supercool_convnet,
          reg=0.5, momentum=0.9, learning_rate=5e-5, batch_size=50, num_epochs=2, # change to 20 epochs
          verbose=True, acc_frequency=50)

with open('best_model1.pkl', 'wb') as f:
    pickle.dump(best_model, f)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc_history)
plt.plot(val_acc_history)
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()

scores = five_layer_convnet(X_test,best_model)
ypred = np.argmax(scores, axis=1)
test_acc = np.mean(ypred == y_test)

print('Test Accuracy under best model: ' , test_acc)
