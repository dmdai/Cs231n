#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from assignment2.cs231n.test1 import *
from mnist_loader import *
from assignment2.cs231n.classifiers.neural_net import *
from assignment2.cs231n.classifier_trainer import *
from assignment2.cs231n.vis_utils import *
import time

start = time.clock()

X_train, y_train, X_val, y_val, X_test, y_test = load_data_wrapper()

# def augment_fn(X):
#     N = X.shape[0]
#     X = X.reshape(N, 28, 28)
#     mask = np.random.randint(2, size=N)
#     out = np.zeros((N, 28, 28))
#     out[mask == 1] = X[mask == 1, :, ::-1]
#     out[mask == 0] = X[mask == 0]
#     out = out.reshape(N, 784)
#
#     return out

mean_image = np.mean(X_train, axis=0, keepdims=True)
X_train -= mean_image
X_test -= mean_image
X_val -= mean_image

# model = init_two_layer_model(784, 200, 10)
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, two_layer_net,
#           reg=1e-5, momentum=0.9, learning_rate=1e-1, batch_size=100, num_epochs=50, # change to 20 epochs
#           verbose=True, acc_frequency=500)


model_ae = init_twolayer_net(input_size=28*28, hidden_dim=100, weight_scale=1e-4, num_classes=28*28)
trainer_ae = ClassifierTrainer()
best_model_ae, loss_history, train_acc_history, val_acc_history = trainer_ae.train(
          X_train, y_train, X_val, y_val, model_ae, ae_twolayer_net,
          reg=1e-3, momentum=0.9, learning_rate=1e-2, batch_size=100, num_epochs=20, # change to 20 epochs
          verbose=True, acc_frequency=500, type='ae')


model_nn = init_twolayer_net(input_size=28*28, hidden_dim=100, weight_scale=1e-4, num_classes=10)
model_nn['W1'], model_nn['b1'] = best_model_ae['W1'], best_model_ae['b1']
trainer_nn = ClassifierTrainer()
best_model_nn, loss_history_nn, train_acc_history_nn, val_acc_history_nn = trainer_nn.train(
          X_train, y_train, X_val, y_val, model_nn, twolayer_net,
          reg=5e-5, momentum=0.9, learning_rate=3e-1, batch_size=100, num_epochs=20, # change to 20 epochs
          verbose=True, acc_frequency=500, type='nn')



# model = init_twolayer_net(input_size=28*28, hidden_dim=100, weight_scale=1e-4, num_classes=10)
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, twolayer_net,
#           reg=5e-5, momentum=0.9, learning_rate=3e-1, batch_size=100, num_epochs=10, # change to 20 epochs
#           verbose=False, acc_frequency=500)



# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_val = X_val.reshape(X_val.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#
# model = int_mnist_convnet(input_size=(1, 28, 28), weight_scale=1e-3)
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train, y_train, X_val, y_val, model, mnist_convnet,
#           reg=1e-5, momentum=0.9, learning_rate=1e-1, batch_size=100, num_epochs=10, # change to 20 epochs
#           verbose=True, acc_frequency=200)



plt.figure()
plt.subplot(2, 1, 1)
plt.plot(loss_history_nn)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc_history_nn)
plt.plot(val_acc_history_nn)
plt.ylim(0.9, 1.)
plt.legend(['Train', 'Val'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')



scores = twolayer_net(X_test, best_model_nn)
y_pred = np.argmax(scores, axis=1)
acc = np.mean(y_pred == y_test)

print('Test accuracy: %f. Take time: %f\n' %(acc, time.clock()-start))

plt.show()