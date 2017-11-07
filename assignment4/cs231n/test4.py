


# import time, os, json
import numpy as np
import matplotlib.pyplot as plt

# from assignment4.cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
# from assignment4.cs231n.rnn_layers import *
from assignment4.cs231n.captioning_solver import CaptioningSolver
# from assignment4.cs231n.classifiers.rnn import CaptioningRNN
# from assignment4.cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
# from assignment4.cs231n.image_utils import image_from_url
from assignment4.cs231n.classifiers.rnn_text import *

# def rel_error(x, y):
#   """ returns relative error """
#   return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#
# data = load_coco_data(pca_features=True)
#
# small_data = load_coco_data(max_train=2000)
#
# small_lstm_model = CaptioningRNN(
#           cell_type='lstm',
#           word_to_idx=data['word_to_idx'],
#           input_dim=data['train_features'].shape[1],
#           hidden_dim=512,
#           wordvec_dim=256,
#           dtype=np.float32,
#         )
#
# small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,
#            update_rule='nestro',
#            num_epochs=10,
#            batch_size=40,
#            optim_config={
#              'learning_rate': 1e-2,
#            },
#            lr_decay=0.995,
#            verbose=True, print_every=10,
#          )
#
# small_lstm_solver.train()
#
# # Plot the training losses
# plt.plot(small_lstm_solver.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training loss history')
# plt.show()
#
# minibatch = sample_coco_minibatch(small_data, split='val', batch_size=20)
# gt_captions, features, urls = minibatch
# gt_captions = decode_captions(gt_captions, data['idx_to_word'])
#
# sample_captions = small_lstm_model.sample(features)
# sample_captions = decode_captions(sample_captions, data['idx_to_word'])
#
# for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
#     plt.imshow(image_from_url(url))
#     plt.title('%s\n%s\nGT: %s' %('val', sample_caption, gt_caption))
#     plt.axis('off')
#     plt.show()


with open('datasets\\input.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('Data has %d characters, %d unique' %(data_size, vocab_size))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


captions = [char_to_idx[ch] for ch in data]


small_lstm_model = TextRNN(
          cell_type='lstm',
          word_to_idx=char_to_idx,
          hidden_dim=100,
          wordvec_dim=64,
          dtype=np.float32,
        )

small_lstm_solver = CaptioningSolver(small_lstm_model, captions,
           update_rule='nestro',
           num_epochs=1,
           batch_size=100,
           optim_config={
             'learning_rate': 8e-1,
           },
           lr_decay=0.995,
           verbose=True, print_every=10,
         )

small_lstm_solver.train()



