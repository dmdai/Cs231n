
import numpy as np
from assignment4.cs231n.rnn_layers import *


class TextRNN(object):

    def __init__(self, word_to_idx, wordvec_dim=128, hidden_dim=128, cell_type='rnn', dtype=np.float32):

        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Error about %s' %cell_type)

        self.word_to_idx = word_to_idx
        self.dtype = dtype
        self.cell_type = cell_type
        self.idx_to_word = {i: ch for i, ch in word_to_idx.items()}
        self.params = {}

        vocab_size = len(self.idx_to_word)

        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim) * 1e-2

        dim_mul = {'rnn': 1, 'lstm': 4}[self.cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim) * 1e-2 * np.sqrt(1. / wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim) * 1e-2 * np.sqrt(1. / hidden_dim)
        self.params['b'] = np.random.randn(dim_mul * hidden_dim)

        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size) * 1e-2 * np.sqrt(1. / hidden_dim)
        self.params['b_vocab'] = np.random.randn(vocab_size)

        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, captions):

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = np.ones_like(captions_out)

        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        x, cache0 = word_embedding_forward(captions_in, W_embed)
        h0 = np.zeros((x.shape[0], Wh.shape[0]))

        if self.cell_type is 'rnn':
            h, cache = rnn_forward(x, h0, Wx, Wh, b)
        else:
            h, cache = lstm_forward(x, h0, Wx, Wh, b)

        out, cache2 = temporal_affine_forward(h, W_vocab, b_vocab)

        loss, dout = temporal_softmax_loss(out, captions_out, mask, verbose=False)

        dh, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, cache2)

        if self.cell_type is 'rnn':
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, cache)
        else:
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh, cache)

        grads['W_embed'] = word_embedding_backward(dx, cache0)

        return loss, grads
