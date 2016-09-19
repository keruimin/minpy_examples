import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase

class RNNNet(ModelBase):
    def __init__(self,
                 batch_size=100,
                 input_size=1,  # input dimension
                 hidden_size=64,
                 num_classes=10):
        super(RNNNet, self).__init__()
        self.add_param(name='h0', shape=(batch_size, hidden_size))\
            .add_param(name='Wx', shape=(input_size, hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, hidden_size),
                       init_rule='constant', 
                       init_config={'value': np.identity(hidden_size)})\
            .add_param(name='b', shape=(hidden_size,),
                       init_rule='constant',
                       init_config={'value': np.zeros(hidden_size)})\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X, mode):
        seq_len = X.shape[1]
        self.params['h0'][:] = 0
        h = self.params['h0']
        for t in xrange(seq_len):
            h = layers.rnn_step(X[:, t, :], h, self.params['Wx'],
                         self.params['Wh'], self.params['b'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])

        return y

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)

class LSTMNet(ModelBase):
    def __init__(self,
                 batch_size=100,
                 input_size=1,  # input dimension
                 hidden_size=64,
                 num_classes=10):
        super(LSTMNet, self).__init__()
        self.add_param(name='h0', shape=(batch_size, hidden_size))\
            .add_param(name='c0', shape=(batch_size, hidden_size))\
            .add_param(name='Wx', shape=(input_size, 4*hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, 4*hidden_size))\
            .add_param(name='b', shape=(4*hidden_size,),
                       init_rule='constant',
                       init_config={'value': np.zeros(4*hidden_size)})\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X, mode):
        seq_len = X.shape[1]
        self.params['h0'][:] = 0
        self.params['c0'][:] = 0
        h = self.params['h0']
        c = self.params['c0']
        for t in xrange(seq_len):
            h, c = layers.lstm_step(X[:, t, :], h, c,
                                    self.params['Wx'],
                                    self.params['Wh'],
                                    self.params['b'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])
        return y

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)

class GRUNet(ModelBase):
    def __init__(self,
                 batch_size=100,
                 input_size=1,  # input dimension
                 hidden_size=64,
                 num_classes=10):
        super(GRUNet, self).__init__()
        self.add_param(name='h0', shape=(batch_size, hidden_size))\
            .add_param(name='Wx', shape=(input_size, 4*hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, 4*hidden_size))\
            .add_param(name='b', shape=(4*hidden_size,))\
            .add_param(name='Wxh', shape=(input_size, hidden_size))\
            .add_param(name='Whh', shape=(hidden_size, hidden_size))\
            .add_param(name='bh', shape=(hidden_size,))\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X, mode):
        seq_len = X.shape[1]
        self.params['h0'][:] = 0
        h = self.params['h0']
        for t in xrange(seq_len):
            h = layers.gru_step(X[:, t, :], h, self.params['Wx'],
                                self.params['Wh'], self.params['b'],
                                self.params['Wxh'], self.params['Whh'],
                                self.params['bh'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])
        return y

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)
