import joblib
import mxnet as mx
import minpy.numpy as np
from minpy.nn.io import NDArrayIter
from minpy.nn.solver import Solver
from minpy.context import set_context, gpu
from models import RNNNet, LSTMNet, GRUNet

set_context(gpu(0))

BATCH = 50
INPUT_DIM = 7
HIDDEN_DIM = 64

data = joblib.load("data/mnist.dat")

_, dim = data["train_data"].shape
seq_len = dim / INPUT_DIM

mean = np.mean(data["train_data"], axis=0)
std = np.std(data["train_data"] - mean, axis=0)
data["train_data"] = (data["train_data"][:] - mean)/(std+1e-5)
data["test_data"] = (data["test_data"][:] - mean)/(std+1e-5)

train_iter = NDArrayIter(data["train_data"][:5000].reshape(5000, seq_len, INPUT_DIM),
                         data["train_label"][:5000],
                         batch_size=BATCH,
                         shuffle=True)

test_iter = NDArrayIter(data["test_data"][:1000].reshape(1000, seq_len, INPUT_DIM),
                        data["test_label"][:1000],
                        batch_size=BATCH,
                        shuffle=False)

model = RNNNet(batch_size=BATCH, input_size=INPUT_DIM, hidden_size=HIDDEN_DIM)

solver = Solver(model,
                train_iter,
                test_iter,
                num_epochs=100,
                init_rule='xavier',
                update_rule='rmsprop',
                optim_config={
                        'learning_rate': 0.0002,
                },
                verbose=True,
                print_every=10)

solver.init()
solver.train()
