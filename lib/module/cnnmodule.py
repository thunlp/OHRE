import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary


# def seed_torch(seed=42):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False


# embedding layers
class Embedding_word(nn.Module):

    def __init__(self, vocab_size, embedding_dim, weights, requires_grad=True):
        """
        the weights will be add one random vector for unk word
                add one zeros vector for blank word
                all weights should be trainable.
        :param vocab_size:
        :param embedding_dim:
        :param weights: the numpy version.
        :param trainable:
        """
        # if not random_init:
        #     seed_torch()
        super(Embedding_word, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size + 2, embedding_dim)
        unk_embedding = torch.nn.init.xavier_uniform_(torch.empty(1, embedding_dim)).cpu().numpy()
        blk_embedding = np.zeros((1, embedding_dim))

        weights = np.concatenate((weights, unk_embedding, blk_embedding), axis=0)

        # Variable(torch.)
        # print(type(self.word_embedding.weight))
        self.word_embedding.weight.data.copy_(torch.from_numpy(weights))
        self.word_embedding.weight.requires_grad = requires_grad

    def forward(self, idx_input):
        return self.word_embedding(idx_input)


def model_init(m):
    # if isinstance(m, nn.Embedding):
    #     nn.init.normal(m.weight)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        nn.init.normal_(m.bias, mean=0.5, std=1e-2)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        nn.init.normal_(m.bias, mean=0, std=2e-1)


class CNN(nn.Module):
    def __init__(self, cnn_input_shape):
        """

        :param cnn_input_shape: [max_len, word_embedding+2 * pos_emb]
        """
        super().__init__()
        # if not random_init:
        #     seed_torch()
        # add kernel regularizer in optimizer
        self.conv1 = nn.Conv1d(in_channels=cnn_input_shape[-1], out_channels=230, kernel_size=3)
        self.max_pool1 = nn.MaxPool1d(kernel_size=cnn_input_shape[0] - 2)

        # self.linear = nn.Linear(230, 128)
        # perfromance 64 > 128
        self.linear = nn.Linear(230, 64)
        # self.apply(model_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = x.reshape(-1, 230)

        # return F.sigmoid(self.linear(x))
        return self.linear(x)


if __name__ == '__main__':
    # test for embedding
    cnn = CNN((120, 60))
    # cnn.t()
    # print(cnn.linear.weight)
    # cnn.apply(print_weight)
    # cnn.apply(cnn_model_init)
    # cnn.apply(print_weight)
    # print(cnn.linear.weight)
    # todo should reshape to (embedding_len, max_len)
    for name, weight in cnn.named_parameters():
        print("name:", name)
        print("weight:", weight)

    # summary(cnn, input_size=(60, 120))
    # print(cnn)

    # a = Embedding_word(10, embedding_dim=100, weights=np.random.random((10, 100)))
    # b = a(Variable(torch.from_numpy(np.array([[0, 1, 2, 11, 11, 11]], dtype=np.int64))))
    # fake_label = torch.rand(size=(6, 100))
    # optimizer = torch.optim.Adam(a.parameters())
    # loss = torch.nn.MSELoss()
    # l = loss(b, fake_label)
    #
    # optimizer.zero_grad()
    # l.backward()
    # # test done!
    # a.word_embedding.weight.grad[-1] = 0
    # optimizer.step()
    #
    # print(a.word_embedding.weight)
