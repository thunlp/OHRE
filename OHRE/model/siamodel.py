import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import sys
# from torch.distributions.kl import kl_divergence
# from torch.distributions.bernoulli import Bernoulli
import torch.distributions as dist

sys.path.append(os.path.abspath('../../'))
from module.cnnmodule import model_init, Embedding_word, CNN

from .triplet_loss import TripletSemihardLoss, TripletLoss
from kit.utils import cudafy

NEAR_0 = 1e-10


# def seed_torch(seed=42):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False


# torch.nn.TripletMarginLoss
class RSN(nn.Module):
    def __init__(self, word_vec_mat, max_len=120, pos_emb_dim=5, dropout=0.2):
        super(RSN, self).__init__()
        # if not random_init:
        #     seed_torch()
        self.batch_shape = (3, max_len)
        # defining layers
        dict_shape = word_vec_mat.shape
        self.word_emb = Embedding_word(dict_shape[0], dict_shape[1], weights=word_vec_mat, requires_grad=True)
        # default trainable.
        self.pos1_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.pos2_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.drop = nn.Dropout(p=dropout)

        cnn_input_shape = (max_len, dict_shape[1] + 2 * pos_emb_dim)
        self.convnet = CNN(cnn_input_shape)

        # self.p = nn.Linear(230 * cnn_input_shape[-1], 1)

        # for name, weight in self.named_parameters():
        #     print("name:", name)
        #     print("weight:", weight)
        # exit()
        # defining optimizer
        # self.apply(model_init)
        lr = 1e-4
        params = []
        params += [{'params': self.word_emb.parameters(), 'lr': lr}]
        params += [{'params': self.pos1_emb.parameters(), 'lr': lr}]
        params += [{'params': self.pos2_emb.parameters(), 'lr': lr}]
        params += [{'params': list(self.convnet.conv1.parameters())[0], 'weight_decay': 2e-4, 'lr': lr}]
        # params += [{'params': list(self.convnet.conv1.parameters())[0]}]
        params += [{'params': list(self.convnet.conv1.parameters())[1], 'lr': lr}]

        params += [{'params': list(self.convnet.linear.parameters())[0], 'weight_decay': 1e-3, 'lr': lr}]
        # params += [{'params': list(self.convnet.linear.parameters())[0]}]
        params += [{'params': list(self.convnet.linear.parameters())[1], 'lr': lr}]
        # params += [{'params': self.p.parameters(), 'lr': lr}]

        self.optimizer = optim.Adam(params)

    def set_train_op(self, batch_size=50, train_loss_type=None,
                     testset_loss_type=None, p_denoise=1.0, margin=1.0,
                     p_cond=0.05, lambda_s=1, p_mult=0.02, squared=True):
        self.batch_size = batch_size
        self.train_loss_type = train_loss_type
        self.testset_loss_type = testset_loss_type
        self.p_denoise = p_denoise
        self.p_cond = p_cond
        self.p_mult = p_mult
        self.margin = margin
        self.lambda_s = lambda_s
        self.squared = squared

    def forward(self, batch_input, perturbation=None):
        pos1 = batch_input[:, 0, :]
        pos2 = batch_input[:, 1, :]
        word = batch_input[:, 2, :]

        pos1_emb = self.pos1_emb(pos1)
        pos2_emb = self.pos2_emb(pos2)
        word_emb = self.word_emb(word)

        drop = self.drop(word_emb)
        if perturbation is not None:
            drop += perturbation

        cnn_input = torch.cat([drop, pos1_emb, pos2_emb], -1)
        cnn_input = cnn_input.permute([0, 2, 1])  # [B, embedding, max_len]
        encoded = self.convnet(cnn_input)

        return word_emb, encoded

    def forward_norm(self, batch_input, pertubation=None):
        word_emb, encoded = self.forward(batch_input, pertubation)
        encoded = self.norm(encoded)
        return word_emb, encoded

    def norm(self, encoded):
        return F.normalize(encoded, p=2, dim=1)

    def siamese_forward(self, left_input, right_input, left_perturbation=None, right_perturbation=None):
        left_word_emb, encoded_l = self.forward(left_input, left_perturbation)
        right_word_emb, encoded_r = self.forward(right_input, right_perturbation)
        both = torch.abs(encoded_l - encoded_r)
        if not self.train_loss_type.startswith("triplet"):
            prediction = self.p(both)
            prediction = F.sigmoid(prediction)
        else:
            encoded_r = self.norm(encoded_r)
            encoded_l = self.norm(encoded_l)
            distances_squared = torch.sum(torch.pow(encoded_r - encoded_l, 2), dim=1)
            if self.squared:
                prediction = distances_squared
            else:
                prediction = distances_squared.sqrt()

        return prediction, left_word_emb, right_word_emb, encoded_l, encoded_r

    def back_propagation(self, loss):
        self.optimizer.zero_grad()

        # print(self.pos1_emb.weight.detach().numpy())
        loss.backward()
        # no gradient for 'blank' word
        self.word_emb.word_embedding.weight.grad[-1] = 0
        self.optimizer.step()

    def get_triplet_semihard_loss(self, encoded, labels, margin=1.0):
        # original # some problem
        criterion = TripletSemihardLoss()
        # print(margin)
        loss = criterion(encoded, labels, margin, squared=self.squared)
        # 1
        # nn.TripletMarginLoss()
        # 2
        # criterion = TripletLoss(margin=0.3)
        # loss = criterion(encoded, labels)
        return loss

    def get_cross_entropy_loss(self, prediction, labels):
        labels_float = labels.float()
        loss = -torch.mean(
            labels_float * torch.log(prediction + NEAR_0) + (1 - labels_float) * torch.log(1 - prediction + NEAR_0))
        return loss

    # nn.CrossEntropyLoss
    def get_cond_loss(self, prediction):
        cond_loss = -torch.mean(
            prediction * torch.log(prediction + NEAR_0) + (1 - prediction) * torch.log(1 - prediction + NEAR_0))

        return cond_loss

    # original vat loss
    def get_v_adv_loss(self, ul_left_input, ul_right_input, p_mult, power_iterations=1):
        bernoulli = dist.Bernoulli
        prob, left_word_emb, right_word_emb = self.siamese_forward(ul_left_input, ul_right_input)[0:3]
        prob = prob.clamp(min=1e-7, max=1. - 1e-7)
        prob_dist = bernoulli(probs=prob)
        # generate virtual adversarial perturbation
        left_d = cudafy(torch.FloatTensor(left_word_emb.shape).uniform_(0, 1))
        right_d = cudafy(torch.FloatTensor(right_word_emb.shape).uniform_(0, 1))
        left_d.requires_grad, right_d.requires_grad = True, True
        # prob_dist.requires_grad = True
        # kl_divergence
        for _ in range(power_iterations):
            left_d = (0.02) * F.normalize(left_d, p=2, dim=1)
            right_d = (0.02) * F.normalize(right_d, p=2, dim=1)
            # d1 = dist.Categorical(a)
            # d2 = dist.Categorical(torch.ones(5))
            p_prob = self.siamese_forward(ul_left_input, ul_right_input, left_d, right_d)[0]
            p_prob = p_prob.clamp(min=1e-7, max=1. - 1e-7)
            # torch.distribution
            try:
                kl = dist.kl_divergence(prob_dist, bernoulli(probs=p_prob))
            except:
                wait = True
            left_gradient, right_gradient = torch.autograd.grad(kl.sum(), [left_d, right_d], retain_graph=True)
            left_d = left_gradient.detach()
            right_d = right_gradient.detach()
        left_d = p_mult * F.normalize(left_d, p=2, dim=1)
        right_d = p_mult * F.normalize(right_d, p=2, dim=1)
        # virtual adversarial loss
        p_prob = self.siamese_forward(ul_left_input, ul_right_input, left_d, right_d)[0].clamp(min=1e-7, max=1. - 1e-7)
        v_adv_losses = dist.kl_divergence(prob_dist, bernoulli(probs=p_prob))
        return torch.mean(v_adv_losses)

    def get_triplet_v_adv_loss(self, ori_triplet_loss, ori_word_emb, batch_input, labels, margins, p_mult,
                               power_iterations=1):
        # bernoulli = dist.Bernoulli
        batch_d = cudafy(torch.FloatTensor(ori_word_emb.shape).uniform_(0, 1))
        batch_d.requires_grad = True
        for _ in range(power_iterations):
            batch_d = (0.02) * F.normalize(batch_d, p=2, dim=1)
            p_encoded = self.forward_norm(batch_input, batch_d)[1]
            triplet_loss = self.get_triplet_semihard_loss(p_encoded, labels, self.margin)
            batch_gradient = torch.autograd.grad(triplet_loss, batch_d, retain_graph=True)[0]
            batch_d = batch_gradient.detach()
            # left_d = left_gradient.detach()
            # right_d = right_gradient.detach()
        # worst
        batch_d = p_mult * F.normalize(batch_d, p=2, dim=1)
        p_encoded = self.forward_norm(batch_input, batch_d)[1]
        # triplet_loss = self.get_triplet_semihard_loss(p_encoded, labels, self.margin)

        triplet_loss = self.get_triplet_semihard_loss(p_encoded, labels, margins)
        # squared
        v_adv_loss = (triplet_loss - ori_triplet_loss).pow(2)
        # abs (better)
        # v_adv_loss = torch.abs(triplet_loss - ori_triplet_loss)

        return v_adv_loss

    def get_triplet_distance_v_adv_loss(self, ori_encoded, ori_word_emb, batch_input, p_mult, power_iterations=1):
        batch_d = cudafy(torch.FloatTensor(ori_word_emb.shape).uniform_(0, 1))
        batch_d.requires_grad = True
        criterion = TripletSemihardLoss()
        batch_size = batch_input.shape[0]
        ori_dist_mat = criterion.pairwise_distance(ori_encoded)
        for _ in range(power_iterations):
            batch_d = (0.02) * F.normalize(batch_d, p=2, dim=1)
            p_encoded = self.forward_norm(batch_input, batch_d)[1]

            p_dist_mat = criterion.pairwise_distance(p_encoded)
            # abs:
            # temp_loss = torch.abs(ori_dist_mat - p_dist_mat).sum()
            # squared:(better)
            temp_loss = (ori_dist_mat - p_dist_mat).pow(2).sum()

            batch_gradient = torch.autograd.grad(temp_loss, batch_d, retain_graph=True)[0]
            batch_d = batch_gradient.detach()

        # worst
        batch_d = p_mult * F.normalize(batch_d, p=2, dim=1)
        p_encoded = self.forward_norm(batch_input, batch_d)[1]
        # distance_loss:

        p_dist_mat = criterion.pairwise_distance(p_encoded)

        # squared
        v_adv_loss = (ori_dist_mat - p_dist_mat).pow(2).mean() * (batch_size / (batch_size - 1))
        # abs
        # v_adv_loss = torch.abs(ori_dist_mat - p_dist_mat).mean()

        return v_adv_loss

    def load_model(self, model_path=None):
        self.load_state_dict(torch.load(model_path))
        print('model loaded from ' + model_path)

    def save_model(self, save_path, global_step):
        torch.save(self.state_dict(), os.path.join(save_path, 'RSN') + "step_" + str(global_step) + ".pt")
        torch.save(self.state_dict(), os.path.join(save_path, 'RSN') + "best.pt")

    def train_RSN(self, dataloader_trainset, dataloader_testset, batch_size=None, same_ratio=0.06):
        if batch_size is None:
            batch_size = self.batch_size
        self.train()

        data_left, data_right, data_label = dataloader_trainset.next_batch(batch_size, same_ratio=same_ratio)
        data_left, data_right, data_label = cudafy(data_left), cudafy(data_right), cudafy(data_label)
        prediction, left_word_emb, right_word_emb, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        if self.train_loss_type == "cross":
            loss = self.get_cross_entropy_loss(prediction, labels=data_label)
        elif self.train_loss_type == "cross_denoise":
            loss = self.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.get_cond_loss(prediction) * self.p_denoise
        elif self.train_loss_type == "v_adv":
            loss = self.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.get_v_adv_loss(data_left, data_right, self.p_mult) * self.lambda_s
        elif self.train_loss_type == "v_adv_denoise":
            loss = self.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.get_v_adv_loss(data_left, data_right, self.p_mult) * self.lambda_s
            loss += self.get_cond_loss(prediction) * self.p_denoise
        else:
            raise NotImplementedError()

        self.back_propagation(loss)

    def train_triplet_loss(self, dataloader_trainset, batch_size=None, dynamic_margin=True, K_num=4,
                           unbalanced_batch=False):
        if batch_size is None:
            batch_size = self.batch_size
        self.train()
        torch.retain_graph = True

        batch_input, data_label, margins = dataloader_trainset.next_triplet_batch(batch_size, K_num, dynamic_margin,
                                                                                  unbalanced_batch)
        batch_input, data_label, margins = cudafy(batch_input), cudafy(data_label), cudafy(margins)
        margins = margins * self.margin if dynamic_margin else self.margin
        # margins = margins if dynamic_margin else self.margin

        word_embed, encoded = self.forward_norm(batch_input)
        loss = self.get_triplet_semihard_loss(encoded, data_label, margins)
        if self.train_loss_type == 'triplet_v_adv':
            get_triplet_v_adv_loss = self.get_triplet_v_adv_loss(loss, word_embed, batch_input, data_label, margins,
                                                                 self.p_mult)
            loss += get_triplet_v_adv_loss
        elif self.train_loss_type == 'triplet_v_adv_distance':
            get_triplet_v_adv_loss = self.get_triplet_distance_v_adv_loss(encoded, word_embed, batch_input, self.p_mult)
            loss += get_triplet_v_adv_loss

        self.back_propagation(loss)

    def train_triplet_same_level(self, dataloader_trainset, batch_size=None, dynamic_margin=True, K_num=4, level=1,
                                 same_v_adv=True):
        if batch_size is None:
            batch_size = self.batch_size
        self.train()
        torch.retain_graph = True

        batch_input, data_label, margins = dataloader_trainset.next_triplet_same_level_batch(batch_size, K_num,
                                                                                             dynamic_margin, level)
        batch_input, data_label, margins = cudafy(batch_input), cudafy(data_label), cudafy(margins)
        margins = margins * self.margin if dynamic_margin else self.margin

        word_embed, encoded = self.forward_norm(batch_input)
        loss = self.get_triplet_semihard_loss(encoded, data_label, margins)
        if self.train_loss_type == 'triplet_v_adv' and same_v_adv:
            get_triplet_v_adv_loss = self.get_triplet_v_adv_loss(loss, word_embed, batch_input, data_label, margins,
                                                                 self.p_mult)
            loss += get_triplet_v_adv_loss
        elif self.train_loss_type == 'triplet_v_adv_distance' and same_v_adv:
            get_triplet_v_adv_loss = self.get_triplet_distance_v_adv_loss(encoded, word_embed, batch_input, self.p_mult)
            loss += get_triplet_v_adv_loss

        self.back_propagation(loss)

    def validation(self, data_left, data_right, data_label):
        self.eval()
        prediction, left_word_emb, right_word_emb, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        p_numpy = prediction.cpu().detach().numpy()
        p_numpy[p_numpy > 1] = 1
        p_numpy[p_numpy < 0] = 0
        p_numpy = np.round(p_numpy)
        label_numpy = data_label.cpu().detach().numpy()
        # 0 is Positive, 1 is Negative
        p_numpy = p_numpy.reshape(-1, 1)
        p_0, p_1 = np.array(p_numpy == 0, dtype=np.int32), np.array(p_numpy == 1, dtype=np.int32)
        t_0, t_1 = np.array(label_numpy == 0, dtype=np.int32), np.array(label_numpy == 1, dtype=np.int32)

        acc = np.sum(p_numpy == label_numpy) / label_numpy.size
        tp = np.sum(p_0 * t_0) / label_numpy.size
        tn = np.sum(p_1 * t_1) / label_numpy.size
        fp = np.sum(np.less(p_numpy, label_numpy)) / label_numpy.size
        fn = np.sum(np.greater(p_numpy, label_numpy)) / label_numpy.size
        # fp = np.sum(p_numpy != label_numpy) / label_numpy.size
        tp = round(float(tp), 5)
        fp = round(float(fp), 5)
        fn = round(float(fn), 5)
        tn = round(float(tn), 5)
        if not self.train_loss_type.startswith('triplet'):
            label_loss = self.get_cross_entropy_loss(prediction, data_label).cpu().detach().numpy()
            label_loss = round(float(label_loss), 5)
            print('tp:', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', acc, 'label_loss', label_loss)
        else:
            print('tp:', tp, 'fp', fp, 'fn', fn, 'tn', tn, 'acc', acc)

        return tp, fp, fn, tn, acc

    def partial_order_validation(self, data_left1, data_left2, data_right, data_label):
        """

        :param data_left1: sample from train dataset.
        :param data_left2: sample from train dataset and instance is not from same relation.
        :param data_right: sample from validation dataset.
        :param data_label: the order
        :return: partially ordered accuracy.
        """
        self.eval()
        prediction1 = self.siamese_forward(data_left1, data_right)[0]
        prediction2 = self.siamese_forward(data_left2, data_right)[0]
        order = prediction1 < prediction2
        acc = np.sum(order.cpu().detach().numpy() == data_label) / len(order)
        return acc

    def pred_X(self, data_left, data_right):
        self.eval()
        if isinstance(data_right, np.ndarray):
            data_right = cudafy(torch.from_numpy(np.array(data_right, dtype=np.int64)))
        if isinstance(data_left, np.ndarray):
            data_left = cudafy(torch.from_numpy(np.array(data_left, dtype=np.int64)))
        prediction, _l, _r, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        return prediction, encoded_l, encoded_r

    def pred_vector(self, data):
        self.eval()
        if isinstance(data, list):
            data = cudafy(torch.from_numpy(np.array(data, dtype=np.int64)))
        if self.train_loss_type.startswith('triplet'):
            _, vectors = self.forward_norm(data)
            # _, vectors = self.forward(data)
        else:
            _, vectors = self.forward(data)
        return vectors
