import json
import random
import numpy as np
from copy import deepcopy
import pickle
import torch
from torch.autograd import Variable
import os


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class dataloader():
    def __init__(self, data_file, wordvec_file, rel2id_file=None, similarity_file=None, same_level_pair_file=None,
                 max_len=120, random_init=False, seed=None):
        if not random_init and seed:
            seed_torch(seed)

        self.max_len = max_len

        # Load and Pre-process word vec
        print('Wordvec Loading!-----')
        with open(wordvec_file, 'r') as r:
            ori_word_vec = json.load(r)
        print('Wordvec Loaded!-----')

        print('Wordvec Preprocessing!-----')
        self.word2id = {}
        self.word_vec_tot = len(ori_word_vec)
        self.UNK = self.word_vec_tot
        self.BLANK = self.word_vec_tot + 1
        self.word_emb_dim = len(ori_word_vec[0]['vec'])
        print("Got {} words of {} dims".format(self.word_vec_tot, self.word_emb_dim))
        print("Building word_vec_mat and mapping...")
        self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_emb_dim), dtype=np.float32)
        for cur_id, word in enumerate(ori_word_vec):
            w = word['word'].lower()
            self.word2id[w] = cur_id
            self.word_vec_mat[cur_id, :] = word['vec']
        self.word2id['UNK'] = self.UNK
        self.word2id['BLANK'] = self.BLANK
        print("Wordvec Preprocessed!-----")

        # Load and preprocess data
        print('Data Loading!-----')
        with open(data_file, 'r') as r:
            data = json.load(r)
        ori_data_len = len(data)
        print('Original data has {} instances'.format(ori_data_len))
        print('Data Loaded!-----')

        print('Data Preprocessing!-----')
        print('Delete long instances and NA, Other instances...')
        unk_word_count = 0
        pop_list = []
        for i in range(len(data)):
            # delete long instances and NA\Other instances
            if (len(data[i]['sentence']) > max_len or data[i]['relation'] == 'NA' or data[i]['relation'] == 'Other'):
                pop_list.append(i)

        for i in range(len(data)):
            if len(data) - i - 1 in pop_list:
                data.pop(len(data) - i - 1)

        for i in range(len(data)):
            # delete None elements
            data[i]['sentence'] = list(filter(None, data[i]['sentence']))
            # ori_sentence = deepcopy(data[i]['sentence'])
            # sentence-as-str to sentence-as-index
            for j in range(len(data[i]['sentence'])):
                data[i]['sentence'][j] = self.word2id.get(data[i]['sentence'][j].lower(), self.UNK)
                if data[i]['sentence'][j] == self.UNK:
                    unk_word_count += 1
                    # print('unknown found! The sentence is:')
                    # print(ori_sentence)
                    # print('The unknown word is:')
                    # print(ori_sentence[j])

        print('Data deletion completed, {} instances deleted, {} instances left'.
              format(ori_data_len - len(data), len(data)))
        print('Data_to_index completed, {} unknown words found'.
              format(unk_word_count))
        print('Data Preprocessed!-----')

        print('Processed Data List Creating!----')
        self.processed_data = []
        for _, item in enumerate(data):
            temp_item = {}
            temp_item['sentence'] = item['sentence']
            temp_item['e1_begin'] = item['head']['e1_begin']
            temp_item['e1_end'] = item['head']['e1_end']
            temp_item['e2_begin'] = item['tail']['e2_begin']
            temp_item['e2_end'] = item['tail']['e2_end']
            temp_item['relid'] = item['relid']
            self.processed_data.append(temp_item)
        print('Processed Data List Created!----')
        self.create_lists_and_dicts(rel2id_file, similarity_file, same_level_pair_file)

    def create_lists_and_dicts(self, rel2id_file=None, similarity_file=None, same_level_pair_file=None):
        # creating some lists and dicts to be used in batching functions
        print('lists_for_single_rel_mention Creating!-----')
        self.lists_for_single_rel_mention = {}
        self.rel_list = []
        for i, item in enumerate(self.processed_data):
            try:
                self.lists_for_single_rel_mention[item['relid']].append(i)
            except:
                self.lists_for_single_rel_mention[item['relid']] = [i]
                self.rel_list.append(item['relid'])

        useless_single_rel_mention_list = []
        for relid in self.lists_for_single_rel_mention:
            if len(self.lists_for_single_rel_mention[relid]) < 2:
                useless_single_rel_mention_list.append(relid)

        for relid in useless_single_rel_mention_list:
            self.lists_for_single_rel_mention.pop(relid)

        print('dict_for_relids Creating!-----')
        self.relid_dict = {}  # remap relid to 0-? as labels for softmax loss
        count = 0
        for item in self.processed_data:
            if item['relid'] not in self.relid_dict.keys():
                self.relid_dict[item['relid']] = count
                count += 1
        if rel2id_file:
            with open(rel2id_file, 'r') as r:
                self.rel2id = json.load(r)

            self.id2rel = dict([(v, k) for (k, v) in self.rel2id.items()])
        if similarity_file:
            self.rel_sim = pickle.load(open(similarity_file, "rb"))

        if same_level_pair_file:
            with open(same_level_pair_file, "r") as r:
                self.same_level_pair = json.load(r)

    def get_two_relation_distance(self, relid1, relid2):
        relation_name1 = self.id2rel[relid1]
        relation_name2 = self.id2rel[relid2]
        return 1 - self.rel_sim[relation_name1][relation_name2]

    def select_relation(self, select_relid_list):
        new_processed_data = []
        for item in self.processed_data:
            if item['relid'] in select_relid_list:
                new_processed_data.append(item)
        self.processed_data = new_processed_data
        self.create_lists_and_dicts()

    def select_sample_num(self, sample_num):
        self.processed_data = random.sample(self.processed_data, sample_num)
        print('---The left sample number is:', len(self.processed_data))
        self.create_lists_and_dicts()

    # data_processing functions
    def posnum_to_posarray(self, posbegin, posend, max_len=120):
        if (posend < posbegin):
            posend = posbegin
        array1 = np.arange(0, posbegin) - posbegin
        array2 = np.zeros(posend - posbegin, dtype=np.int32)
        array3 = np.arange(posend, max_len) - posend
        posarray = np.append(np.append(array1, array2), array3) + max_len
        return posarray

    def data_to_padded_idx_data(self, data, max_len=120):
        '''padded_idx_data as [pos1array, pos2array, sentence]'''
        padded_idx_data = np.zeros([3, max_len], dtype=np.int32)
        padded_idx_data[0] = self.posnum_to_posarray(data['e1_begin'], data['e1_end'])
        padded_idx_data[1] = self.posnum_to_posarray(data['e2_begin'], data['e2_end'])
        padnum = max_len - len(data['sentence'])
        padded_idx_data[2] = np.append(np.array(data['sentence']), np.array([self.BLANK] * padnum))
        return padded_idx_data

    # batching functions
    def next_batch_same(self, batch_size):  # return a list
        batch_data_same_left = []
        batch_data_same_right = []
        for i in range(batch_size):
            next_rel_index = random.choice(list(self.lists_for_single_rel_mention.keys()))
            temp_index = random.sample(self.lists_for_single_rel_mention[next_rel_index], 2)
            batch_data_same_left.append(self.data_to_padded_idx_data(self.processed_data[temp_index[0]]))
            batch_data_same_right.append(self.data_to_padded_idx_data(self.processed_data[temp_index[1]]))

        return batch_data_same_left, batch_data_same_right

    def next_batch_rand(self, batch_size, active_selector=None, select_num=1, return_rel_id=False):
        batch_data_rand_left = []
        batch_data_rand_right = []
        left_data_relid = []
        right_data_relid = []

        idx_list = np.arange(len(self.processed_data))
        rnd_list = np.random.choice(idx_list, 2 * batch_size * select_num)

        for i in range(batch_size * select_num):
            temp_index = rnd_list[2 * i:2 * i + 2]
            while 1:
                if (self.processed_data[temp_index[0]]['relid'] != self.processed_data[temp_index[1]]['relid']):
                    batch_data_rand_left.append(self.data_to_padded_idx_data(self.processed_data[temp_index[0]]))
                    batch_data_rand_right.append(self.data_to_padded_idx_data(self.processed_data[temp_index[1]]))
                    left_data_relid.append(self.processed_data[temp_index[0]]['relid'])
                    right_data_relid.append(self.processed_data[temp_index[1]]['relid'])
                    break
                else:
                    temp_index = np.random.choice(idx_list, 2, replace=False)

        if active_selector is not None:
            selected_index = active_selector(batch_data_rand_left, batch_data_rand_right).argsort()[:batch_size]
            batch_data_rand_left_selected = []
            batch_data_rand_right_selected = []
            for temp_index in selected_index:
                batch_data_rand_left_selected.append(batch_data_rand_left[int(temp_index)])
                batch_data_rand_right_selected.append(batch_data_rand_right[int(temp_index)])
            batch_data_rand_left = batch_data_rand_left_selected
            batch_data_rand_right = batch_data_rand_right_selected
        if return_rel_id:
            return batch_data_rand_left, batch_data_rand_right, left_data_relid, right_data_relid
        return batch_data_rand_left, batch_data_rand_right

    def next_batch(self, batch_size, same_ratio=0.5, active_selector=None, select_num=1):
        same_batch_size = int(np.round(batch_size * same_ratio))
        batch_data_same_left, batch_data_same_right = self.next_batch_same(same_batch_size)
        batch_data_rand_left, batch_data_rand_right = self.next_batch_rand(
            batch_size - same_batch_size, active_selector=active_selector, select_num=select_num)
        batch_data_left = batch_data_same_left + batch_data_rand_left
        batch_data_right = batch_data_same_right + batch_data_rand_right
        batch_data_label = [[0]] * len(batch_data_same_left) + [[1]] * len(batch_data_rand_left)
        batch_data_left = torch.from_numpy(np.array(batch_data_left, dtype=np.int64))
        batch_data_right = torch.from_numpy(np.array(batch_data_right, dtype=np.int64))
        batch_data_label = torch.from_numpy(np.array(batch_data_label, dtype=np.int64))
        return batch_data_left, batch_data_right, batch_data_label

    def train_partial_order_batch(self, batch_size, active_selector=None, select_num=1):
        # return batch_data_rand_left, batch_data_rand_right, left_data_relid, right_data_relid
        batch_data_rand_left, batch_data_rand_right, left_data_relid, right_data_relid = self.next_batch_rand(
            batch_size, active_selector=active_selector, select_num=select_num, return_rel_id=True)

        batch_data_left = torch.from_numpy(np.array(batch_data_rand_left, dtype=np.int64))
        batch_data_right = torch.from_numpy(np.array(batch_data_rand_right, dtype=np.int64))

        return batch_data_left, batch_data_right, left_data_relid, right_data_relid

    def test_partial_order_batch(self, batch_size, train_left_data1_relid, train_left_data2_relid):
        test_batch_data = []
        test_rel_id_list = []
        assert len(train_left_data1_relid) == batch_size
        for i in range(batch_size):
            next_rel_index = random.choice(list(self.lists_for_single_rel_mention.keys()))
            temp_index = random.sample(self.lists_for_single_rel_mention[next_rel_index], 1)
            test_batch_data.append(self.data_to_padded_idx_data(self.processed_data[temp_index[0]]))
            test_rel_id_list.append(self.processed_data[temp_index[0]]['relid'])

        partially_order_label = []
        for i in range(batch_size):
            left_relid1 = train_left_data1_relid[i]
            left_relid2 = train_left_data2_relid[i]
            right_relid = test_rel_id_list[i]
            distance1 = self.get_two_relation_distance(left_relid1, right_relid)
            distance2 = self.get_two_relation_distance(left_relid2, right_relid)
            if distance1 < distance2:
                partially_order_label.append(1)
            else:
                partially_order_label.append(0)

        test_batch_data = torch.from_numpy(np.array(test_batch_data, dtype=np.int64))

        return test_batch_data, partially_order_label

    def next_triplet_batch(self, batch_size, K_num=4, dynamic_margin=True, unbalanced_batch=False):
        batch_data_input = []
        labels = []
        P = int(batch_size / K_num)
        indexes_list = []
        for i in range(K_num):
            next_rel_index = random.choice(list(self.lists_for_single_rel_mention.keys()))
            indexes_list.append(next_rel_index)
            temp_index = random.sample(self.lists_for_single_rel_mention[next_rel_index], P)
            labels.extend([next_rel_index] * P)
            for index in temp_index:
                batch_data_input.append(self.data_to_padded_idx_data(self.processed_data[index]))
        margins = np.zeros((batch_size, batch_size))
        if dynamic_margin:
            for i, relation_index in enumerate(indexes_list):

                margins[i * P:(i + 1) * P, i * P:(i + 1) * P] = 0
                for j, another_relation_index in enumerate(indexes_list[i + 1:]):
                    distance = self.get_two_relation_distance(relation_index, another_relation_index)
                    margins[i * P:(i + 1) * P, (i + j + 1) * P:(i + 1 + j + 1) * P] = distance
                    margins[(i + j + 1) * P:(i + 1 + j + 1) * P, i * P:(i + 1) * P] = distance

        if unbalanced_batch:
            rel_data_nums = [len(self.lists_for_single_rel_mention[key]) for key in
                             self.lists_for_single_rel_mention.keys()]
            total_num = sum(rel_data_nums)
            p = [rel_data_num / total_num for rel_data_num in rel_data_nums]
            p = torch.nn.functional.softmax(torch.Tensor(np.array(p))).cpu().numpy()
            batch_data_input = []
            labels = []
            indexes_list = []
            for i in range(batch_size):
                next_rel_index = np.random.choice(list(self.lists_for_single_rel_mention.keys()), p=p)
                indexes_list.append(next_rel_index)
                temp_index = random.sample(self.lists_for_single_rel_mention[next_rel_index], 1)
                labels.extend([next_rel_index])
                for index in temp_index:
                    batch_data_input.append(self.data_to_padded_idx_data(self.processed_data[index]))

            margins = np.zeros((batch_size, batch_size))
            P = 1
            if dynamic_margin:  # todo
                for i, relation_index in enumerate(indexes_list):
                    # same label, set to 1
                    margins[i * P:(i + 1) * P, i * P:(i + 1) * P] = 0
                    for j, another_relation_index in enumerate(indexes_list[i + 1:]):
                        distance = self.get_two_relation_distance(relation_index, another_relation_index)
                        margins[i * P:(i + 1) * P, (i + j + 1) * P:(i + 1 + j + 1) * P] = distance
                        margins[(i + j + 1) * P:(i + 1 + j + 1) * P, i * P:(i + 1) * P] = distance

        torch_batch_input = Variable(torch.from_numpy(np.array(batch_data_input, dtype=np.int64)))
        torch_batch_labels = Variable(torch.from_numpy(np.array(np.array(labels))))
        torch_margins = Variable(torch.from_numpy(np.array(margins)))
        return torch_batch_input, torch_batch_labels, torch_margins

    def next_triplet_same_level_batch(self, batch_size, K_num=4, dynamic_margin=True, level=1):
        batch_data_input = []
        labels = []
        P = int(batch_size / K_num)
        indexes_list = []
        batch_rel_list = []
        for i in range(int(K_num / 2)):  # since it's pair
            if i == 0:
                rel1, rel2 = random.choice(self.same_level_pair[str(level)])
                batch_rel_list.append(rel1)
                batch_rel_list.append(rel2)
            else:
                while True:
                    rel1, rel2 = random.choice(self.same_level_pair[str(level)])
                    if rel1 not in batch_rel_list and rel2 not in batch_rel_list:
                        batch_rel_list.append(rel1)
                        batch_rel_list.append(rel2)
                        break

        for rel in batch_rel_list:
            indexes_list.append(self.rel2id[rel])
        for rel_index in indexes_list:
            temp_index = random.sample(self.lists_for_single_rel_mention[rel_index], P)
            labels.extend([rel_index] * P)
            for index in temp_index:
                batch_data_input.append(self.data_to_padded_idx_data(self.processed_data[index]))
        margins = np.zeros((batch_size, batch_size))

        if dynamic_margin:
            for i, relation_index in enumerate(indexes_list):
                # same label, set to 1
                margins[i * P:(i + 1) * P, i * P:(i + 1) * P] = 0
                for j, another_relation_index in enumerate(indexes_list[i + 1:]):
                    distance = self.get_two_relation_distance(relation_index, another_relation_index)
                    margins[i * P:(i + 1) * P, (i + j + 1) * P:(i + 1 + j + 1) * P] = distance
                    margins[(i + j + 1) * P:(i + 1 + j + 1) * P, i * P:(i + 1) * P] = distance

        torch_batch_input = Variable(torch.from_numpy(np.array(batch_data_input, dtype=np.int64)))
        torch_batch_labels = Variable(torch.from_numpy(np.array(np.array(labels))))
        torch_margins = Variable(torch.from_numpy(np.array(margins)))
        return torch_batch_input, torch_batch_labels, torch_margins

    def next_batch_same_self(self, batch_size):
        batch_data_same_left = []
        batch_data_same_right = []
        for i in range(batch_size):
            temp_index = random.randint(0, len(self.processed_data) - 1)
            batch_data_same_left.append(self.data_to_padded_idx_data(self.processed_data[temp_index]))
            batch_data_same_right.append(self.data_to_padded_idx_data(self.processed_data[temp_index]))
        return batch_data_same_left, batch_data_same_right

    def next_batch_rand_self(self, batch_size):
        batch_data_rand_left = []
        batch_data_rand_right = []

        for i in range(batch_size):
            temp_index = random.sample(np.arange(len(self.processed_data)).tolist(), 2)
            batch_data_rand_left.append(self.data_to_padded_idx_data(self.processed_data[temp_index[0]]))
            batch_data_rand_right.append(self.data_to_padded_idx_data(self.processed_data[temp_index[1]]))

        return batch_data_rand_left, batch_data_rand_right

    def next_batch_self(self, batch_size, same_ratio=0.5):
        same_batch_size = int(np.round(batch_size * same_ratio))
        batch_data_same_left, batch_data_same_right = self.next_batch_same_self(same_batch_size)
        batch_data_rand_left, batch_data_rand_right = self.next_batch_rand_self(batch_size - same_batch_size)
        batch_data_left = batch_data_same_left + batch_data_rand_left
        batch_data_right = batch_data_same_right + batch_data_rand_right
        batch_data_label = [[0]] * len(batch_data_same_left) + [[1]] * len(batch_data_rand_left)
        return np.array(batch_data_left), np.array(batch_data_right), np.array(batch_data_label)

    def next_batch_ul(self, batch_size):
        batch_data_left = []
        batch_data_right = []
        for _ in range(batch_size):
            temp_index = random.sample(np.arange(len(self.processed_data)).tolist(), 2)
            batch_data_left.append(self.data_to_padded_idx_data(self.processed_data[temp_index[0]]))
            batch_data_right.append(self.data_to_padded_idx_data(self.processed_data[temp_index[1]]))
        return np.array(batch_data_left, dtype=np.int32), np.array(batch_data_right, dtype=np.int32)

    def next_batch_cnn(self, batch_size):
        batch_size = min(len(self.processed_data), batch_size)
        batch_data = []
        label = []
        index_list = random.sample(np.arange(len(self.processed_data)).tolist(), batch_size)
        for idx in index_list:
            batch_data.append(self.data_to_padded_idx_data(self.processed_data[idx]))
            label.append([self.relid_dict[self.processed_data[idx]['relid']]])
        return np.array(batch_data, dtype=np.int32), np.array(label, dtype=np.int32)

    def _data_and_relid_to_cluster_(self, num_of_data=2000, num_of_type=20, balanced=True):
        data_relid = []
        data_to_cluster = []
        if num_of_type > len(self.rel_list):
            temp_type_list = self.rel_list
        else:
            temp_type_list = random.sample(self.rel_list, num_of_type)

        for relid in temp_type_list:
            if balanced == True:
                temp_sample_num = int(num_of_data / num_of_type)
            else:
                temp_sample_num = int((0.5 + random.random()) * num_of_data / num_of_type)

            if temp_sample_num > len(self.lists_for_single_rel_mention[relid]):
                temp_sample_num = len(self.lists_for_single_rel_mention[relid])

            temp_data_index = random.sample(np.arange(len(self.lists_for_single_rel_mention[relid])).tolist()
                                            , temp_sample_num)
            for index in temp_data_index:
                data_to_cluster.append(
                    self.data_to_padded_idx_data(self.processed_data[self.lists_for_single_rel_mention[relid][index]]))
            data_relid += [relid] * temp_sample_num

        return data_to_cluster, data_relid

    def _data_(self):
        data_relid = []
        data_to_cluster = []
        for item in self.processed_data:
            data_to_cluster.append(self.data_to_padded_idx_data(item))
            data_relid.append(item['relid'])
        # return data_to_cluster[:100], data_relid[:100]
        return data_to_cluster, data_relid

    def _part_data_(self, each_class_datanum):
        data_relid = []
        data_to_cluster = []
        reltype_counter = dict()
        for item in self.processed_data:
            if reltype_counter.get(item['relid'], 0) >= each_class_datanum:
                continue
            else:
                data_to_cluster.append(self.data_to_padded_idx_data(item))
                data_relid.append(item['relid'])
                reltype_counter[item['relid']] = reltype_counter.get(item['relid'], 0) + 1
        return data_to_cluster, data_relid

    # data_information functions
    def _word_emb_dim_(self):
        return self.word_emb_dim

    def _word_vec_mat_(self):
        return self.word_vec_mat


if __name__ == '__main__':
    ''' test the dataloader'''
    dataloader_test = dataloader('../../data/fewrel/testset_test.json', '../../data/wordvec/word_vec.json')
    batch_data_left, batch_data_right, batch_data_label = dataloader_test.next_batch(10)
    print(batch_data_left[0])
    print(batch_data_right[0])
    print(batch_data_label)
