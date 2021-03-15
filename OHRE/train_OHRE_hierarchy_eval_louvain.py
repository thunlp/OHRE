"""
Louvain no isolation first then inserting.
"""
import numpy as np
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath('../lib/'))

from dataloader.dataloader import dataloader
from model.siamodel import RSN
from module.semiclusters import prepare_cluster_list, Top_Down_Louvain_with_test_cluster_done_avg_link_list

from module.clusters import Louvain_no_isolation, Hierarchical_Louvain
from evaluation.evaluation import HierarchyClusterEvaluation, ClusterEvaluationNew, ClusterEvaluation, \
    HierarchyClusterEvaluationTypes
from kit.messager import messager
from kit.utils import cudafy


def train_SN(train_data_file, val_data_file, test_data_file, wordvec_file, load_model_name=None, save_model_name='SN',
             trainset_loss_type='triplet', testset_loss_type='none', testset_loss_mask_epoch=3, p_cond=0.03,
             p_denoise=1.0, rel2id_file=None, similarity_file=None, dynamic_margin=True, margin=1.0,
             louvain_weighted=False, level_train=False, shallow_to_deep=False, same_level_pair_file=None,
             max_len=120, pos_emb_dim=5, same_ratio=0.06, batch_size=64, batch_num=10000, epoch_num=1,
             val_size=10000, select_cluster=None, omit_relid=None, labeled_sample_num=None, squared=True,
             same_level_part=None, mask_same_level_epoch=1, same_v_adv=False, random_init=False, seed=42,
             K_num=4, evaluate_hierarchy=False, train_for_cluster_file=None, train_structure_file=None,
             all_structure_file=None, to_cluster_data_num=100, incre_threshold=0,
             iso_threshold=5, avg_link_increment=True, modularity_increment=False):
    # preparing saving files.
    if select_cluster is None:
        select_cluster = ['Louvain']
    if load_model_name is not None:

        load_path = os.path.join('model_file', load_model_name).replace('\\', '/')
    else:
        load_path = None

    save_path = os.path.join('model_file', save_model_name).replace('\\', '/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,
                     types=['train_data_file', 'val_data_file', 'test_data_file', 'load_model_name', 'save_model_name',
                            'trainset_loss_type', 'testset_loss_type', 'testset_loss_mask_epoch', 'p_cond', 'p_denoise',
                            'same_ratio', 'labeled_sample_num'],
                     json_name='train_msg.json')
    msger.record_message([train_data_file, val_data_file, test_data_file, load_model_name, save_model_name,
                          trainset_loss_type, testset_loss_type, testset_loss_mask_epoch, p_cond, p_denoise, same_ratio,
                          labeled_sample_num])
    msger.save_json()

    print('-----Data Loading-----')
    # for train
    dataloader_train = dataloader(train_data_file, wordvec_file, rel2id_file, similarity_file, same_level_pair_file,
                                  max_len=max_len, random_init=random_init, seed=seed)
    # for cluster never seen instances
    dataloader_train_for_cluster = dataloader(train_for_cluster_file, wordvec_file, rel2id_file, similarity_file,
                                              same_level_pair_file, max_len=max_len, random_init=random_init, seed=seed)
    # for validation, to select best model
    dataloader_val = dataloader(val_data_file, wordvec_file, rel2id_file, similarity_file, max_len=max_len)
    # for cluster
    dataloader_test = dataloader(test_data_file, wordvec_file, rel2id_file, similarity_file, max_len=max_len)
    word_emb_dim = dataloader_train._word_emb_dim_()
    word_vec_mat = dataloader_train._word_vec_mat_()
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Initializing-----')

    rsn = RSN(word_vec_mat=word_vec_mat, max_len=max_len, pos_emb_dim=pos_emb_dim, dropout=0.2)

    if load_path:
        rsn.load_model(load_path)
    rsn = cudafy(rsn)
    rsn.set_train_op(batch_size=batch_size, train_loss_type=trainset_loss_type, testset_loss_type=testset_loss_type,
                     p_cond=p_cond, p_denoise=p_denoise, p_mult=0.02, squared=squared, margin=margin)

    print('-----Validation Data Preparing-----')

    val_data, val_data_label = dataloader_val._part_data_(100)

    print('-----Clustering Data Preparing-----')
    train_hierarchy_structure_info = json.load(open(train_structure_file))
    all_hierarchy_structure_info = json.load(open(all_structure_file))
    train_hierarchy_cluster_list, gt_hierarchy_cluster_list, train_data_num, test_data_num, train_data, train_label, test_data, test_label = prepare_cluster_list(
        dataloader_train_for_cluster,
        dataloader_test,
        train_hierarchy_structure_info,
        all_hierarchy_structure_info,
        to_cluster_data_num)
    batch_num_list = [batch_num] * epoch_num
    # start_cluster_accuracy = 0.5
    best_validation_f1 = 0
    least_epoch = 1
    best_step = 0
    for epoch in range(epoch_num):
        msger = messager(save_path=save_path,
                         types=['batch_num', 'train_tp', 'train_fp', 'train_fn', 'train_tn', 'train_l',
                                'test_tp', 'test_fp', 'test_fn', 'test_tn', 'test_l'],
                         json_name='SNmsg' + str(epoch) + '.json')
        # for cluster
        # test_data, test_data_label = dataloader_test._data_()
        print('------epoch {}------'.format(epoch))
        print('max batch num to train is {}'.format(batch_num_list[epoch]))
        for i in range(1, batch_num_list[epoch] + 1):
            to_cluster_flag = False
            if trainset_loss_type.startswith("triplet"):
                if level_train and epoch < mask_same_level_epoch:
                    if i <= 1 / same_level_part * batch_num_list[epoch]:
                        rsn.train_triplet_same_level(dataloader_train, batch_size=batch_size, K_num=4,
                                                     dynamic_margin=dynamic_margin, level=1, same_v_adv=same_v_adv)
                    elif i <= 2 / same_level_part * batch_num_list[epoch]:
                        rsn.train_triplet_same_level(dataloader_train, batch_size=batch_size, K_num=4,
                                                     dynamic_margin=dynamic_margin, level=2, same_v_adv=same_v_adv)
                    else:
                        rsn.train_triplet_loss(dataloader_train, batch_size=batch_size, dynamic_margin=dynamic_margin)
                else:
                    rsn.train_triplet_loss(dataloader_train, batch_size=batch_size, dynamic_margin=dynamic_margin)
            else:
                rsn.train_RSN(dataloader_train, dataloader_test, batch_size=batch_size)

            if i % 100 == 0:
                print('temp_batch_num: ', i, ' total_batch_num: ', batch_num_list[epoch])
            if i % 1000 == 0 and epoch >= least_epoch:
                print(save_model_name, 'epoch:', epoch)

                print('Validation:')
                cluster_result, cluster_msg = Louvain_no_isolation(dataset=val_data, edge_measure=rsn.pred_X,
                                                                   weighted=louvain_weighted)
                cluster_eval_b3 = ClusterEvaluation(val_data_label, cluster_result).printEvaluation(
                    print_flag=False)

                cluster_eval_new = ClusterEvaluationNew(val_data_label, cluster_result).printEvaluation(
                    print_flag=False)
                two_f1 = cluster_eval_new['F1']
                if two_f1 > best_validation_f1:  # acc
                    to_cluster_flag = True
                    best_step = i
                    best_validation_f1 = two_f1

            if to_cluster_flag:
                # if True:
                if 'Louvain' in select_cluster:
                    print('-----Top Down Hierarchy Louvain Clustering-----')
                    if avg_link_increment:
                        # link_th_list = [0.5, 1, 2, 5, 10, 15, 20, 50, 100]
                        # link_th_list = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.4]
                        # link_th_list = [i * 0.02 for i in range(1, 100)]
                        link_th_list = [0.05]
                        cluster_result, cluster_msg = Louvain_no_isolation(dataset=test_data, edge_measure=rsn.pred_X,
                                                                           weighted=louvain_weighted)

                        predicted_cluster_dict_list = Top_Down_Louvain_with_test_cluster_done_avg_link_list(
                            # predicted_cluster_dict_list = Louvain_with_test_cluster_done_avg_link_list(
                            cluster_result,
                            train_data_num,
                            test_data_num,
                            train_data,
                            test_data,
                            train_hierarchy_cluster_list,
                            rsn.pred_X,
                            link_th_list)
                        best_hyper_score = 0
                        best_eval_info = None
                        for predicted_cluster_dict in predicted_cluster_dict_list:
                            predicted_cluster_list = predicted_cluster_dict['list']
                            evaluation = HierarchyClusterEvaluation(gt_hierarchy_cluster_list,
                                                                    predicted_cluster_list,
                                                                    test_data_num)
                            eval_info = evaluation.printEvaluation()
                            if eval_info['total_F1'] > best_hyper_score:
                                best_eval_info = eval_info
                                best_hyper_score = eval_info['total_F1']

                    rsn.save_model(save_path=save_path, global_step=i + epoch * batch_num)
                    print('model and clustering messages saved.')
        print('End: The model is:', save_model_name, trainset_loss_type, testset_loss_type, 'p_cond is:', p_cond)
    print(seed)
    print("best step:", best_step)
    print("new metric Info:")
    print("F1(%)")
    print(best_eval_info['match_f1'] * 100)

    print("taxonomy Info:")
    print("Precision(%); Recall(%); F1(%)")
    print(round(best_eval_info['taxonomy_precision'] * 100, 3), "; ", round(best_eval_info['taxonomy_recall'] * 100, 3),
          "; ",
          round(best_eval_info['taxonomy_F1'] * 100, 3))

    print("Total Info:")
    print("Precision(%); Recall(%); F1(%)")
    print(round(best_eval_info['total_precision'] * 100, 3), "; ", round(best_eval_info['total_recall'] * 100, 3), "; ",
          round(best_eval_info['total_F1'] * 100, 3))

    # print("Best New Metric Info:")
    # print("F1(%)")
    # print(best_cluster_eval_new['total_f1'] * 100)
    #
    # print("")
    # print("B3 Info:")
    # print("B3 Precision(%);B3 Recall(%);B3 F1(%); B3 F05(%)")
    # print(best_cluster_eval_b3['precision'] * 100, "; ", best_cluster_eval_b3['recall'] * 100, "; ",
    #       best_cluster_eval_b3['F1'] * 100, "; ", best_cluster_eval_b3['F0.5'] * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument("--dataset", type=str, default='ori')
    parser.add_argument("--train_data_file", type=str, default='../data/fewrel_ori/hierarchy_fewrel80_train.json')
    parser.add_argument("--train_for_cluster_file", type=str,
                        default='../data/fewrel_ori/hierarchy_fewrel80_train_100.json')
    parser.add_argument("--val_data_file", type=str, default='../data/fewrel_ori/fewrel80_test_train.json')
    parser.add_argument("--test_data_file", type=str, default='../data/fewrel_ori/fewrel80_test_test.json')
    parser.add_argument("--wordvec_file", type=str, default='../data/wordvec/word_vec.json')

    parser.add_argument("--rel2id_file", type=str, default='../data/support_files/rel2id.json')
    parser.add_argument("--similarity_file", type=str, default='../data/support_files/all_similarity.pkl')
    parser.add_argument("--same_level_pair_file", type=str, default='../data/support_files/same_level_pair.json')
    parser.add_argument("--all_structure_file", type=str, default='../data/support_files/all_structure.json')
    parser.add_argument("--train_structure_file", type=str, default='../data/support_files/train_structure.json')

    parser.add_argument("--dynamic_margin", type=int, default=1)
    parser.add_argument("--distance_squared", type=int, default=1)
    parser.add_argument("--louvain_weighted", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--level_train", type=int, default=0)
    parser.add_argument("--shallow_to_deep", type=int, default=1)
    parser.add_argument("--same_level_part", type=int, default=200)
    parser.add_argument("--mask_same_level_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--K_num", type=int, default=4)

    parser.add_argument("--evaluate_hierarchy", type=int, default=1)
    parser.add_argument("--incre_threshold", type=float, default=0)
    parser.add_argument("--iso_threshold", type=int, default=5)
    parser.add_argument("--partially_order_validation", type=int, default=0)

    parser.add_argument("--same_v_adv", type=int, default=0)
    parser.add_argument("--random_init", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--load_model_name", type=str, default=None)
    parser.add_argument("--save_model_name", type=str, default='ori/')
    parser.add_argument("--select_cluster", type=int, default=1)
    parser.add_argument("--trainset_loss_type", type=str, default='triplet')
    parser.add_argument("--testset_loss_type", type=str, default='none')
    parser.add_argument("--testset_loss_mask_epoch", type=int, default=0)
    parser.add_argument("--p_cond", type=float, default=0.03)
    parser.add_argument("--p_denoise", type=float, default=1.0)
    parser.add_argument("--same_ratio", type=float, default=0.06)
    parser.add_argument("--batch_num", type=int, default=10000)
    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--omit_relid", type=int, default=None, help=
    "None means not omit; 0 means unsupervised mode; otherwise means reserving all the relations with relid<=omit_relid from trainset")
    parser.add_argument("--labeled_sample_num", type=int, default=None)
    args = parser.parse_args()
    cluster_dict = {0: [], 1: ['Louvain'], 2: ['HAC'], 3: ['Louvain', 'HAC']}
    args.select_cluster = cluster_dict[args.select_cluster]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.dynamic_margin = True if args.dynamic_margin == 1 else False
    args.distance_squared = True if args.distance_squared == 1 else False
    args.louvain_weighted = True if args.louvain_weighted == 1 else False
    args.level_train = True if args.level_train == 1 else False
    args.shallow_to_deep = True if args.shallow_to_deep == 1 else False
    args.same_v_adv = True if args.same_v_adv == 1 else False
    args.random_init = True if args.random_init == 1 else False
    args.evaluate_hierarchy = True if args.evaluate_hierarchy == 1 else False
    args.partially_order_validation = True if args.partially_order_validation == 1 else False

    if args.dataset == 'ori':
        args.train_data_file = '../data/fewrel_ori/hierarchy_fewrel80_train.json'
        # args.train_data_file = '../data/fewrel_ori/hierarchy_fewrel80_train_600.json'
        args.train_for_cluster_file = '../data/fewrel_ori/hierarchy_fewrel80_train_100.json'
        args.val_data_file = '../data/fewrel_ori/hierarchy_fewrel80_test_train.json'
        args.test_data_file = '../data/fewrel_ori/hierarchy_fewrel80_test_test.json'

    train_SN(
        train_data_file=args.train_data_file,
        val_data_file=args.val_data_file,
        test_data_file=args.test_data_file,
        wordvec_file=args.wordvec_file,
        load_model_name=args.load_model_name,
        save_model_name=args.save_model_name,
        select_cluster=args.select_cluster,
        trainset_loss_type=args.trainset_loss_type,
        testset_loss_type=args.testset_loss_type,
        testset_loss_mask_epoch=args.testset_loss_mask_epoch,
        p_cond=args.p_cond,
        p_denoise=args.p_denoise,
        same_ratio=args.same_ratio,
        batch_num=args.batch_num,
        epoch_num=args.epoch_num,
        val_size=args.val_size,
        omit_relid=args.omit_relid,
        labeled_sample_num=args.labeled_sample_num,
        rel2id_file=args.rel2id_file,
        similarity_file=args.similarity_file,
        same_level_pair_file=args.same_level_pair_file,
        dynamic_margin=args.dynamic_margin,
        squared=args.distance_squared,
        margin=args.margin,
        louvain_weighted=args.louvain_weighted,
        level_train=args.level_train,
        shallow_to_deep=args.shallow_to_deep,
        same_level_part=args.same_level_part,
        mask_same_level_epoch=args.mask_same_level_epoch,
        same_v_adv=args.same_v_adv,
        random_init=args.random_init,
        seed=args.seed,
        batch_size=args.batch_size,
        K_num=args.K_num,
        evaluate_hierarchy=args.evaluate_hierarchy,
        train_structure_file=args.train_structure_file,
        all_structure_file=args.all_structure_file,
        train_for_cluster_file=args.train_for_cluster_file,
        incre_threshold=args.incre_threshold,
        iso_threshold=args.iso_threshold
    )
