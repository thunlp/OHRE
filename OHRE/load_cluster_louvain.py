"""
Louvain no isolation first then inserting.
"""
import numpy as np
import json
import sys
import os
import argparse
import pickle

sys.path.append(os.path.abspath('../lib/'))

from dataloader.dataloader import dataloader
from model.siamodel import RSN
from module.semiclusters import prepare_cluster_list, \
    Top_Down_Louvain_with_test_cluster_done_avg_link_list, Top_Down_Louvain_with_test_cluster_done_avg_link_list_golden

from module.clusters import Louvain_no_isolation, Hierarchical_Louvain
from evaluation.evaluation import HierarchyClusterEvaluation, ClusterEvaluationNew, ClusterEvaluation, \
    HierarchyClusterEvaluationTypes, ClusterEvaluationB3Types
from kit.utils import cudafy


def load_cluster(train_data_file, test_data_file, wordvec_file, load_model_name=None, all_structure_file=None,
                 trainset_loss_type='triplet', testset_loss_type='none', p_cond=0.03, to_cluster_data_num=100,
                 p_denoise=1.0, rel2id_file=None, similarity_file=None, margin=1.0, save_cluster=False,
                 louvain_weighted=False, same_level_pair_file=None, train_for_cluster_file=None,
                 train_structure_file=None, test_infos_file=None, val_hier=False, golden=False,
                 max_len=120, pos_emb_dim=5, batch_size=64, squared=True, random_init=False, seed=42):
    if load_model_name is not None:
        load_path = os.path.join('model_file', load_model_name).replace('\\', '/')
    else:
        load_path = None

    print('-----Data Loading-----')
    # for train
    dataloader_train = dataloader(train_data_file, wordvec_file, rel2id_file, similarity_file, same_level_pair_file,
                                  max_len=max_len, random_init=random_init, seed=seed)
    # for cluster never seen instances
    dataloader_train_for_cluster = dataloader(train_for_cluster_file, wordvec_file, rel2id_file, similarity_file,
                                              same_level_pair_file, max_len=max_len)

    dataloader_test = dataloader(test_data_file, wordvec_file, rel2id_file, similarity_file, max_len=max_len)
    word_emb_dim = dataloader_train._word_emb_dim_()
    word_vec_mat = dataloader_train._word_vec_mat_()
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Initializing-----')

    rsn = RSN(word_vec_mat=word_vec_mat, max_len=max_len, pos_emb_dim=pos_emb_dim, dropout=0)
    rsn.set_train_op(batch_size=batch_size, train_loss_type=trainset_loss_type, testset_loss_type=testset_loss_type,
                     p_cond=p_cond, p_denoise=p_denoise, p_mult=0.02, squared=squared, margin=margin)

    if load_path:
        rsn.load_model(load_path + "/RSNbest.pt")
    rsn = cudafy(rsn)
    rsn.eval()
    print('-----Louvain Clustering-----')

    if val_hier:
        print('-----Top Down Hierarchy Expansion-----')
        train_hierarchy_structure_info = json.load(open(train_structure_file))
        all_hierarchy_structure_info = json.load(open(all_structure_file))
        train_hierarchy_cluster_list, gt_hierarchy_cluster_list, train_data_num, test_data_num, train_data, train_label, test_data, test_label = prepare_cluster_list(
            dataloader_train_for_cluster,
            dataloader_test,
            train_hierarchy_structure_info,
            all_hierarchy_structure_info,
            to_cluster_data_num)
        link_th_list = [0.2]

        if golden:
            link_th_list = [0.3]
            predicted_cluster_dict_list = Top_Down_Louvain_with_test_cluster_done_avg_link_list_golden(
                gt_hierarchy_cluster_list,
                train_data_num,
                test_data_num,
                train_data,
                test_data,
                train_hierarchy_cluster_list,
                rsn.pred_X,
                link_th_list)
        else:
            cluster_result, cluster_msg = Louvain_no_isolation(dataset=test_data, edge_measure=rsn.pred_X,
                                                               weighted=louvain_weighted)
            predicted_cluster_dict_list = Top_Down_Louvain_with_test_cluster_done_avg_link_list(
                cluster_result,
                train_data_num,
                test_data_num,
                train_data,
                test_data,
                train_hierarchy_cluster_list,
                rsn.pred_X,
                link_th_list)
            if save_cluster:
                json.dump(cluster_result, open("cluster_result.json", "w"))
                pickle.dump(predicted_cluster_dict_list, open("predicted_cluster_dict_list.pkl", "wb"))
                pickle.dump(gt_hierarchy_cluster_list, open("gt_hierarchy_cluster_list.pkl", "wb"))
                print("saved results!")
        for predicted_cluster_dict in predicted_cluster_dict_list:
            print("\n\n")
            predicted_cluster_list = predicted_cluster_dict['list']
            print("Isolation threhold", predicted_cluster_dict['iso'])
            print("Average Link threhold", predicted_cluster_dict['link_th'])
            pickle.dump(predicted_cluster_list, open("predicted_cluster_list.pkl", "wb"))
            evaluation = HierarchyClusterEvaluation(gt_hierarchy_cluster_list,
                                                    predicted_cluster_list,
                                                    test_data_num)
            eval_info = evaluation.printEvaluation(print_flag=True)
            HierarchyClusterEvaluationTypes(gt_hierarchy_cluster_list, predicted_cluster_list,
                                            test_infos_file, rel2id_file).printEvaluation()
    else:
        test_data, test_data_label = dataloader_test._data_()
        cluster_result, cluster_msg = Louvain_no_isolation(dataset=test_data, edge_measure=rsn.pred_X,
                                                           weighted=louvain_weighted)

        cluster_eval_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
            print_flag=True, extra_info=True)

        ClusterEvaluationB3Types(test_data_label, cluster_result, test_infos_file,
                                 rel2id_file).printEvaluation()
        print("100 times")

        print({k: v * 100 for k, v in cluster_eval_b3.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--dataset", type=str, default='ori')
    parser.add_argument("--train_data_file", type=str, default='')
    parser.add_argument("--train_for_cluster_file", type=str, default='')
    parser.add_argument("--val_data_file", type=str, default='')
    parser.add_argument("--test_data_file", type=str, default='')
    parser.add_argument("--wordvec_file", type=str, default='../data/wordvec/word_vec.json')

    parser.add_argument("--rel2id_file", type=str, default='../data/support_files/rel2id.json')
    parser.add_argument("--similarity_file", type=str, default='../data/support_files/all_similarity.pkl')
    parser.add_argument("--same_level_pair_file", type=str, default='../data/support_files/same_level_pair.json')
    parser.add_argument("--all_structure_file", type=str, default='../data/support_files/all_structure.json')
    parser.add_argument("--train_structure_file", type=str, default='../data/support_files/train_structure.json')
    parser.add_argument("--test_infos_file", type=str, default='../data/support_files/test_relations_info.json')
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

    parser.add_argument("--val_hier", type=int, default=0)
    parser.add_argument("--golden", type=int, default=0)
    parser.add_argument("--save_cluster", type=int, default=0)

    parser.add_argument("--incre_threshold", type=float, default=0)
    parser.add_argument("--iso_threshold", type=int, default=5)
    parser.add_argument("--partially_order_validation", type=int, default=0)

    parser.add_argument("--same_v_adv", type=int, default=0)
    parser.add_argument("--random_init", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--load_model_name", type=str, default='ori')
    parser.add_argument("--save_model_name", type=str, default='ori/')
    parser.add_argument("--select_cluster", type=int, default=1)
    parser.add_argument("--trainset_loss_type", type=str, default='triplet_v_adv_distance')
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
    args.val_hier = True if args.val_hier == 1 else False
    args.partially_order_validation = True if args.partially_order_validation == 1 else False
    args.golden = True if args.golden == 1 else False
    args.save_cluster = True if args.save_cluster == 1 else False

    if args.dataset == 'ori':
        args.train_data_file = '../data/fewrel_ori/hierarchy_fewrel80_train.json'
        args.train_for_cluster_file = '../data/fewrel_ori/hierarchy_fewrel80_train_100.json'
        args.val_data_file = '../data/fewrel_ori/hierarchy_fewrel80_test_train.json'
        args.test_data_file = '../data/fewrel_ori/hierarchy_fewrel80_test_test.json'

    if args.dataset == 'nytfb':
        args.train_data_file = '../nyt_fb_10_hierarchy/nyt_fb_10_train.json'
        args.val_data_file = '../nyt_fb_10_hierarchy/nyt_fb_10_dev.json'
        args.test_data_file = '../nyt_fb_10_hierarchy/nyt_fb_10_test.json'
        args.rel2id_file = "../nyt_fb_10_hierarchy/rel2id.json"
        args.similarity_file = "../nyt_fb_10_hierarchy/train_similarity.pkl"
        args.same_level_pair_file = "../nyt_fb_10_hierarchy/same_level_pair.json"

    load_cluster(
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        wordvec_file=args.wordvec_file,
        load_model_name=args.load_model_name,
        trainset_loss_type=args.trainset_loss_type,
        testset_loss_type=args.testset_loss_type,
        p_cond=args.p_cond,
        p_denoise=args.p_denoise,
        rel2id_file=args.rel2id_file,
        similarity_file=args.similarity_file,
        same_level_pair_file=args.same_level_pair_file,
        squared=args.distance_squared,
        margin=args.margin,
        louvain_weighted=args.louvain_weighted,
        random_init=args.random_init,
        seed=int(args.seed),
        batch_size=args.batch_size,
        train_structure_file=args.train_structure_file,
        all_structure_file=args.all_structure_file,
        train_for_cluster_file=args.train_for_cluster_file,
        test_infos_file=args.test_infos_file,
        val_hier=args.val_hier,
        golden=args.golden,
        save_cluster=args.save_cluster
    )
