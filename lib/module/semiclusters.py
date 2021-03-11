from .louvain import generate_dendrogram, best_partition, partition_at_level, Status, generate_all_info, \
    __one_level, check_random_state, __modularity
# from louvain import generate_dendrogram, best_partition, partition_at_level, Status, generate_all_info, \
#     __one_level, check_random_state, __modularity

import networkx as nx
import numpy as np
from copy import copy


class HierarchyCluster():
    def __init__(self, rel_wiki_id: str = "", instances: list = None, sons: list = None, fathers: list = None,
                 degree: float = 0, rel_type=""):
        self.rel_wiki_id = rel_wiki_id
        self.instances = instances if instances else []
        # self.instance_num = len(instances)
        self.sons = sons if sons else []
        self.fathers = fathers if fathers else []
        self.degree = degree  # inner connection
        self.rel_type = rel_type
        self.insert_paths = []  # contains tuple (father, avg_link score)

    def __copy__(self):
        return HierarchyCluster(self.rel_wiki_id, self.instances.copy(), self.sons.copy(), self.fathers.copy(),
                                self.degree, self.rel_type)


def __add_degree(train_hierarchy_cluster_list, train_weighted_links):
    train2train_weight_dict = dict()
    for fir_node, sec_node, weight_dict in train_weighted_links:
        weight = weight_dict['weight']
        if weight == 0:  #
            continue
        if fir_node not in train2train_weight_dict.keys():
            train2train_weight_dict[fir_node] = dict()
        if sec_node not in train2train_weight_dict.keys():
            train2train_weight_dict[sec_node] = dict()
        train2train_weight_dict[fir_node][sec_node] = weight
        train2train_weight_dict[sec_node][fir_node] = weight
    for train_cluster in train_hierarchy_cluster_list:
        degree = 0
        # original paper degree
        # for fir_node in train_cluster.instances:
        #     if fir_node not in train2train_weight_dict.keys():
        #         continue
        #     for sec_node in train_cluster.instances:
        #         degree += train2train_weight_dict[fir_node].get(sec_node, 0)  # maybe fir == sec
        # train_cluster.degree = degree

        # this louvain degree
        for fir_node in train_cluster.instances:
            if fir_node not in train2train_weight_dict.keys():
                continue
            degree += sum(train2train_weight_dict[fir_node].values())
            # for sec_node in train_cluster.instances:
            #     degree += train2train_weight_dict[fir_node].get(sec_node, 0)  # maybe fir == sec
        train_cluster.degree = degree
    return


def __new_com_neigh_weights(old_node2com, new_nodes, train_test_links):
    # get internals
    weights_dict = {}
    for link in train_test_links:
        # i from train, j from test
        i, j, weight = link
        if j not in new_nodes:
            continue
        old_com = old_node2com[i]
        weights_dict[old_com] = weights_dict.get(old_com, 0) + weight
    return weights_dict


def __com2node(node2com) -> dict:
    com2node = dict()
    for node, com in node2com.items():
        nodes = com2node.get(com, [])
        nodes.append(node)
        com2node[com] = nodes
    return com2node


# process the answer to the Hierarchy Cluster
def get_hierarchy_cluster_list(data, data_label, dataloader, hierarchy_structure_info, data_start_id,
                               required_relation_type=None):
    if not required_relation_type:
        required_relation_type = ['train', 'test', 'wiki']
    hierarchy_cluster_list = []
    rel_instances_dict = {}
    for rel_id in set(data_label):  # add instances
        rel_instances_dict[rel_id] = []

    for instance_id in range(data_start_id, data_start_id + len(data)):
        rel_instances_dict[data_label[instance_id - data_start_id]].append(instance_id)

    for rel_wiki_id in hierarchy_structure_info.keys():
        rel_type = hierarchy_structure_info[rel_wiki_id]['relation_type']
        if not rel_type in required_relation_type:
            continue
        instances = []

        hierarchy_structure_info[rel_wiki_id]['instances'] = []
        sons = hierarchy_structure_info[rel_wiki_id]['sons']
        fathers = hierarchy_structure_info[rel_wiki_id]['fathers']
        # if required_relation_type == ["test"]:
        #     for rel_id in rel_instances_dict.keys():
        #         if dataloader.id2rel[rel_id] == rel_wiki_id:
        #             instances += rel_instances_dict[rel_id]
        #     hierarchy_cluster_list.append(HierarchyCluster(rel_wiki_id, instances, sons, fathers, rel_type=rel_type))
        # else:
        #     for rel_id in rel_instances_dict.keys():
        #         if dataloader.id2rel[rel_id] in sons or dataloader.id2rel[rel_id] == rel_wiki_id:
        #             instances += rel_instances_dict[rel_id]
        #     hierarchy_cluster_list.append(HierarchyCluster(rel_wiki_id, instances, sons, fathers, rel_type=rel_type))
        for rel_id in rel_instances_dict.keys():
            if dataloader.id2rel[rel_id] in sons or dataloader.id2rel[rel_id] == rel_wiki_id:
                instances += rel_instances_dict[rel_id]
        hierarchy_cluster_list.append(HierarchyCluster(rel_wiki_id, instances, sons, fathers, rel_type=rel_type))
    return hierarchy_cluster_list


def _calculate_distance(dataset, predict_links, edge_measure):
    """calculate train test link, test test link"""
    batch_count = 0
    batch_size = 12000
    left_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=np.int32)
    right_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=np.int32)
    # only for test below 3 lines of code
    # left_data = np.zeros(list((batch_size,) + (1,)), dtype=np.int32)
    # right_data = np.zeros(list((batch_size,) + (1,)), dtype=np.int32)
    # dataset = list(range(len(datase]t)))

    edge_list = []
    print_count = 0
    for count, idx_pair in enumerate(predict_links):  # note this is the similarity need to calculated.
        left_data[batch_count] = dataset[idx_pair[0]]
        right_data[batch_count] = dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:

            temp_edge_list, a, b = edge_measure(left_data, right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
            print_count += 1
            if print_count % 100 == 0:
                print('predicting...', str(round(count / len(predict_links) * 100, 2)) + '%')
    if batch_count != 0:
        # print('predicting...')
        temp_edge_list, a, b = edge_measure(left_data[:batch_count], right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    return edge_list


def __links(train_data_num, test_data_num):
    train_test_links = []
    for i in range(train_data_num + test_data_num):
        # the similarity of trainset is certain.
        if i < train_data_num:
            for j in range(train_data_num, train_data_num + test_data_num):
                train_test_links.append((i, j))
        else:
            continue
    return train_test_links


def dist_measure_for_test(a: np.ndarray, b: np.ndarray):
    return np.random.rand(a.shape[0]), a, b


def __distance_to_weight(distance_list, retain_weight_threshold):
    weight_list = []
    for distance in distance_list:
        weight = 0 if distance >= retain_weight_threshold else 1 - distance
        weight_list.append(weight)

    return weight_list


def __weighted_link_list(links, weights):
    weighted_links = []
    for index, pair in enumerate(links):
        i, j = pair
        if weights[index] > 0:
            weighted_links.append((i, j, {'weight': weights[index]}))
    return weighted_links


def Louvain(test_nodes, weighted_test_links, top_level=1, iso_thres=20):
    graph = nx.Graph()
    graph.add_nodes_from(test_nodes)

    graph.add_edges_from(weighted_test_links)

    dendrogram, graphs, status = generate_all_info(graph)

    partition_level = max(len(dendrogram) - top_level, 0)
    partition = partition_at_level(dendrogram, partition_level)
    test_status = status[partition_level]

    simi_matrix = np.zeros([len(test_nodes), len(test_nodes)])
    train_node_num = test_nodes[0]
    for count, idx_pair in enumerate(weighted_test_links):
        simi_matrix[idx_pair[0] - train_node_num, idx_pair[1] - train_node_num] = weighted_test_links[count][2][
            'weight']
        simi_matrix[idx_pair[1] - train_node_num, idx_pair[0] - train_node_num] = weighted_test_links[count][2][
            'weight']
    label_list = [0] * len(test_nodes)
    for key in partition:
        label_list[key - train_node_num] = partition[key]
    print('solving isolation...')
    cluster_datanum_dict = {}
    reverse_partition_dict = {}
    for id, reltype in enumerate(label_list):
        if reltype in cluster_datanum_dict.keys():
            cluster_datanum_dict[reltype] += 1
            reverse_partition_dict[reltype].append(id + train_node_num)
        else:
            cluster_datanum_dict[reltype] = 1
            reverse_partition_dict[reltype] = []
    iso_reltype_list = []
    for reltype in cluster_datanum_dict:
        if cluster_datanum_dict[reltype] <= iso_thres:
            iso_reltype_list.append(reltype)

    for point_idx, reltype in enumerate(label_list):
        if reltype in iso_reltype_list:
            search_idx_list = np.argsort(simi_matrix[point_idx])[::-1]  # from big to small # it's weight.
            for idx in search_idx_list:
                if label_list[idx] not in iso_reltype_list:
                    label_list[point_idx] = label_list[idx]
                    partition[point_idx + train_node_num] = label_list[idx]
                    test_status.degrees[label_list[idx]] += graph.degree(point_idx + train_node_num, 'weight')
                    break
    return partition, test_status


def insert_into_clusters(test_com2node, train_hierarchy_cluster_list, train_test_weighted_links, test_status,
                         total_graph_weight, resolution=1., incre_threshold=0):
    predict_hierarchy_cluster_list = []

    # 1. process the all weight link
    train2test_node_weight_dict, test2train_node_weight_dict = dict(), dict()
    for train_node, test_node, weight_dict in train_test_weighted_links:
        weight = weight_dict['weight']
        # if weight == 0:  #
        #     continue
        if train_node not in train2test_node_weight_dict.keys():
            train2test_node_weight_dict[train_node] = dict()
        train2test_node_weight_dict[train_node][test_node] = weight

        if test_node not in test2train_node_weight_dict.keys():
            test2train_node_weight_dict[test_node] = dict()
        test2train_node_weight_dict[test_node][train_node] = weight

    for com_key, nodes in test_com2node.items():
        test_cluster = HierarchyCluster(rel_wiki_id=com_key, instances=nodes)
        best_increase = incre_threshold
        best_cluster = None
        # test_com's graph degrees
        com_graph_degree = 0
        for test_node in nodes:
            if test_node in test2train_node_weight_dict.keys():
                com_graph_degree += sum(test2train_node_weight_dict[test_node].values())
        # com_graph_degree = sum()
        degc_totw = com_graph_degree / (total_graph_weight * 2.)
        remove_cost = (test_status.degrees[com_key] - com_graph_degree) * degc_totw
        # test community inner degree
        # test_status.degrees[com_key]
        for train_cluster in train_hierarchy_cluster_list:
            # calculate modularity delta.
            # dnc is the connect weights of other community
            dnc = 0
            for test_node in nodes:
                if test_node in test2train_node_weight_dict.keys():
                    for train_node in train_cluster.instances:
                        dnc += test2train_node_weight_dict[test_node].get(train_node, 0)
            incr = remove_cost + resolution * dnc - train_cluster.degree * degc_totw
            if incr > best_increase:
                best_increase = incr
                best_cluster = train_cluster

        if best_increase > 0:  # find father in all train trees.
            print("best_increase", best_increase)
            test_cluster.fathers = best_cluster.fathers + [best_cluster.rel_wiki_id]
            # print("\n test community added")

        predict_hierarchy_cluster_list.append(test_cluster)
    return predict_hierarchy_cluster_list


def prepare_cluster_list(train_dataloader, test_dataloader, train_hierarchy_structure_info,
                         all_hierarchy_structure_info, to_cluster_data_num):
    train_data, train_label = train_dataloader._part_data_(to_cluster_data_num)
    test_data, test_label = test_dataloader._part_data_(to_cluster_data_num)
    train_data_num, test_data_num, all_data_num = len(train_label), len(test_label), len(train_label) + len(test_label)
    train_data_ids, test_data_ids = list(range(train_data_num)), list(range(train_data_num, all_data_num))
    train_hierarchy_cluster_list = get_hierarchy_cluster_list(train_data_ids, train_label, train_dataloader,
                                                              train_hierarchy_structure_info, 0)
    gt_hierarchy_cluster_list = get_hierarchy_cluster_list(test_data_ids, test_label, test_dataloader,
                                                           all_hierarchy_structure_info, train_data_num, ['test'])

    return train_hierarchy_cluster_list, gt_hierarchy_cluster_list, train_data_num, test_data_num, train_data, train_label, test_data, test_label


def insert_into_clusters_avg_link(test_com2node, train_hierarchy_cluster_list, train_test_weighted_links, link_th=0.5):
    predict_hierarchy_cluster_list = []

    # 1. process the all weight link
    train2test_node_weight_dict, test2train_node_weight_dict = dict(), dict()
    for train_node, test_node, weight_dict in train_test_weighted_links:
        weight = weight_dict['weight']
        if train_node not in train2test_node_weight_dict.keys():
            train2test_node_weight_dict[train_node] = dict()
        train2test_node_weight_dict[train_node][test_node] = weight

        if test_node not in test2train_node_weight_dict.keys():
            test2train_node_weight_dict[test_node] = dict()
        test2train_node_weight_dict[test_node][train_node] = weight

    for com_key, nodes in test_com2node.items():
        test_cluster = HierarchyCluster(rel_wiki_id=com_key, instances=nodes)
        best_cluster = None
        best_link = 0
        # test_com's graph degrees
        # com_graph_degree = 0
        # for test_node in nodes:
        #     if test_node in test2train_node_weight_dict.keys():
        #         com_graph_degree += sum(test2train_node_weight_dict[test_node].values())
        for train_cluster in train_hierarchy_cluster_list:
            # calculate modularity delta.
            # dnc is the connect weights of other community
            dnc = 0
            for test_node in nodes:
                if test_node in test2train_node_weight_dict.keys():
                    for train_node in train_cluster.instances:
                        dnc += test2train_node_weight_dict[test_node].get(train_node, 0)
            avg_link = dnc / (len(train_cluster.instances) * len(nodes))
            if avg_link > best_link:
                best_cluster = train_cluster
                best_link = avg_link

        if best_cluster:  # find father in all train trees.
            print("best link", best_link)
            test_cluster.fathers = best_cluster.fathers + [best_cluster.rel_wiki_id]
        predict_hierarchy_cluster_list.append(test_cluster)
    return predict_hierarchy_cluster_list


def __list2node(cluster_result, train_data_num) -> dict:
    com2node = dict()
    for node, com in enumerate(cluster_result):
        nodes = com2node.get(com, [])
        nodes.append(node + train_data_num)
        com2node[com] = nodes
    return com2node


def top_down_insert_into_clusters_avg_link(test_com2node, train_hierarchy_cluster_list, train_test_weighted_links,
                                           link_th):
    predict_hierarchy_cluster_list = []

    # 1. process the all weight link
    train2test_node_weight_dict, test2train_node_weight_dict = dict(), dict()
    for train_node, test_node, weight_dict in train_test_weighted_links:
        weight = weight_dict['weight']
        if train_node not in train2test_node_weight_dict.keys():
            train2test_node_weight_dict[train_node] = dict()
        train2test_node_weight_dict[train_node][test_node] = weight

        if test_node not in test2train_node_weight_dict.keys():
            test2train_node_weight_dict[test_node] = dict()
        test2train_node_weight_dict[test_node][train_node] = weight

    top_level_clusters = []
    for train_cluster in train_hierarchy_cluster_list:
        # if len(train_cluster.fathers) and len(train_cluster.sons) == 0: # drop single relation
        if len(train_cluster.fathers) == 0:
            top_level_clusters.append(train_cluster)

    for com_key, nodes in test_com2node.items():
        test_cluster = HierarchyCluster(rel_wiki_id=com_key, instances=nodes)
        best_cluster = None
        best_link = 0
        search_flag = True
        level_clusters = [copy(c) for c in top_level_clusters]
        score_list = []
        while search_flag:
            search_flag = False
            for train_cluster in level_clusters:
                dnc = 0
                for test_node in nodes:
                    if test_node in test2train_node_weight_dict.keys():
                        for train_node in train_cluster.instances:
                            dnc += test2train_node_weight_dict[test_node].get(train_node, 0)
                # avg_link = dnc / (len(train_cluster.instances) * len(nodes)) * (1 + len(train_cluster.sons))
                avg_link = dnc / (len(train_cluster.instances) * len(nodes)) * np.sqrt((1 + len(train_cluster.sons)))

                score_list.append((train_cluster.rel_wiki_id, avg_link))
                if avg_link > best_link:
                    best_cluster = train_cluster
                    best_link = avg_link
                    search_flag = True  # can continue search
            sorted_score = sorted(score_list, key=lambda x: x[1], reverse=True)  # score from high to low.
            test_cluster.insert_paths.append(sorted_score)
            if search_flag:
                level_clusters = []

                for rel, s in sorted_score[:1]:
                    for train_cluster in train_hierarchy_cluster_list:
                        if train_cluster.rel_wiki_id == rel:
                            cluster = train_cluster
                            # break
                            for train_cluster in train_hierarchy_cluster_list:
                                if train_cluster.rel_wiki_id in cluster.sons:  # search sons.
                                    level_clusters.append(train_cluster)
                score_list = []
                # single search
                # for train_cluster in train_hierarchy_cluster_list:
                #     if train_cluster.rel_wiki_id in best_cluster.sons:  # search sons.
                #         level_clusters.append(train_cluster)

        if best_cluster and best_link > link_th:  # find father in all train trees.
            print("best link", best_link)
            test_cluster.fathers = best_cluster.fathers + [best_cluster.rel_wiki_id]
            print("test_cluster.insert_paths", test_cluster.insert_paths)
        predict_hierarchy_cluster_list.append(test_cluster)
    return predict_hierarchy_cluster_list


def Louvain_with_test_cluster_done_avg_link_list(cluster_result, train_data_num, test_data_num, train_data, test_data,
                                                 train_hierarchy_cluster_list, edge_measure, link_th_list):
    # only get test.
    # prepare links and corresponding distances

    train_test_links = __links(train_data_num, test_data_num)
    train_test_dist_list = _calculate_distance(train_data + test_data, train_test_links, edge_measure)

    # transfer distance to weight
    train_test_weight_list = __distance_to_weight(train_test_dist_list, 0.5)
    train_test_weighted_links = __weighted_link_list(train_test_links, train_test_weight_list)

    all_result = []
    com2node = __list2node(cluster_result, train_data_num)
    for link_th in link_th_list:
        one_result_dict = {}
        one_result_dict['iso'] = 5
        one_result_dict['link_th'] = link_th

        predicted_cluster_list = insert_into_clusters_avg_link(com2node, train_hierarchy_cluster_list,
                                                               train_test_weighted_links, link_th)
        one_result_dict['list'] = predicted_cluster_list

        all_result.append(one_result_dict)
    return all_result


def Top_Down_Louvain_with_test_cluster_done_avg_link_list(cluster_result, train_data_num, test_data_num, train_data,
                                                          test_data,
                                                          train_hierarchy_cluster_list, edge_measure, link_th_list):
    # only get test.
    # prepare links and corresponding distances

    train_test_links = __links(train_data_num, test_data_num)
    train_test_dist_list = _calculate_distance(train_data + test_data, train_test_links, edge_measure)

    # transfer distance to weight
    train_test_weight_list = __distance_to_weight(train_test_dist_list, 0.5)
    train_test_weighted_links = __weighted_link_list(train_test_links, train_test_weight_list)

    all_result = []
    com2node = __list2node(cluster_result, train_data_num)
    # import pickle
    # pickle.dump(train_hierarchy_cluster_list, open("train_hierarchy_cluster_list", "wb"))
    # pickle.dump(train_test_weighted_links, open("train_test_weighted_links", "wb"))
    # pickle.dump(com2node, open("com2node", "wb"))
    for link_th in link_th_list:
        one_result_dict = {}
        one_result_dict['iso'] = 5
        one_result_dict['link_th'] = link_th

        predicted_cluster_list = top_down_insert_into_clusters_avg_link(com2node, train_hierarchy_cluster_list,
                                                                        train_test_weighted_links, link_th)
        one_result_dict['list'] = predicted_cluster_list

        all_result.append(one_result_dict)
    return all_result


def Top_Down_Louvain_with_test_cluster_done_avg_link_list_golden(gt_hierarchy_cluster_list, train_data_num,
                                                                 test_data_num, train_data,
                                                                 test_data,
                                                                 train_hierarchy_cluster_list, edge_measure,
                                                                 link_th_list):
    # only get test.
    # prepare links and corresponding distances

    train_test_links = __links(train_data_num, test_data_num)
    train_test_dist_list = _calculate_distance(train_data + test_data, train_test_links, edge_measure)

    # transfer distance to weight
    train_test_weight_list = __distance_to_weight(train_test_dist_list, 0.5)
    train_test_weighted_links = __weighted_link_list(train_test_links, train_test_weight_list)

    all_result = []
    com2node = dict()

    for cluster in gt_hierarchy_cluster_list:
        com2node[cluster.rel_wiki_id] = cluster.instances

    # import pickle
    # pickle.dump(train_hierarchy_cluster_list, open("train_hierarchy_cluster_list", "wb"))
    # pickle.dump(train_test_weighted_links, open("train_test_weighted_links", "wb"))
    # pickle.dump(com2node, open("com2node", "wb"))
    for link_th in link_th_list:
        one_result_dict = {}
        one_result_dict['iso'] = 5
        one_result_dict['link_th'] = link_th

        predicted_cluster_list = top_down_insert_into_clusters_avg_link(com2node, train_hierarchy_cluster_list,
                                                                        train_test_weighted_links, link_th)
        one_result_dict['list'] = predicted_cluster_list

        all_result.append(one_result_dict)
    return all_result
