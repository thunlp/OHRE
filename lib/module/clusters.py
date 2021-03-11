import numpy as np
from copy import deepcopy
import networkx as nx
import community


def find_close(M):
    s_index, l_index = 0, 0
    min_list = np.zeros([len(M)], dtype=np.float32)
    min_index_list = np.zeros([len(M)], dtype=np.int32)
    for i, item in enumerate(M):
        if len(item):
            temp_min = min(item)
            min_list[i] = temp_min
            min_index_list[i] = item.index(temp_min)
        else:
            min_list[i] = 10000
    l_index = int(np.where(min_list == np.min(min_list))[0][0])
    s_index = min_index_list[l_index]
    return s_index, l_index  # s_index < l_index


# model
def complete_HAC(dataset, HAC_dist, k, datatype=np.int32):
    # initialize C and M, C is a list of clusters, M is a list as dist_matrix

    print('the len of dataset to cluster is:' + str(len(dataset)))
    print('initializing...')
    idx_C, M, idxM = [], [], []
    for i, item in enumerate(dataset):
        idx_Ci = [i]
        idx_C.append(idx_Ci)

    print('initializing dist_matrix...')
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(idx_C)):
        for j in range(len(idx_C)):
            if j == i:
                break
            idx_list.append([i, j])

    print('calculating dist_list...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    dist_list = []
    for count, idx_pair in enumerate(idx_list):
        left_data[batch_count] = dataset[idx_pair[0]]
        right_data[batch_count] = dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting', str(round(count / len(idx_list) * 100, 2)) + '%')
            temp_dist_list = HAC_dist(left_data, right_data)
            dist_list = dist_list + temp_dist_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count != 0:
        print('predicting...')
        temp_dist_list = HAC_dist(left_data[:batch_count], right_data[:batch_count])
        dist_list = dist_list + temp_dist_list.reshape(batch_count).tolist()

    print('preparing dist_matrix...')
    count = 0
    for i in range(len(idx_C)):
        Mi = []
        for j in range(len(idx_C)):
            if j == i:
                break
            Mi.append(dist_list[count])
            count += 1
        M.append(Mi)

    # combine two classes
    q = len(idx_C)
    while q > k:
        s_index, l_index = find_close(M)
        idx_C[s_index].extend(idx_C[l_index])
        del idx_C[l_index]

        M_next = deepcopy(M[:-1])
        for i in range(len(idx_C)):
            for j in range(len(idx_C)):
                if j == i:
                    break

                i_old, j_old = i, j
                if i >= l_index:
                    i_old = i + 1
                if j >= l_index:
                    j_old = j + 1

                if i != s_index and j != s_index:
                    M_next[i][j] = M[i_old][j_old]
                elif i == s_index:
                    M_next[i][j] = max(M[s_index][j_old], M[l_index][j_old])
                elif j == s_index:
                    if i_old < l_index:
                        M_next[i][j] = max(M[i_old][s_index], M[l_index][i_old])
                    elif i_old > l_index:
                        M_next[i][j] = max(M[i_old][s_index], M[i_old][l_index])
        q -= 1
        print('temp cluster num is:', q, ',', s_index, 'and', l_index, 'are combined, metric is:', M[l_index][s_index])
        M = M_next

    # decode to get label_list
    label_list = [0] * len(dataset)
    for label, temp_cluster in enumerate(idx_C):
        for idx in temp_cluster:
            label_list[idx] = label

    return label_list, create_msg(label_list)


def Louvain(dataset, edge_measure, datatype=np.int32):
    print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i, j))

    print('calculating edges...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    edge_list = []
    for count, idx_pair in enumerate(idx_list):
        left_data[batch_count] = dataset[idx_pair[0]]
        right_data[batch_count] = dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...', str(round(count / len(idx_list) * 100, 2)) + '%')
            temp_edge_list = edge_measure(left_data, right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count != 0:
        print('predicting...')
        temp_edge_list = edge_measure(left_data[:batch_count], right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    edge_list = np.int32(np.round(edge_list))

    print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if edge_list[i] == 0:
            true_edge_list.append(idx_list[i])
    g.add_edges_from(true_edge_list)

    print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    print('decoding to get label_list...')
    label_list = [0] * len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    return label_list, create_msg(label_list)


def Louvain_no_isolation(dataset, edge_measure, datatype=np.int32, iso_thres=5, weighted=False):
    # print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    # print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i, j))

    # print('calculating edges...')
    batch_count = 0
    print_count = 0
    batch_size = 12000
    left_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,) + dataset[0].shape), dtype=datatype)
    edge_list = []
    for count, idx_pair in enumerate(idx_list):
        left_data[batch_count] = dataset[idx_pair[0]]
        right_data[batch_count] = dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:

            temp_edge_list, a, b = edge_measure(left_data, right_data)

            # print(temp_edge_list)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
            print_count += 1
            # if print_count % 100 == 0:
            #     print('predicting...', str(round(count / len(idx_list) * 100, 2)) + '%')
    if batch_count != 0:
        # print('predicting...')
        temp_edge_list, a, b = edge_measure(left_data[:batch_count], right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    simi_list, edge_list = np.array(edge_list), np.array(edge_list)
    simi_list[simi_list > 1] = 1
    edge_list[edge_list > 1] = 1
    if not weighted:
        edge_list[edge_list >= 0.5] = 1
        edge_list[edge_list < 0.5] = 0
    edge_list = edge_list.tolist()
    simi_list = simi_list.tolist()

    # ------------------
    # print('forming simi_matrix...')
    simi_matrix = np.zeros([len(dataset), len(dataset)])
    for count, idx_pair in enumerate(idx_list):
        simi_matrix[idx_pair[0], idx_pair[1]] = simi_list[count]
        simi_matrix[idx_pair[1], idx_pair[0]] = simi_list[count]
    # ------------------

    # print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if not weighted:
            if edge_list[i] == 0:
                true_edge_list.append(idx_list[i])
        else:
            if edge_list[i] < 0.5:
                true_edge_list.append((idx_list[i][0], idx_list[i][1], {'weight': 1 - edge_list[i]}))

    g.add_edges_from(true_edge_list)

    # print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    # print('decoding to get label_list...')
    label_list = [0] * len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    # ------------------
    # print('solving isolation...')
    cluster_datanum_dict = {}
    for reltype in label_list:
        if reltype in cluster_datanum_dict.keys():
            cluster_datanum_dict[reltype] += 1
        else:
            cluster_datanum_dict[reltype] = 1

    iso_reltype_list = []
    for reltype in cluster_datanum_dict:
        if cluster_datanum_dict[reltype] <= iso_thres:
            iso_reltype_list.append(reltype)

    for point_idx, reltype in enumerate(label_list):
        if reltype in iso_reltype_list:
            search_idx_list = np.argsort(simi_matrix[point_idx])  # from small to big
            for idx in search_idx_list:
                if label_list[idx] not in iso_reltype_list:
                    label_list[point_idx] = label_list[idx]
                    break
    # ------------------

    return label_list, create_msg(label_list)


def Hierarchical_Louvain(train_dataset: list, test_dataset: list, trainset_relids, edge_measure,
                         get_two_relation_distance, datatype=np.int32, iso_thres=5, weighted=True):
    """

    :param train_dataset: all data instances in train set.
    :param test_dataset: all data instances in test set.
    :param trainset_relids: the relation ids in trainset. the length is equal to train_dataset.
    :param edge_measure: function to calculate similarity between two instances.
    :param get_two_relation_distance: function to get distance of two relation.
    :param datatype:
    :param iso_thres: least num of instances to support a cluster
    :param weighted: if similarity is only 0 or 1, or weighted.
    :return: hierarchy info
    """
    print('initializing the graph...')
    g = nx.Graph()

    train_and_test_dataset = train_dataset.copy() + test_dataset.copy()

    # train set instance similarities are the shortest path related metric.
    # test set instance similarities are calculated based on the RSN model.

    g.add_nodes_from(np.arange(len(train_and_test_dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(train_and_test_dataset)):
        # the similarity of trainset is certain.
        left_boundary = max(len(train_dataset), i + 1)
        for j in range(left_boundary, len(train_and_test_dataset)):
            idx_list.append((i, j))
    train_idx_list = []
    for i in range(len(train_dataset)):
        for j in range(i + 1, len(train_dataset)):
            train_idx_list.append((i, j))

    print("Forming distance of trainset...")
    train_edge_list = []
    for count, trainset_pairs in enumerate(train_idx_list):
        instance1, instance2 = trainset_pairs
        rel1, rel2 = trainset_relids[instance1], trainset_relids[instance2]
        train_edge_list.append(get_two_relation_distance(rel1, rel2))

    print('calculating edges...')
    batch_count = 0
    batch_size = 5000
    left_data = np.zeros(list((batch_size,) + train_and_test_dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,) + train_and_test_dataset[0].shape), dtype=datatype)
    # only for test
    # left_data = np.zeros(list((batch_size,) + (1,)), dtype=datatype)
    # right_data = np.zeros(list((batch_size,) + (1,)), dtype=datatype)
    edge_list = []
    for count, idx_pair in enumerate(idx_list):  # note this is the similarity need to calculated.
        left_data[batch_count] = train_and_test_dataset[idx_pair[0]]
        right_data[batch_count] = train_and_test_dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...', str(round(count / len(idx_list) * 100, 2)) + '%')
            temp_edge_list, a, b = edge_measure(left_data, right_data)

            # print(temp_edge_list)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count != 0:
        print('predicting...')
        temp_edge_list, a, b = edge_measure(left_data[:batch_count], right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()

    distance_list, edge_list = np.array(edge_list), np.array(edge_list)
    distance_list[distance_list > 1] = 1
    edge_list[edge_list > 1] = 1
    if not weighted:
        edge_list[edge_list >= 0.5] = 1
        edge_list[edge_list < 0.5] = 0
    edge_list = edge_list.tolist()
    distance_list = distance_list.tolist()

    # ------------------

    print('forming simi_matrix...')
    simi_matrix = np.zeros([len(train_and_test_dataset), len(train_and_test_dataset)])
    for count, idx_pair in enumerate(idx_list):
        simi_matrix[idx_pair[0], idx_pair[1]] = distance_list[count]
        simi_matrix[idx_pair[1], idx_pair[0]] = distance_list[count]

    for count, idx_pair in enumerate(train_idx_list):
        simi_matrix[idx_pair[0], idx_pair[1]] = train_edge_list[count]
        simi_matrix[idx_pair[1], idx_pair[0]] = train_edge_list[count]

    # ------------------

    print('adding edges...')
    true_edge_list = []

    for i in range(len(idx_list)):
        if not weighted:
            if edge_list[i] == 0:
                true_edge_list.append(idx_list[i])
        else:
            if edge_list[i] < 0.5:
                true_edge_list.append((idx_list[i][0], idx_list[i][1], {'weight': 1 - edge_list[i]}))
    # add train edges.
    for i in range(len(train_idx_list)):
        if train_edge_list[i] < 1:
            true_edge_list.append((train_idx_list[i][0], train_idx_list[i][1], {'weight': 1 - train_edge_list[i]}))

    g.add_edges_from(true_edge_list)

    print('Clustering...')
    # list of clustering hierarchy.
    dendrogram = community.generate_dendrogram(g)
    new_dendrogram = []
    # the lowest level, do the isolation thing, etc.

    # lowest_partition = dendrogram[0]

    # todo check done here!
    min_level = 100
    for level in range(len(dendrogram)):
        if len(dendrogram[level].keys()) < 100:  # start cut and regard as clusters.
            min_level = level - 1
    # at least two levels.
    back_second_level = len(dendrogram) - 1  # the second top level cluster results.
    dendrogram_level = min(min_level, back_second_level)
    dendrogram_level = max(dendrogram_level, 0)  # in case only 1 level of clusters.
    new_lowest_partition = community.partition_at_level(dendrogram, dendrogram_level)
    new_dendrogram.append(new_lowest_partition)
    for level in range(dendrogram_level + 1, len(dendrogram)):
        new_dendrogram.append(dendrogram[level])
    if len(dendrogram) != len(new_dendrogram):
        print("dendrogram changed")
    # decode to get label_list
    # print('solving the lowest level...')
    # # solve the isolation in the lowest level.
    # label_list = [0] * len(train_and_test_dataset)
    # for key in lowest_partition.keys():
    #     label_list[key] = lowest_partition[key]

    # ------------------

    # print('solving isolation...')
    # cluster_datanum_dict = {}
    # for reltype in label_list:
    #     if reltype in cluster_datanum_dict.keys():
    #         cluster_datanum_dict[reltype] += 1
    #     else:
    #         cluster_datanum_dict[reltype] = 1
    #
    # iso_reltype_list = []
    # for reltype in cluster_datanum_dict:
    #     if cluster_datanum_dict[reltype] <= iso_thres:
    #         iso_reltype_list.append(reltype)
    #
    # for point_idx, reltype in enumerate(label_list):
    #     if reltype in iso_reltype_list:
    #         search_idx_list = np.argsort(simi_matrix[point_idx])  # from small to big
    #         for idx in search_idx_list:
    #             if label_list[idx] not in iso_reltype_list:
    #                 label_list[point_idx] = label_list[idx]
    #                 break
    # new_dendrogram.append(label_list)  # after isolation
    # # solve the higher level of clustering results.
    # # todo check this.
    # # move the relation in iso_reltype_list
    # if len(dendrogram) > 1 and len(iso_reltype_list) > 0:
    #     clusters_to_check = iso_reltype_list
    #     for level in range(1, len(dendrogram)):
    #         # check if the higher level community only have isolation community.
    #         new_clusters_to_check = []
    #         one_level_clustering = dendrogram[level]
    #         reverse_clusters = dict()
    #         for k, v in one_level_clustering.items():
    #             reverse_clusters[v] = reverse_clusters.get(v, [])
    #             reverse_clusters[v].append(k)
    #
    #         for cluster_id, low_communities in reverse_clusters.items():
    #             for one_low_community in low_communities:
    #                 if one_low_community not in clusters_to_check:
    #                     break
    #             else:  # all communities in this high level cluster is a isolation in the low level.
    #                 new_clusters_to_check.append(cluster_id)
    #
    #         label_list = [0] * len(one_level_clustering)
    #         for key in one_level_clustering:
    #             label_list[key] = one_level_clustering[key]
    #         new_dendrogram.append(label_list)
    #         if len(new_clusters_to_check) > 0:
    #             clusters_to_check = new_clusters_to_check
    #         else:
    #             break
    #
    # for level in range(len(new_dendrogram), len(dendrogram)):
    #     label_list = [0] * len(dendrogram[level])
    #     for key in dendrogram[level].keys():
    #         label_list[key] = dendrogram[level][key]
    #     new_dendrogram.append(label_list)

    # todo solve the hierarchical structure.
    louvain_cluster_info = dict()

    # prepare all information in hierarchy structure.
    reverse_dendrogram = []  # list of dict in which {'cluster_id':[instances of last level]}
    louvain_cluster_dicts = []
    cluster_mapping_dicts = []
    cluster_count = 0
    cluster_sons_dicts = []
    reverse_louvain_cluster_dicts = []
    for level in range(len(new_dendrogram)):
        reverse_clusters = dict()
        one_relation_mapping_dict = dict()
        one_level_cluster_sons = dict()
        for instance_id, cluster_id in new_dendrogram[level].items():
            reverse_clusters[cluster_id] = reverse_clusters.get(cluster_id, [])
            reverse_clusters[cluster_id].append(instance_id)
        reverse_dendrogram.append(reverse_clusters)
        louvain_cluster_dicts.append(community.partition_at_level(new_dendrogram, level))

        reverse_louvain_clusters = dict()
        for instance_id, cluster_id in louvain_cluster_dicts[-1].items():
            reverse_louvain_clusters[cluster_id] = reverse_louvain_clusters.get(cluster_id, [])
            reverse_louvain_clusters[cluster_id].append(instance_id)
        reverse_louvain_cluster_dicts.append(reverse_louvain_clusters)
        for k in reverse_clusters.keys():
            one_relation_mapping_dict[k] = cluster_count
            one_level_cluster_sons[k] = []
            cluster_count += 1
        cluster_mapping_dicts.append(one_relation_mapping_dict)
        if level == 0:
            cluster_sons_dicts.append(one_level_cluster_sons)
        else:
            cluster_sons_dicts.append(
                trace_sons(level, reverse_clusters, cluster_mapping_dicts, cluster_sons_dicts[-1]))

        # cluster_sons_dicts.append(one_level_cluster_sons)

    for level in range(len(new_dendrogram)):
        reverse_clusters = reverse_dendrogram[level]
        for cluster_id_in_this_level, instances in reverse_clusters.items():
            real_cluster_id = cluster_mapping_dicts[level][cluster_id_in_this_level]
            louvain_cluster_info[real_cluster_id] = dict()
            louvain_cluster_info[real_cluster_id]['fathers'] = trace_fathers(level, cluster_id_in_this_level,
                                                                             reverse_dendrogram, cluster_mapping_dicts)
            louvain_cluster_info[real_cluster_id]['sons'] = cluster_sons_dicts[level][cluster_id_in_this_level]
            louvain_cluster_info[real_cluster_id]['instances'] = reverse_louvain_cluster_dicts[level][
                cluster_id_in_this_level]
            louvain_cluster_info[real_cluster_id]['level'] = level
            louvain_cluster_info[real_cluster_id]['cluster_id_in_ori_level'] = cluster_id_in_this_level

    return louvain_cluster_info


def trace_fathers(level, cluster_id_in_this_level, reverse_dendrogram, cluster_mapping_dicts) -> list:
    if level + 1 == len(reverse_dendrogram):
        return []  # no fathers.

    for cluster_id_in_higher_level, instances in reverse_dendrogram[level + 1].items():
        if cluster_id_in_this_level in instances:
            absolute_cluster_id = cluster_mapping_dicts[level + 1][cluster_id_in_higher_level]
            return [absolute_cluster_id] + trace_fathers(level + 1, cluster_id_in_higher_level, reverse_dendrogram,
                                                         cluster_mapping_dicts)


def trace_sons(level, reverse_clusters, cluster_mapping_dicts, last_previous_level_cluster_sons):
    sons = dict()
    for cluster_id_in_this_level, instances in reverse_clusters.items():
        sons[cluster_id_in_this_level] = []
        for one_instance in instances:
            sons[cluster_id_in_this_level].append(cluster_mapping_dicts[level - 1][one_instance])
            # add previous sons.
            sons[cluster_id_in_this_level] += last_previous_level_cluster_sons[one_instance]
    return sons


def create_msg(label_list):
    print('creating cluster messages...')
    msg = {}
    msg['num_of_nodes'] = len(label_list)
    msg['num_of_clusters'] = len(list(set(label_list)))
    msg['num_of_data_in_clusters'] = {}
    for reltype in label_list:
        try:
            msg['num_of_data_in_clusters'][reltype] += 1
        except:
            msg['num_of_data_in_clusters'][reltype] = 1

    return msg
