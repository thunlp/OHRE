import math
import numpy as np
import json
from sklearn import metrics
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
                                self.degree)


class ClusterEvaluation:
    '''
    groundtruthlabels and predicted_clusters should be two list, for example:
    groundtruthlabels = [0, 0, 1, 1], that means the 0th and 1th data is in cluster 0,
    and the 2th and 3th data is in cluster 1
    '''

    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet = self.createGroundTruthSets(groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)
        self.groundtruthlabels = groundtruthlabels
        self.predicted_clusters = predicted_clusters

    def createGroundTruthSets(self, labels):

        groundtruthsets = {}
        assessableElems = set()

        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)

        return groundtruthsets, assessableElems

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalPrecision += self.b3precision(self.predictedsets[c],
                                                   self.findCluster(r, self.groundtruthsets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalRecall += self.b3recall(self.predictedsets[c], self.findCluster(r, self.groundtruthsets))

        return totalRecall / float(len(self.assessableElemSet))

    def findCluster(self, a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    def printEvaluation(self, print_flag=True, extra_info=True):

        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1 + betasquare) * recB3 * precB3) / ((betasquare * precB3) + recB3)

        m = {'F1': F1B3, 'F0.5': F05B3, 'precision': precB3, 'recall': recB3}
        if print_flag:
            print("B3 Info:")
            print("B3 Precision(%);B3 Recall(%);B3 F1(%); B3 F05(%)")
            print(precB3 * 100, "; ", recB3 * 100, "; ", F1B3 * 100, "; ", F05B3 * 100)
        if extra_info:
            labels = self.groundtruthlabels
            pred = self.predicted_clusters
            del m['F0.5']
            m['v_measure'] = metrics.v_measure_score(labels, pred)
            m['homogeneity_score'] = metrics.homogeneity_score(labels, pred)
            m['completeness_score'] = metrics.completeness_score(labels, pred)
            m['ARI'] = metrics.adjusted_rand_score(labels, pred)
        return m

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1 + betasquare) * recB3 * precB3) / ((betasquare * precB3) + recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3


class ClusterEvaluationB3Types:
    def __init__(self, groundtruthlabels, predicted_clusters, test_relation_infos_file: str, rel2id_file: str):
        self.groundtruthlabels = groundtruthlabels
        self.predicted_clusters = predicted_clusters
        self.test_relation_infos = json.load(open(test_relation_infos_file))
        self.rel2id = json.load(open(rel2id_file))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

    def printEvaluation(self):
        for type, rels in self.test_relation_infos.items():
            data_points_in_this_types = []
            for data_index, label in enumerate(self.groundtruthlabels):
                if self.id2rel[label] in rels:  # this data point is this relation type.
                    data_points_in_this_types.append(data_index)
            new_pred_clusters = [self.predicted_clusters[index] for index in data_points_in_this_types]
            new_gt_labels = [self.groundtruthlabels[index] for index in data_points_in_this_types]
            eval = ClusterEvaluation(new_gt_labels, new_pred_clusters).printEvaluation(False, False)
            print("type:", type)
            print("metric:", {k: v * 100 for k, v in eval.items()})
        return


class ClusterEvaluationNew:
    '''
    groundtruthlabels and predicted_clusters should be two list, for example:
    groundtruthlabels = [0, 0, 1, 1], that means the 0th and 1th data is in cluster 0,
    and the 2th and 3th data is in cluster 1
    '''

    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet, self.groundtruthMatched = self.createGroundTruthSets(
            groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)
        # mapping keys to column and row index.
        self.F1_matrix = np.zeros((len(self.predictedsets.keys()), len(self.groundtruthsets.keys())))
        self.predicted_cluster2row, self.gt_cluster2column = self.createMappingDict()
        self.row2predicted_cluster = {v: k for k, v in self.predicted_cluster2row.items()}
        self.column2gt_cluster = {v: k for k, v in self.gt_cluster2column.items()}

    def createMappingDict(self):
        predicted_cluster2row = dict()
        predicted_cluster_count = 0
        gt_cluster2column = dict()
        gt_cluster_count = 0
        for key in self.predictedsets.keys():
            predicted_cluster2row[key] = predicted_cluster_count
            predicted_cluster_count += 1

        for key in self.groundtruthsets.keys():
            gt_cluster2column[key] = gt_cluster_count
            gt_cluster_count += 1
        return predicted_cluster2row, gt_cluster2column

    def createGroundTruthSets(self, labels):

        groundtruthsets = {}
        assessableElems = set()
        groundtruthMatched = dict()
        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)
        for key in groundtruthsets.keys():
            groundtruthMatched[key] = False

        return groundtruthsets, assessableElems, groundtruthMatched

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def matchCluster(self, c):
        """

        :param c: the predicted cluster.
        :return: the matched ground truth cluster
        """
        match_dict = dict()

        for key, value in self.groundtruthsets.items():
            match_dict[key] = 0
            p = self.precision(c, value)
            r = self.recall(c, value)
            f1 = 2 * r * p / (p + r) if p + r > 0 else 0
            match_dict[key] = f1

        # sort f1 score
        sorted_match_items = sorted(list(match_dict.items()), key=lambda x: x[1], reverse=True)
        for key, f1 in sorted_match_items:
            if not self.groundtruthMatched[key]:
                self.groundtruthMatched[key] = True
                return self.groundtruthsets[key]
        else:  # not searched.
            return set()

    def precision(self, response_a, reference_a):
        if len(response_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(response_a))

    def recall(self, response_a, reference_a):
        if len(reference_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def calculate_all_f1(self):
        for p_key, p_cluster in self.predictedsets.items():
            for gt_key, gt_cluster in self.groundtruthsets.items():
                p = self.precision(p_cluster, gt_cluster)
                r = self.recall(p_cluster, gt_cluster)
                f1 = 2 * r * p / (p + r) if p + r > 0 else 0
                self.F1_matrix[self.predicted_cluster2row[p_key], self.gt_cluster2column[gt_key]] = f1

    def TotalElementF1(self):
        totalF1 = 0.0
        # calculate all F1
        self.calculate_all_f1()

        # get max
        sorted_f1_matrix = np.sort(self.F1_matrix, axis=1)
        arg_sorted_f1_matrix = np.argsort(self.F1_matrix, axis=1)
        # from small to big. the last column is the highest F1 score for each predicted set.
        each_predicted_set_max_f1_score = sorted_f1_matrix[:, -1]
        # from big to small , highest score
        arg_sorted_predicted_max_f1_score = np.argsort(each_predicted_set_max_f1_score)[::-1]

        for row_index in arg_sorted_predicted_max_f1_score:
            for col_index in arg_sorted_f1_matrix[row_index][::-1]:  # select from high to low
                if not self.groundtruthMatched[self.column2gt_cluster[col_index]]:  # not matched before
                    f1_score = self.F1_matrix[row_index][col_index]
                    self.groundtruthMatched[self.column2gt_cluster[col_index]] = True
                    weight = len(self.predictedsets[self.row2predicted_cluster[row_index]]) / len(
                        self.assessableElemSet)
                    totalF1 += weight * f1_score
                    break

        return totalF1

    def printEvaluation(self, print_flag=True):

        total_f1 = self.TotalElementF1()

        m = {'F1': total_f1}
        if print_flag:
            print("New Metric Info:")
            print("F1(%)")
            print(total_f1 * 100)

        return m


class HierarchyClusterEvaluation:
    def __init__(self, gt_cluster_list, predicted_cluster_list, test_data_num):
        """unduplicated match"""
        self.gt_cluster_list = gt_cluster_list
        self.predicted_cluster_list = predicted_cluster_list
        self.relation_ground_dict = dict()  # ground the predicted relation cluster to gt relation cluster
        self.reverse_relation_ground_dict = dict()
        # to avoid same and easy calculate precision and recall (set operation)
        for index, predicted_cluster in enumerate(predicted_cluster_list):
            self.relation_ground_dict[predicted_cluster.rel_wiki_id] = 'Not grounded' + str(index)
        for index, gt_cluster in enumerate(gt_cluster_list):
            self.reverse_relation_ground_dict[gt_cluster.rel_wiki_id] = 'Not grounded' + str(index)
        # predicted and gt element num are same!
        self.all_element_num = test_data_num

    def match_all_predicted_cluster(self):
        all_match_f1 = np.zeros((len(self.predicted_cluster_list), len(self.gt_cluster_list)))
        for p_i, p_c in enumerate(self.predicted_cluster_list):
            for g_i, g_c in enumerate(self.gt_cluster_list):
                p = self.precision(set(p_c.instances), set(g_c.instances))
                r = self.recall(set(p_c.instances), set(g_c.instances))
                match_f1 = 2 * r * p / (p + r) if p + r > 0 else 0
                all_match_f1[p_i, g_i] = match_f1

        for i in range(len(self.predicted_cluster_list)):
            if np.max(all_match_f1) <= 0:  # all matched
                break
            row_i = np.argmax(all_match_f1) // len(self.gt_cluster_list)
            col_i = np.argmax(all_match_f1) % len(self.gt_cluster_list)
            p_c = self.predicted_cluster_list[row_i]
            g_c = self.gt_cluster_list[col_i]
            # set match
            self.relation_ground_dict[p_c.rel_wiki_id] = g_c.rel_wiki_id
            self.reverse_relation_ground_dict[g_c.rel_wiki_id] = p_c.rel_wiki_id
            # set zeros after matched
            all_match_f1[row_i, :] = -1
            all_match_f1[:, col_i] = -1
        return

    def precision(self, response_a, reference_a):
        if len(response_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(response_a))

    def recall(self, response_a, reference_a):
        if len(reference_a) == 0:
            return 0
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def TotalElementR_P(self):
        totalRecall = 0.0
        totalPrecision = 0.0
        taxonomyRecall = 0.0
        taxonomyPrecision = 0.0

        f1_dict = dict()  # record the cluster F1 score to avoid more calculation.
        reversed_relation_grounded_dict = dict()

        # calculate TP_{sc}
        for predicted_cluster in self.predicted_cluster_list:
            predicted_cluster_instances = set(predicted_cluster.instances)
            predicted_taxonomy_nodes = set(predicted_cluster.sons + predicted_cluster.fathers)
            matched_gt_key = self.relation_ground_dict[predicted_cluster.rel_wiki_id]
            if not matched_gt_key.startswith('Not grounded'):
                for gt_cluster in self.gt_cluster_list:
                    if gt_cluster.rel_wiki_id == matched_gt_key:
                        matched_gt_cluster = gt_cluster
                gt_matched_cluster_instances = set(matched_gt_cluster.instances)
                gt_taxonomy_nodes = set(matched_gt_cluster.sons + matched_gt_cluster.fathers)
            else:
                gt_matched_cluster_instances = set()
                gt_taxonomy_nodes = set()

            # calculate the cluster
            cluster_recall = self.recall(predicted_cluster_instances, gt_matched_cluster_instances)
            cluster_precision = self.precision(predicted_cluster_instances, gt_matched_cluster_instances)
            if cluster_precision + cluster_recall > 0:
                cluster_f1 = 2 * cluster_precision * cluster_recall / (cluster_precision + cluster_recall)
            else:
                cluster_f1 = 0
            # calculate the taxonomy
            if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0 and not matched_gt_key.startswith(
                    'Not grounded'):
                taxonomy_precision = 1
            else:
                taxonomy_precision = self.precision(predicted_taxonomy_nodes, gt_taxonomy_nodes)
                # todo print insert path and link score for caseStudy.
                if matched_gt_key =="P86":
                    print("fathers:", predicted_cluster.fathers)
                    for path in predicted_cluster.insert_paths:
                        print(path)
                # if not matched_gt_key.startswith('Not grounded') and taxonomy_precision > 0:
                #     print(matched_gt_key, " score:", taxonomy_precision)
                #     print("fathers:", predicted_cluster.fathers)
                #     print(predicted_cluster.insert_paths)

            taxonomyPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision
            # combine above.
            # print(1 / len(self.predicted_cluster_list) * taxonomy_precision)

            totalPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision * cluster_f1
            # record information for recall calculation.
            if not matched_gt_key.startswith('Not grounded'):
                f1_dict[matched_gt_key] = cluster_f1
                reversed_relation_grounded_dict[matched_gt_key] = predicted_cluster.rel_wiki_id

        # calculate TR_{sc}

        for gt_cluster in self.gt_cluster_list:
            if gt_cluster.rel_wiki_id in reversed_relation_grounded_dict.keys():  # grounded.
                cluster_f1 = f1_dict[gt_cluster.rel_wiki_id]
                predicted_cluster_id = reversed_relation_grounded_dict[gt_cluster.rel_wiki_id]
                for predicted_cluster in self.predicted_cluster_list:
                    if predicted_cluster.rel_wiki_id == predicted_cluster_id:
                        matched_predicted_cluster = predicted_cluster
                        break
                gt_taxonomy_nodes = set(gt_cluster.sons + gt_cluster.fathers)
                predicted_taxonomy_nodes = set(matched_predicted_cluster.sons + matched_predicted_cluster.fathers)
                if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0:
                    taxonomy_recall = 1
                else:
                    taxonomy_recall = self.recall(predicted_taxonomy_nodes, gt_taxonomy_nodes)
            else:
                cluster_f1 = 0
                taxonomy_recall = 0

            taxonomyRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall
            totalRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall * cluster_f1
        return taxonomyRecall, taxonomyPrecision, totalRecall, totalPrecision

    def MatchF1(self):
        totalF1 = 0.0
        for p_i, p_c in enumerate(self.predicted_cluster_list):
            g_c_id = self.relation_ground_dict[p_c.rel_wiki_id]
            if g_c_id.startswith("Not grounded"):
                continue
            else:
                for g_c in self.gt_cluster_list:
                    if g_c_id == g_c.rel_wiki_id:
                        p = self.precision(set(p_c.instances), set(g_c.instances))
                        r = self.recall(set(p_c.instances), set(g_c.instances))
                        match_f1 = 2 * r * p / (p + r) if p + r > 0 else 0
                        totalF1 += match_f1 * (len(p_c.instances) / self.all_element_num)
        return totalF1

    def printEvaluation(self, print_flag=True):
        self.match_all_predicted_cluster()
        match_f1 = self.MatchF1()
        taxonomy_rec, taxonomy_prec, total_rec, total_prec = self.TotalElementR_P()
        if total_rec == 0 and total_prec == 0:
            total_f1 = 0
        else:
            total_f1 = (2 * total_rec * total_prec) / (total_rec + total_prec)
        if taxonomy_rec == 0 and taxonomy_prec == 0:
            taxonomy_f1 = 0
        else:
            taxonomy_f1 = (2 * taxonomy_rec * taxonomy_prec) / (taxonomy_rec + taxonomy_prec)
        if print_flag:
            print("new metric Info:")
            print("F1(%)")
            print(match_f1 * 100)

            print("taxonomy Info:")
            print("Precision(%); Recall(%); F1(%)")
            print(round(taxonomy_prec * 100, 3), "; ", round(taxonomy_rec * 100, 3), "; ", round(taxonomy_f1 * 100, 3))

            print("Total Info:")
            print("Precision(%); Recall(%); F1(%)")
            print(round(total_prec * 100, 3), "; ", round(total_rec * 100, 3), "; ", round(total_f1 * 100, 3))

        m = {'match_f1': match_f1, 'total_F1': total_f1, 'total_precision': total_prec,
             'total_recall': total_rec,
             'taxonomy_F1': taxonomy_f1, 'taxonomy_precision': taxonomy_prec, 'taxonomy_recall': taxonomy_rec}
        # m = {k: v * 100 for k, v in m.items()}
        return m


# class HierarchyClusterEvaluation:
#     def __init__(self, gt_cluster_list, predicted_cluster_list, test_data_num, duplicate_match=False):
#         self.gt_cluster_list = gt_cluster_list
#         self.predicted_cluster_list = predicted_cluster_list
#         self.groundtruthMatched = dict()
#         for gt_cluster in gt_cluster_list:
#             self.groundtruthMatched[gt_cluster.rel_wiki_id] = False
#         self.relation_ground_dict = dict()  # ground the predicted relation cluster to gt relation cluster
#         self.reverse_relation_ground_dict = dict()
#         # to avoid same and easy calculate precision and recall (set operation)
#         for index, predicted_cluster in enumerate(predicted_cluster_list):
#             self.relation_ground_dict[predicted_cluster.rel_wiki_id] = 'Not grounded' + str(index)
#         for index, gt_cluster in enumerate(gt_cluster_list):
#             self.reverse_relation_ground_dict[gt_cluster.rel_wiki_id] = 'Not grounded' + str(index)
#         # predicted and gt element num are same!
#         self.all_element_num = test_data_num
#         self.duplicate_match = duplicate_match
#
#     def reset_match(self):
#         for gt_cluster in self.gt_cluster_list:
#             self.groundtruthMatched[gt_cluster.rel_wiki_id] = False
#         for index, predicted_rel_id in enumerate(self.predicted_cluster_list):
#             self.relation_ground_dict[predicted_rel_id.rel_wiki_id] = 'Not grounded' + str(index)
#
#     def match_predicted_cluster(self, predicted_cluster):
#         """
#         find the most match cluster in gt hierarchy cluster and fill the mapping related dict.
#         :param predicted_cluster: predicted_cluster
#         :return: matched flag, the matched ground truth cluster key
#         """
#         match_dict = dict()
#         c = set(predicted_cluster.instances)
#         for gt_cluster in self.gt_cluster_list:
#             match_dict[gt_cluster.rel_wiki_id] = 0
#             p = self.precision(c, set(gt_cluster.instances))
#             r = self.recall(c, set(gt_cluster.instances))
#             f1 = 2 * r * p / (p + r) if p + r > 0 else 0
#             match_dict[gt_cluster.rel_wiki_id] = f1
#
#         # sort f1 score
#         sorted_match_items = sorted(list(match_dict.items()), key=lambda x: x[1], reverse=True)
#         for key, f1 in sorted_match_items:
#             if not self.duplicate_match and not self.groundtruthMatched[key]:
#                 self.groundtruthMatched[key] = True
#                 self.relation_ground_dict[predicted_cluster.rel_wiki_id] = key
#                 return True, key
#             else:
#                 self.relation_ground_dict[predicted_cluster.rel_wiki_id] = key
#                 return True, key
#         else:  # not grounded.
#             return False, None
#
#     def match_gt_cluster(self, gt_cluster):
#         """
#         only in duplicated match situation.
#         """
#         match_dict = dict()
#         c = set(gt_cluster.instances)
#         for predicted_cluster in self.predicted_cluster_list:
#             match_dict[predicted_cluster.rel_wiki_id] = 0
#             p = self.precision(c, set(predicted_cluster.instances))
#             r = self.recall(c, set(predicted_cluster.instances))
#             f1 = 2 * r * p / (p + r) if p + r > 0 else 0
#             match_dict[predicted_cluster.rel_wiki_id] = f1
#
#         # sort f1 score
#         sorted_match_items = sorted(list(match_dict.items()), key=lambda x: x[1], reverse=True)
#         for key, f1 in sorted_match_items:
#             self.reverse_relation_ground_dict[gt_cluster.rel_wiki_id] = key
#             return True, key
#         else:  # not grounded.
#             return False, None
#
#     def precision(self, response_a, reference_a):
#         if len(response_a) == 0:
#             return 0
#         return len(response_a.intersection(reference_a)) / float(len(response_a))
#
#     def recall(self, response_a, reference_a):
#         if len(reference_a) == 0:
#             return 0
#         return len(response_a.intersection(reference_a)) / float(len(reference_a))
#
#     def TotalElementR_P(self):
#         totalRecall = 0.0
#         totalPrecision = 0.0
#         taxonomyRecall = 0.0
#         taxonomyPrecision = 0.0
#         # find match and update match info
#         for predicted_cluster in self.predicted_cluster_list:
#             self.match_predicted_cluster(predicted_cluster)
#
#         f1_dict = dict()  # record the cluster F1 score to avoid more calculation.
#         reversed_relation_grounded_dict = dict()
#
#         # calculate TP_{sc}
#         for predicted_cluster in self.predicted_cluster_list:
#             predicted_cluster_instances = set(predicted_cluster.instances)
#             # here do not use set, since they are some not grounded flag.
#             predicted_taxonomy_nodes = set(predicted_cluster.sons + predicted_cluster.fathers)
#             matched_gt_key = self.relation_ground_dict[predicted_cluster.rel_wiki_id]
#             if not matched_gt_key.startswith('Not grounded'):
#                 for gt_cluster in self.gt_cluster_list:
#                     if gt_cluster.rel_wiki_id == matched_gt_key:
#                         matched_gt_cluster = gt_cluster
#                 gt_matched_cluster_instances = set(matched_gt_cluster.instances)
#                 gt_taxonomy_nodes = set(matched_gt_cluster.sons + matched_gt_cluster.fathers)
#             else:
#                 gt_matched_cluster_instances = set()
#                 gt_taxonomy_nodes = set()
#
#             # calculate the cluster
#             cluster_recall = self.recall(predicted_cluster_instances, gt_matched_cluster_instances)
#             cluster_precision = self.precision(predicted_cluster_instances, gt_matched_cluster_instances)
#             if cluster_precision + cluster_recall > 0:
#                 cluster_f1 = 2 * cluster_precision * cluster_recall / (cluster_precision + cluster_recall)
#             else:
#                 cluster_f1 = 0
#             # calculate the taxonomy
#             if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0:
#                 taxonomy_precision = 1
#             else:
#                 taxonomy_precision = self.precision(predicted_taxonomy_nodes, gt_taxonomy_nodes)
#             taxonomyPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision
#             # combine above.
#
#             totalPrecision += 1 / len(self.predicted_cluster_list) * taxonomy_precision * cluster_f1
#             # record information for precision calculation.
#             if not matched_gt_key.startswith('Not grounded'):
#                 f1_dict[matched_gt_key] = cluster_f1
#                 reversed_relation_grounded_dict[matched_gt_key] = predicted_cluster.rel_wiki_id
#
#         # calculate TR_{sc}
#         if not self.duplicate_match:
#             for gt_cluster in self.gt_cluster_list:
#                 if gt_cluster.rel_wiki_id in reversed_relation_grounded_dict.keys():  # grounded.
#                     cluster_f1 = f1_dict[gt_cluster.rel_wiki_id]
#                     predicted_cluster_id = reversed_relation_grounded_dict[gt_cluster.rel_wiki_id]
#                     for predicted_cluster in self.predicted_cluster_list:
#                         if predicted_cluster.rel_wiki_id == predicted_cluster_id:
#                             matched_predicted_cluster = predicted_cluster
#                             break
#                     gt_taxonomy_nodes = set(gt_cluster.sons + gt_cluster.fathers)
#                     predicted_taxonomy_nodes = set(matched_predicted_cluster.sons + matched_predicted_cluster.fathers)
#                     if len(predicted_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0:
#                         taxonomy_recall = 1
#                     else:
#                         taxonomy_recall = self.recall(predicted_taxonomy_nodes, gt_taxonomy_nodes)
#                 else:
#                     cluster_f1 = 0
#                     taxonomy_recall = 0
#
#                 taxonomyRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall
#                 totalRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall * cluster_f1
#         else:
#             for gt_cluster in self.gt_cluster_list:
#                 gt_taxonomy_nodes = set(gt_cluster.sons + gt_cluster.fathers)
#                 gt_instances = set(gt_cluster.instances)
#                 matched_flag, matched_predicted_key = self.match_gt_cluster(gt_cluster)
#                 if matched_flag:
#                     for predicted_cluster in self.predicted_cluster_list:
#                         if predicted_cluster.rel_wiki_id == matched_predicted_key:
#                             matched_predicted_cluster_instances = set(predicted_cluster.instances)
#                             matched_predicted_cluster_taxonomy_nodes = set(
#                                 predicted_cluster.sons + predicted_cluster.fathers)
#                             if len(matched_predicted_cluster_taxonomy_nodes) == 0 and len(gt_taxonomy_nodes) == 0:
#                                 taxonomy_recall = 1
#                             else:
#                                 taxonomy_recall = self.recall(matched_predicted_cluster_taxonomy_nodes,
#                                                               gt_taxonomy_nodes)
#
#                             cluster_recall = self.recall(matched_predicted_cluster_instances, gt_instances)
#                             cluster_precision = self.precision(matched_predicted_cluster_instances, gt_instances)
#                             if cluster_precision + cluster_recall > 0:
#                                 cluster_f1 = 2 * cluster_precision * cluster_recall / (
#                                         cluster_precision + cluster_recall)
#                             else:
#                                 cluster_f1 = 0
#                             break
#                 else:
#                     cluster_f1 = 0
#                     taxonomy_recall = 0
#                 taxonomyRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall
#                 totalRecall += 1 / len(self.gt_cluster_list) * taxonomy_recall * cluster_f1
#         return taxonomyRecall, taxonomyPrecision, totalRecall, totalPrecision
#
#     def NoStructureElementF1(self):
#         # totalRecall = 0.0
#         # totalPrecision = 0.0
#         totalF1 = 0.0
#         for predicted_cluster in self.predicted_cluster_list:
#             matched_flag, matched_gt_key = self.match_predicted_cluster(predicted_cluster)
#             predicted_cluster_instances = set(predicted_cluster.instances)
#             if matched_flag:
#                 for gt_cluster in self.gt_cluster_list:
#                     if gt_cluster.rel_wiki_id == matched_gt_key:
#                         matched_gt_cluster = gt_cluster
#                         break
#                 gt_matched_cluster_instances = set(matched_gt_cluster.instances)
#             else:
#                 gt_matched_cluster_instances = set()  # not matched, compare with the empty set
#             recall = self.recall(predicted_cluster_instances, gt_matched_cluster_instances)
#             precision = self.precision(predicted_cluster_instances, gt_matched_cluster_instances)
#             f1 = (2 * recall * precision) / (recall + precision) if recall + precision > 0 else 0
#             totalF1 += f1 * (len(predicted_cluster_instances) / self.all_element_num)
#
#         return totalF1
#
#     def printEvaluation(self, print_flag=True):
#         new_metric_f1 = self.NoStructureElementF1()
#         self.reset_match()
#         taxonomy_rec, taxonomy_prec, total_rec, total_prec = self.TotalElementR_P()
#         if total_rec == 0 and total_prec == 0:
#             total_f1 = 0
#         else:
#             total_f1 = (2 * total_rec * total_prec) / (total_rec + total_prec)
#         if taxonomy_rec == 0 and taxonomy_prec == 0:
#             taxonomy_f1 = 0
#         else:
#             taxonomy_f1 = (2 * taxonomy_rec * taxonomy_prec) / (taxonomy_rec + taxonomy_prec)
#         if print_flag:
#             print("new metric Info:")
#             print("F1(%)")
#             print(new_metric_f1 * 100)
#
#             print("taxonomy Info:")
#             print("Precision(%); Recall(%); F1(%)")
#             print(round(taxonomy_prec * 100, 3), "; ", round(taxonomy_rec * 100, 3), "; ", round(taxonomy_f1 * 100, 3))
#
#             print("Total Info:")
#             print("Precision(%); Recall(%); F1(%)")
#             print(round(total_prec * 100, 3), "; ", round(total_rec * 100, 3), "; ", round(total_f1 * 100, 3))
#
#         m = {'new_metric_f1': new_metric_f1, 'total_F1': total_f1, 'total_precision': total_prec,
#              'total_recall': total_rec,
#              'taxonomy_F1': taxonomy_f1, 'taxonomy_precision': taxonomy_prec, 'taxonomy_recall': taxonomy_rec}
#         # m = {k: v * 100 for k, v in m.items()}
#         return m


class HierarchyClusterEvaluationTypes:
    def __init__(self, gt_cluster_list, predicted_cluster_list, test_relation_infos_file: str, rel2id_file: str):
        self.gt_cluster_list = gt_cluster_list
        self.predicted_cluster_list = predicted_cluster_list
        self.test_relation_infos = json.load(open(test_relation_infos_file))
        self.rel2id = json.load(open(rel2id_file))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

    def printEvaluation(self):
        for type, rels in self.test_relation_infos.items():
            type_gt_clusters = []
            type_pred_clusters = []
            data_indexs = []
            for gt_cluster in self.gt_cluster_list:
                if gt_cluster.rel_wiki_id in rels:
                    type_gt_clusters.append(copy(gt_cluster))
                    data_indexs += gt_cluster.instances
            for pred_cluster in self.predicted_cluster_list:
                sub_pred_instances = set(pred_cluster.instances).intersection(set(data_indexs))
                if len(sub_pred_instances) == 0:
                    continue
                new_pred_cluster = copy(pred_cluster)
                new_pred_cluster.instances = sub_pred_instances
                type_pred_clusters.append(new_pred_cluster)

            eval = HierarchyClusterEvaluation(type_gt_clusters, type_pred_clusters, len(data_indexs)).printEvaluation(
                False)
            print("type:", type)
            print("metric:", {k: v * 100 for k, v in eval.items()})
        return


if __name__ == '__main__':
    gt = [10, 10, 100, 10, 100, 10, 200, 100]
    pt = [200, 10, 100, 200, 10, 100, 200, 100]
    a = ClusterEvaluationNew(gt, pt)
    print(a.printEvaluation())
    a = ClusterEvaluation(gt, pt)
    print(a.printEvaluation())
