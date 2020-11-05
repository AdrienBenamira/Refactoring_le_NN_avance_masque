from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import accuracy_score
from tqdm import tqdm


import matplotlib.pyplot as plt

class Quality_masks:


    def __init__(self, args, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth, all_clfs):
        self.args = args
        self.table_of_truth = table_of_truth
        self.path_save_model = path_save_model
        self.nn_model_ref = nn_model_ref
        self.get_masks_gen = get_masks_gen
        self.generator_data = generator_data
        self.X_train_proba = generator_data.X_proba_train
        self.Y_train_proba = generator_data.Y_create_proba_train
        self.X_eval_proba = generator_data.X_proba_val
        self.Y_eval_proba = generator_data.Y_create_proba_val
        self.masks = self.get_masks_gen.masks
        self.param_best = {'alpha': 0.3922345684859479,
                      'average': False,
                      'l1_ratio': 0.5605798090010486,
                      'loss': 'hinge',
                      'penalty': 'elasticnet',
                      'tol': 0.01}
        self.columns = ["name_feat", "source", "compression", "accuracy", "hamming", "score_LGBM", "ranked_LGBM",
                        "IQR", "mean", "median", "sum", "min", "max", "independance_y", "nbre_dependance_var"]
        self.index = [x for x in range(len(self.masks[0]))]
        self.masks_reshape = np.array(self.masks).transpose()
        self.df_ = pd.DataFrame(index=self.index, columns=self.columns)
        self.df_["name_feat"] = self.table_of_truth.features_name
        self.df_["source"] = self.get_masks_gen.masks_infos
        self.df_["compression"] = self.table_of_truth.mask_infos_compression
        self.df_["hamming"] = self.table_of_truth.mask_infos_hamming
        self.df_["score_LGBM"] = all_clfs.masks_infos_score
        self.df_["ranked_LGBM"] = all_clfs.masks_infos_rank
        self.res_all_accuracy=[]
        self.res_all_accuracy2 = []
        self.res_all_mean = []
        self.res_all_median = []
        self.res_all_sum = []
        self.res_all_IQR = []
        self.res_all_min = []
        self.res_all_max = []

        self.m1 = []
        self.m2 = []
        self.m12 = []
        self.acc1 = []
        self.acc2 = []
        self.acc = []

        self.labelici = []




    def start_all(self):
        self.df_["independance_y"] = self.independance_feature_label()
        for index, _ in tqdm(enumerate(self.masks_reshape)):
            self.start_one_masks_acc_statistics(index)



        """for index1, _ in tqdm(enumerate(self.masks_reshape)):
            for index2, _ in tqdm(enumerate(self.masks_reshape)):
                if index2 >index1:
                    #self.labelici.append((self.masks_reshape[index1], self.masks_reshape[index2]))
                    #print((self.masks_reshape[index1], self.masks_reshape[index2]))
                    acc = self.start_one_masks_acc_statistics2(index1, index2)
                    acc1, acc2 = self.res_all_accuracy[index1], self.res_all_accuracy[index2]
                    m10, m20 = self.masks_reshape[index1], self.masks_reshape[index2]
                    m1bis = ""
                    for mmmm in m10:
                        m1bis+=str(mmmm)+"_"
                    m1 = m1bis[:-1]
                    m2bis = ""
                    for mmmm in m20:
                        m2bis+=str(mmmm)+"_"
                    m2 = m2bis[:-1]
                    m120 = self.masks_reshape[index1] + self.masks_reshape[index2]
                    m12bis = ""
                    for mmmm in m120:
                        m12bis += str(mmmm) + "_"
                    m12 = m12bis[:-1]
                    self.m1.append(m1)
                    self.m2.append(m2)
                    self.m12.append(m12)
                    self.acc1.append(acc1)
                    self.acc2.append(acc2)
                    self.acc.append(acc)

        self.index2 = [x for x in range(len(self.acc))]
        self.columns2 = ["m1", "m2", "m12", "acc1", "acc2", "acc"]
        self.df_2 = pd.DataFrame(index=self.index2, columns=self.columns2)
        self.df_2["m1"] = np.array(self.m1)
        self.df_2["m2"] = np.array(self.m2)
        self.df_2["m12"] = np.array(self.m12)
        self.df_2["acc1"] = np.array(self.acc1)
        self.df_2["acc2"] = np.array(self.acc2)
        self.df_2["acc"] = np.array(self.acc)
        self.df_2.to_csv(self.path_save_model + "quality_masks2.csv", index=False)"""

        self.df_["accuracy"] = np.array(self.res_all_accuracy)
        self.df_["IQR"] = np.array(self.res_all_IQR)
        self.df_["mean"] = np.array(self.res_all_mean)
        self.df_["median"] = np.array(self.res_all_median)
        self.df_["sum"] = np.array(self.res_all_sum)
        self.df_["min"] = np.array(self.res_all_min)
        self.df_["max"] = np.array(self.res_all_max)
        self.df = pd.DataFrame(self.X_train_proba, columns=self.table_of_truth.features_name)
        self.save_masks()
        if self.args.compute_independance_feature:
            self.independance_feature()
            self.df_["nbre_dependance_var"] = np.sum(self.res2 < self.args.alpha_test, axis=0)
        self.df_.to_csv(self.path_save_model + "quality_masks.csv", index=False)



    def save_masks(self):
        with open(self.path_save_model + "masks_all.txt", "w") as file:
            for i in range(len(self.args.inputs_type)):
                file.write(str(self.masks[i]))
                file.write("\n")



    def start_one_masks_acc_statistics(self, index):
        X_t = self.X_train_proba[:, index].reshape(-1, 1)
        X_DDT_val = self.X_eval_proba[:, index].reshape(-1, 1)
        clf = DecisionTreeClassifier(random_state=self.args.seed)
        clf.fit(X_t, self.Y_train_proba)
        y_pred = clf.predict(X_DDT_val)
        self.res_all_accuracy.append(accuracy_score(y_pred=y_pred, y_true=self.Y_eval_proba))
        self.res_all_mean.append( np.mean(X_t) )
        self.res_all_median.append( np.median(X_t) )
        self.res_all_sum.append( np.sum(X_t, axis=0)[0] / len(X_t) )
        self.res_all_IQR.append( np.quantile(X_t, 0.75) - np.quantile(X_t, 0.25) )
        self.res_all_max.append(np.max(X_t))
        self.res_all_min.append(np.min(X_t))

    def start_one_masks_acc_statistics2(self, index1, index2):
        X_t = self.X_train_proba[:, [index1, index2]].reshape(-1, 2)
        X_DDT_val = self.X_eval_proba[:, [index1, index2]].reshape(-1, 2)
        #X_t2 = self.X_train_proba[:, index2].reshape(-1, 1)
        #X_DDT_val2 = self.X_eval_proba[:, index2].reshape(-1, 1)
        #X_t = np.array([X_t1, X_t2]).reshape(-1, 2)
        #X_DDT_val = np.array([X_DDT_val1, X_DDT_val2]).reshape(-1, 2)
        clf = DecisionTreeClassifier(random_state=self.args.seed)
        clf.fit(X_t, self.Y_train_proba)
        y_pred = clf.predict(X_DDT_val)
        #self.res_all_accuracy2.append(accuracy_score(y_pred=y_pred, y_true=self.Y_eval_proba))
        #print(accuracy_score(y_pred=y_pred, y_true=self.Y_eval_proba))
        #print(self.res_all_accuracy[index1], self.res_all_accuracy[index2])
        return accuracy_score(y_pred=y_pred, y_true=self.Y_eval_proba)



    def independance_feature_label(self):
        X = self.X_train_proba
        y = self.Y_train_proba
        chi_scores = f_classif(X, y)
        p_values = pd.Series(chi_scores[1], index=self.table_of_truth.features_name)
        p_values.sort_values(ascending=False, inplace=True)
        p_values.to_csv(self.path_save_model + "INDEPENACE FEATURES LABELS.csv")
        return chi_scores[1]

    def independance_feature(self):
        res = np.zeros((len(self.table_of_truth.features_name), len(self.table_of_truth.features_name)))
        for i, _ in enumerate(tqdm(self.table_of_truth.features_name)):
            if i < len(self.table_of_truth.features_name) - 1:
                feature_name_ici = str(self.table_of_truth.features_name[i])
                X = self.df.drop(feature_name_ici, axis=1)
                y = self.df[feature_name_ici]
                chi_scores = f_classif(X, y)
                p_values = pd.Series(chi_scores[1], index=X.columns)
                p_values.sort_values(ascending=False, inplace=True)
                for index_index, index_v in enumerate(p_values.index):
                    index_v_new = self.table_of_truth.features_name.index(index_v)
                    res[i, int(index_v_new)] = p_values.values[index_index]
                del X, y
                if len(self.df.columns) > 1:
                    self.df = self.df.drop(feature_name_ici, axis=1)
        self.res2 = res + res.T
        df2 = pd.DataFrame(self.res2, index=self.table_of_truth.features_name, columns=self.table_of_truth.features_name)
        """
        vals = np.around(df2.values, 2)
        colours = plt.cm.RdBu(vals)
        fig = plt.figure(figsize=(100, 100))
        fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
        plt.table(cellText=vals, rowLabels=df2.index, colLabels=df2.columns,
                              colWidths=[0.03] * vals.shape[1], loc='center',
                              cellColours=colours)
        plt.savefig(self.path_save_model + "COMPARASION INTRA FEATURES XI 2.png")
        """
        df2.to_csv(self.path_save_model + "COMPARASION INTRA FEATURES XI 2.csv")






