from src.classifiers.nn_classifier_keras import train_speck_distinguisher
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.tree import export_graphviz
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


class All_classifier:


    def __init__(self, args, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth):
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
        self.masks_infos_score = None
        self.masks_infos_rank = None
        if args.retrain_nn_ref:
            self.retrain_classifier_final(args, nn_model_ref)



    def classify_all(self):
        for clf in self.args.classifiers_ours:
            if clf == "NN":
                print("START CLASSIFY NN")
                print()
                self.classifier_nn()
            if clf == "LGBM":
                print("START CLASSIFY LGBM")
                print()
                self.classifier_lgbm()
                if self.args.retrain_with_import_features and self.args.keep_number_most_impactfull >0:
                    self.classifier_lgbm_retrict()
            if clf == "RF":
                print("START CLASSIFY RF")
                print()
                self.classifier_RF()
                if self.args.retrain_with_import_features and self.args.keep_number_most_impactfull >0:
                    self.classifier_RF_retrict()




    def retrain_classifier_final(self, args, nn_model_ref):
        nn_model_ref.epochs = args.num_epch_2
        nn_model_ref.batch_size_2 = args.batch_size_2
        nn_model_ref.net.freeze()
        X_train_proba_feat, X_eval_proba_feat = nn_model_ref.all_intermediaire, nn_model_ref.all_intermediaire_val
        Y_train_proba = self.generator_data.Y_create_proba_train
        Y_eval_proba = self.generator_data.Y_create_proba_val
        print("START RETRAIN LINEAR NN GOHR ")
        print()
        net_retrain, h = train_speck_distinguisher(args, X_train_proba_feat.shape[1], X_train_proba_feat,
                                                   Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                   bs=args.batch_size_2,
                                                   epoch=args.num_epch_2, name_ici="retrain_nn_gohr",
                                                   wdir=self.path_save_model)


    def classifier_nn(self):
        net2, h = train_speck_distinguisher(self.args, len(self.get_masks_gen.masks[0]), self.X_train_proba,
                                            self.Y_train_proba, self.X_eval_proba, self.Y_eval_proba,
                                            bs=self.args.batch_size_our,
                                            epoch=self.args.num_epch_our, name_ici="our_model",
                                            wdir=self.path_save_model)

    def classifier_lgbm_general(self, X_DDTpd, X_eval, features):
        best_params_ = {
            'objective': 'binary',
            'num_leaves': 50,
            'min_data_in_leaf': 10,
            'max_depth': 10,
            'max_bin': 50,
            'learning_rate': 0.01,
            'dart': False,
            'reg_alpha': 0.1,
            'reg_lambda': 0,
            'n_estimators': 1000,
            'bootstrap': True,
            'dart': False
        }
        final_model = lgb.LGBMClassifier(**best_params_, random_state=self.args.seed)
        final_model.fit(X_DDTpd, self.Y_train_proba)
        self.plot_feat_importance(final_model, features,
                                  self.path_save_model + "features_importances_LGBM_nbrefeat_"+str(len(features))+".png")
        y_pred = final_model.predict(X_eval)
        self.save_logs(self.path_save_model + "logs_lgbm_"+str(len(features))+".txt", y_pred, self.Y_eval_proba)
        lgb.create_tree_digraph(final_model).save(directory=self.path_save_model, filename="tree_LGBM_nbrefeat_"+str(len(features))+".dot")
        os.system("dot -Tpng " + self.path_save_model + "tree_LGBM_nbrefeat_"+str(len(features))+".dot > " + self.path_save_model + "tree_LGBM_nbrefeat_"+str(len(features))+".png")
        del X_DDTpd
        self.importances = final_model.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]

        with open(self.path_save_model + "features_impotances_order_nbrefeat_"+str(len(features))+".txt", "w") as file:
            file.write(str(np.array(features)[self.indices]) + str(self.importances[self.indices]))
            file.write("\n")
        if self.masks_infos_score is None:
            self.masks_infos_score = self.importances.copy()
            self.masks_infos_rank = np.array([np.where(self.indices==x)[0][0] for x in range(len(self.importances))])



    def classifier_lgbm(self):
        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.features_name)
        self.classifier_lgbm_general(X_DDTpd, self.X_eval_proba, self.table_of_truth.features_name)



    def classifier_lgbm_retrict(self):
        indices = np.argsort(self.importances)[::-1][:self.args.keep_number_most_impactfull]
        X_DDTpd = pd.DataFrame(data=self.X_train_proba[:, indices],
                               columns=np.array(self.table_of_truth.features_name)[indices])
        self.classifier_lgbm_general(X_DDTpd, self.X_eval_proba[:, indices], np.array(self.table_of_truth.features_name)[indices])


    def classifier_RF_general(self, X_DDTpd, X_eval, features):
        best_params_RF = {'n_estimators': 100,
                          'max_features': 'auto',
                          'max_depth': 100,
                          'min_samples_split': 5,
                          'min_samples_leaf': 2,
                          'bootstrap': True}
        final_model = RandomForestClassifier(**best_params_RF, random_state=self.args.seed)
        final_model.fit(X_DDTpd, self.Y_train_proba)
        self.plot_feat_importance(final_model, features,
                                  self.path_save_model + "features_importances_RF_nbrefeat_"+str(len(features))+".png")
        y_pred = final_model.predict(X_eval)
        self.save_logs(self.path_save_model + "logs_RF_"+str(len(features))+".txt", y_pred, self.Y_eval_proba)

        export_graphviz(final_model.estimators_[5],
                        out_file=self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".dot",
                        feature_names=features, class_names=["Random", "Speck"],
                        rounded=True, proportion=False, precision=2, filled=True)

        os.system("dot -Tpng " + self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".dot > " + self.path_save_model + "tree_RF_nbrefeat_"+str(len(features))+".png")
        del X_DDTpd
        self.importances = final_model.feature_importances_
        indices = np.argsort(self.importances)[::-1]
        with open(self.path_save_model + "features_impotances_order_RF_nbrefeat_"+str(len(features))+".txt", "w") as file:
            file.write(str(np.array(features)[indices]) + str(self.importances[indices]))
            file.write("\n")


    def classifier_RF(self):
        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.features_name)
        self.classifier_RF_general(X_DDTpd, self.X_eval_proba, self.table_of_truth.features_name)

    def classifier_RF_retrict(self):
        indices = np.argsort(self.importances)[::-1][:self.args.keep_number_most_impactfull]
        X_DDTpd = pd.DataFrame(data=self.X_train_proba[:, indices],
                               columns=np.array(self.table_of_truth.features_name)[indices])
        self.classifier_RF_general(X_DDTpd, self.X_eval_proba[:, indices], np.array(self.table_of_truth.features_name)[indices])




    def save_logs(self, path_save_model_txt, y_pred,Y_vf):
        with open(path_save_model_txt, 'w') as f:
            print("ACCURACY")
            f.write("ACCURACY")
            f.write("\n")
            print(accuracy_score(y_pred=y_pred, y_true=Y_vf))
            f.write(str(accuracy_score(y_pred=y_pred, y_true=Y_vf)))
            f.write("\n")
            print("Confusion matrix")
            f.write("Confusion matrix")
            f.write("\n")
            print(confusion_matrix(y_pred=y_pred, y_true=Y_vf))
            f.write(str(confusion_matrix(y_pred=y_pred, y_true=Y_vf)))
            f.write("\n")
            print(confusion_matrix(y_pred=y_pred, y_true=Y_vf, normalize="true"))
            f.write(str(confusion_matrix(y_pred=y_pred, y_true=Y_vf, normalize="true")))
            f.write("\n")
            print()
            print(metrics.classification_report(Y_vf, y_pred, target_names=["random", "speck"], digits=4))
            f.write(str(metrics.classification_report(Y_vf, y_pred, target_names=["random", "speck"], digits=4)))
            f.write("\n")
            print()
            print('Mean Absolute Error:', metrics.mean_absolute_error(Y_vf, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(Y_vf, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_vf, y_pred)))
            f.write('Mean Absolute Error: '+ str(metrics.mean_absolute_error(Y_vf, y_pred)))
            f.write("\n")
            f.write('Mean Squared Error: '+ str(metrics.mean_squared_error(Y_vf, y_pred)))
            f.write("\n")
            f.write('Root Mean Squared Error: '+ str(np.sqrt(metrics.mean_squared_error(Y_vf, y_pred))))
            f.close()

    def plot_feat_importance(self, final_model, feature_name, name):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        colnames = feature_name
        fig, ax = plt.subplots(1, 1, figsize=(75, 75))
        ax.set_title("Feature importances")
        ax.barh(range(len(colnames)), importances[indices[::-1]],
                color="r", align="center")
        ax.set_yticks(range(len(colnames)))
        ax.set_yticklabels(np.array(colnames)[indices][::-1])
        plt.savefig(name)