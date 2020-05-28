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


class Get_masks:


    def __init__(self, args, create_data_g, path_save_model, generator_data, get_masks_gen, nn_model_ref, table_of_truth):
        self.args = args
        self.table_of_truth = table_of_truth
        self.create_data_g = create_data_g
        self.path_save_model = path_save_model
        self.nn_model_ref = nn_model_ref
        self.get_masks_gen = get_masks_gen
        self.generator_data = generator_data
        self.X_train_proba = generator_data.X_proba_train
        self.Y_train_proba = generator_data.Y_create_proba_train
        self.X_eval_proba = generator_data.X_proba_val
        self.Y_eval_proba = generator_data.Y_create_proba_val
        if args.retrain_nn_ref:
            self.retrain_classifier_final()



    def classify_all(self):
        pass



    def retrain_classifier_final(self, args, nn_model_ref):
        nn_model_ref.epochs = args.num_epch_2
        nn_model_ref.batch_size_2 = args.batch_size_2
        nn_model_ref.net.freeze()
        X_train_proba_feat, X_eval_proba_feat = nn_model_ref.all_intermediaire, nn_model_ref.all_intermediaire_val
        Y_train_proba = generator_data.Y_create_proba_train
        Y_eval_proba = generator_data.Y_create_proba_val
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

    def classifier_lgbm(self):
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

        X_DDTpd = pd.DataFrame(data=self.X_train_proba, columns=self.table_of_truth.feature_names)
        final_model = lgb.LGBMClassifier(**best_params_, random_state=self.args.seed)
        final_model.fit(X_DDTpd, self.Y_train_proba)
        self.plot_feat_importance(final_model, self.get_masks_gen.features_name, self.path_save_model + "features_importances_LGBM.png")
        y_pred = final_model.predict(self.X_eval_proba)
        self.save_logs(self.path_save_model + 'logs_lgbm.txt', y_pred, self.Y_eval_proba)
        lgb.create_tree_digraph(final_model).save(directory=self.path_save_model, filename='tree_LGBM.dot')
        os.system("dot -Tpng " + self.path_save_model + "tree_LGBM.dot > " + self.path_save_model + "tree_LGBM.png")
        del X_DDTpd
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        with open(self.path_save_model + "features_impotances_order_1.txt", "w") as file:
            file.write(str(np.array(self.table_of_truth.feature_names)[indices]), str(importances[indices]))
            file.write("\n")

    def classifier_RF(self):
        pass





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