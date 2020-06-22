import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import os
from src.nn.nn_model_ref import NN_Model_Ref
import matplotlib.pyplot as plt


from src.data_cipher.create_data import Create_data_binary
from src.utils.initialisation_run import init_all_for_run, init_cipher

import pandas as pd
import numpy as np
from src.utils.config import Config
import argparse
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type
import torch.nn.utils.prune as prune
import math


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser

print("TODO: MULTITHREADING + ASSERT PATH EXIST DEPEND ON CONDITION + save DDT + deleate some dataset")

config = Config()
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--logs_tensorboard", default=config.general.logs_tensorboard)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--models_path_load", default=config.general.models_path_load)
parser.add_argument("--cipher", default=config.general.cipher, choices=["speck", "simon", "aes228", "aes224", "simeck", "gimli"])
parser.add_argument("--nombre_round_eval", default=config.general.nombre_round_eval, type=two_args_str_int)
parser.add_argument("--inputs_type", default=config.general.inputs_type, type=transform_input_type)
parser.add_argument("--word_size", default=config.general.word_size, type=two_args_str_int)
parser.add_argument("--alpha", default=config.general.alpha, type=two_args_str_int)
parser.add_argument("--beta", default=config.general.beta, type=two_args_str_int)
parser.add_argument("--type_create_data", default=config.general.type_create_data, choices=["normal", "real_difference"])


parser.add_argument("--retain_model_gohr_ref", default=config.train_nn.retain_model_gohr_ref, type=str2bool)
parser.add_argument("--load_special", default=config.train_nn.load_special, type=str2bool)
parser.add_argument("--finetunning", default=config.train_nn.finetunning, type=str2bool)
parser.add_argument("--model_finetunne", default=config.train_nn.model_finetunne, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--load_nn_path", default=config.train_nn.load_nn_path)
parser.add_argument("--countinuous_learning", default=config.train_nn.countinuous_learning, type=str2bool)
parser.add_argument("--curriculum_learning", default=config.train_nn.curriculum_learning, type=str2bool)
parser.add_argument("--nbre_epoch_per_stage", default=config.train_nn.nbre_epoch_per_stage, type=two_args_str_int)
parser.add_argument("--type_model", default=config.train_nn.type_model, choices=["baseline", "cnn_attention", "multihead", "deepset", "baseline_bin"])
parser.add_argument("--nbre_sample_train", default=config.train_nn.nbre_sample_train, type=two_args_str_int)
parser.add_argument("--nbre_sample_eval", default=config.train_nn.nbre_sample_eval, type=two_args_str_int)
parser.add_argument("--num_epochs", default=config.train_nn.num_epochs, type=two_args_str_int)
parser.add_argument("--batch_size", default=config.train_nn.batch_size, type=two_args_str_int)
parser.add_argument("--loss_type", default=config.train_nn.loss_type, choices=["BCE", "MSE", "SmoothL1Loss", "CrossEntropyLoss", "F1"])
parser.add_argument("--lr_nn", default=config.train_nn.lr_nn, type=two_args_str_float)
parser.add_argument("--weight_decay_nn", default=config.train_nn.weight_decay_nn, type=two_args_str_float)
parser.add_argument("--momentum_nn", default=config.train_nn.momentum_nn, type=two_args_str_float)
parser.add_argument("--base_lr", default=config.train_nn.base_lr, type=two_args_str_float)
parser.add_argument("--max_lr", default=config.train_nn.max_lr, type=two_args_str_float)
parser.add_argument("--demicycle_1", default=config.train_nn.demicycle_1, type=two_args_str_int)
parser.add_argument("--optimizer_type", default=config.train_nn.optimizer_type, choices=["Adam", "AdamW", "SGD"])
parser.add_argument("--scheduler_type", default=config.train_nn.scheduler_type, choices=["CyclicLR", "None"])
parser.add_argument("--numLayers", default=config.train_nn.numLayers, type=two_args_str_int)
parser.add_argument("--out_channel0", default=config.train_nn.out_channel0, type=two_args_str_int)
parser.add_argument("--out_channel1", default=config.train_nn.out_channel1, type=two_args_str_int)
parser.add_argument("--hidden1", default=config.train_nn.hidden1, type=two_args_str_int)
parser.add_argument("--kernel_size0", default=config.train_nn.kernel_size0, type=two_args_str_int)
parser.add_argument("--kernel_size1", default=config.train_nn.kernel_size1, type=two_args_str_int)
parser.add_argument("--num_workers", default=config.train_nn.num_workers, type=two_args_str_int)
parser.add_argument("--clip_grad_norm", default=config.train_nn.clip_grad_norm, type=two_args_str_float)
parser.add_argument("--end_after_training", default=config.train_nn.end_after_training, type=str2bool)



parser.add_argument("--load_masks", default=config.getting_masks.load_masks, type=str2bool)
parser.add_argument("--file_mask", default=config.getting_masks.file_mask)
parser.add_argument("--nbre_max_masks_load", default=config.getting_masks.nbre_max_masks_load, type=two_args_str_int)
parser.add_argument("--nbre_generate_data_train_val", default=config.getting_masks.nbre_generate_data_train_val, type=two_args_str_int)
parser.add_argument("--nbre_necessaire_val_SV", default=config.getting_masks.nbre_necessaire_val_SV, type=two_args_str_int)
parser.add_argument("--nbre_max_batch", default=config.getting_masks.nbre_max_batch, type=two_args_str_int)
parser.add_argument("--liste_segmentation_prediction", default=config.getting_masks.liste_segmentation_prediction)
parser.add_argument("--liste_methode_extraction", default=config.getting_masks.liste_methode_extraction, type=transform_input_type)
parser.add_argument("--liste_methode_selection", default=config.getting_masks.liste_methode_selection, type=transform_input_type)
parser.add_argument("--hamming_weigth", default=config.getting_masks.hamming_weigth, type=str2list)
parser.add_argument("--thr_value", default=config.getting_masks.thr_value, type=str2list)
parser.add_argument("--research_new_masks", default=config.getting_masks.research_new_masks, type=str2bool)
parser.add_argument("--save_fig_plot_feature_before_mask", default=config.getting_masks.save_fig_plot_feature_before_mask, type=str2bool)
parser.add_argument("--end_after_step2", default=config.getting_masks.end_after_step2, type=str2bool)


parser.add_argument("--create_new_data_for_ToT", default=config.make_ToT.create_new_data_for_ToT, type=str2bool)
parser.add_argument("--create_ToT_with_only_sample_from_cipher", default=config.make_ToT.create_ToT_with_only_sample_from_cipher, type=str2bool)
parser.add_argument("--nbre_sample_create_ToT", default=config.make_ToT.nbre_sample_create_ToT, type=two_args_str_int)

parser.add_argument("--create_new_data_for_classifier", default=config.make_data_classifier.create_new_data_for_classifier, type=str2bool)
parser.add_argument("--nbre_sample_train_classifier", default=config.make_data_classifier.nbre_sample_train_classifier, type=two_args_str_int)
parser.add_argument("--nbre_sample_val_classifier", default=config.make_data_classifier.nbre_sample_val_classifier, type=two_args_str_int)

parser.add_argument("--retrain_nn_ref", default=config.compare_classifer.retrain_nn_ref, type=str2bool)
parser.add_argument("--num_epch_2", default=config.compare_classifer.num_epch_2, type=two_args_str_int)
parser.add_argument("--batch_size_2", default=config.compare_classifer.batch_size_2, type=two_args_str_int)
parser.add_argument("--classifiers_ours", default=config.compare_classifer.classifiers_ours, type=str2list)
parser.add_argument("--retrain_with_import_features", default=config.compare_classifer.retrain_with_import_features, type=str2bool)
parser.add_argument("--keep_number_most_impactfull", default=config.compare_classifer.keep_number_most_impactfull, type=two_args_str_int)
parser.add_argument("--num_epch_our", default=config.compare_classifer.num_epch_our, type=two_args_str_int)
parser.add_argument("--batch_size_our", default=config.compare_classifer.batch_size_our, type=two_args_str_int)
parser.add_argument("--alpha_test", default=config.compare_classifer.alpha_test, type=two_args_str_float)
parser.add_argument("--quality_of_masks", default=config.compare_classifer.quality_of_masks, type=str2bool)
parser.add_argument("--end_after_step4", default=config.compare_classifer.end_after_step4, type=str2bool)
parser.add_argument("--eval_nn_ref", default=config.compare_classifer.eval_nn_ref, type=str2bool)
parser.add_argument("--compute_independance_feature", default=config.compare_classifer.compute_independance_feature, type=str2bool)
parser.add_argument("--save_data_proba", default=config.compare_classifer.save_data_proba, type=str2bool)

parser.add_argument("--model_to_prune", default=config.prunning.model_to_prune)
parser.add_argument("--values_prunning", default=config.prunning.values_prunning, type=str2list)
parser.add_argument("--layers_NOT_to_prune", default=config.prunning.layers_NOT_to_prune, type=str2list)
parser.add_argument("--save_model_prune", default=config.prunning.save_model_prune, type=str2bool)
parser.add_argument("--logs_layers", default=config.prunning.logs_layers, type=str2bool)
parser.add_argument("--nbre_sample_eval_prunning", default=config.prunning.nbre_sample_eval_prunning, type=two_args_str_int)
parser.add_argument("--inputs_type_prunning", default=config.general.inputs_type, type=transform_input_type)
parser.add_argument("--a_bit", default=config.train_nn.a_bit, type=two_args_str_int)


args = parser.parse_args()

args.load_special = True
args.finetunning = False
args.logs_tensorboard = args.logs_tensorboard.replace("test", "function_bool")
args.load_nn_path = args.model_to_prune
args.nbre_sample_eval = args.nbre_sample_eval_prunning
args.inputs_type = args.inputs_type_prunning

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
print("---" * 100)
writer, device, rng, path_save_model, path_save_model_train, name_input = init_all_for_run(args)


print("LOAD CIPHER")
print()
cipher = init_cipher(args)
creator_data_binary = Create_data_binary(args, cipher, rng)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("---" * 100)
print("FUNCTION BOOLEAN")

nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref.load_nn()


flag_test = False
acc_retain=[]
nn_model_ref.eval_all(["train", "val"])
X_eval_proba_feat = nn_model_ref.all_intermediaire_val
Y_eval_proba = nn_model_ref.Y_val_nn_binaire
X_train_proba_feat = nn_model_ref.all_intermediaire
Y_train_proba = nn_model_ref.Y_train_nn_binaire


df = pd.read_csv("results/table_of_truth_5round/table_of_truth_0922_final_with_pad.csv", index_col= 0)

if flag_test:
    for index_sample in range(len(nn_model_ref.X_val_nn_binaire[0])):
        X_org = nn_model_ref.X_val_nn_binaire[index_sample]
        X = X_org.reshape(4,16)
        X_desir = nn_model_ref.all_intermediaire_val[index_sample]
        X_transform = np.zeros((64,16))
        for i in range(16):
            if i == 0:
                conditions_im1 = (df['DL[i-1]'] == "PAD") & (df['DV[i-1]'] == "PAD") & (df['V0[i-1]'] == "PAD") & (df['V1[i-1]'] == "PAD")
                conditions_i = (df['DL[i]'] == X[0][i]) & (df['DV[i]'] == X[1][i]) & (df['V0[i]'] ==X[2][i]) & (df['V1[i]'] ==X[3][i])
                conditions_ip1 = (df['DL[i+1]'] ==  str(X[0][i+1]) + ".0") & (df['DV[i+1]'] == str(X[1][i+1]) + ".0") & (df['V0[i+1]'] ==str(X[2][i+1]) + ".0") & (df['V1[i+1]'] ==str(X[3][i+1]) + ".0")
                X_b = df.loc[conditions_im1 & conditions_ip1 & conditions_i].values[0][12:]
            elif i ==15:
                conditions_im1 = (df['DL[i+1]'] == "PAD") & (df['DV[i+1]'] == "PAD") & (df['V0[i+1]'] == "PAD") & (
                            df['V1[i+1]'] == "PAD")
                conditions_i = (df['DL[i]'] == X[0][i]) & (df['DV[i]'] == X[1][i]) & (df['V0[i]'] == X[2][i]) & (
                            df['V1[i]'] == X[3][i])
                conditions_ip1 = (df['DL[i-1]'] == str(X[0][i - 1]) + ".0") & (df['DV[i-1]'] == str(X[1][i - 1]) + ".0") & (
                            df['V0[i-1]'] == str(X[2][i - 1]) + ".0") & (df['V1[i-1]'] == str(X[3][i - 1]) + ".0")
                X_b = df.loc[conditions_im1 & conditions_ip1 & conditions_i].values[0][12:]
                #print(df.loc[conditions_im1 & conditions_ip1 & conditions_i].values[0][:12])
            else:
                conditions_im1 = (df['DL[i-1]'] == str(X[0][i - 1]) + ".0") & (df['DV[i-1]'] == str(X[1][i - 1]) + ".0") & (
                            df['V0[i-1]'] == str(X[2][i - 1]) + ".0") & (df['V1[i-1]'] == str(X[3][i - 1]) + ".0")
                conditions_i = (df['DL[i]'] == X[0][i]) & (df['DV[i]'] == X[1][i]) & (df['V0[i]'] == X[2][i]) & (
                            df['V1[i]'] == X[3][i])
                conditions_ip1 = (df['DL[i+1]'] == str(X[0][i + 1]) + ".0") & (df['DV[i+1]'] == str(X[1][i + 1]) + ".0") & (
                            df['V0[i+1]'] == str(X[2][i + 1]) + ".0") & (df['V1[i+1]'] == str(X[3][i + 1]) + ".0")
                X_b = df.loc[conditions_im1 & conditions_ip1 & conditions_i].values[0][12:]
                #print(df.loc[conditions_im1 & conditions_ip1 & conditions_i].values[0][:12])

            X_transform[:,i] = X_b
        X_f = X_transform.flatten()
        assert np.sum(X_f == X_desir)/1024 ==1



from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
import numpy as np
from pickle import dump

#NN
from tqdm import tqdm

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr);
    return (res);


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True);
    return (res);

def make_classifier(input_size=84, d1=1024, d2=512, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(input_size,));
    dense1 = Dense(d1)(inp);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2)(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation=final_activation)(dense2);
    model = Model(inputs=inp, outputs=out);
    return (model);

def train_speck_distinguisher(n_feat, X, Y, X_eval, Y_eval, epoch, bs, name_ici="", wdir= "./"):
    # create the network
    net = make_classifier(input_size=n_feat);
    net.compile(optimizer='adam', loss='mse', metrics=['acc']);
    # generate training and validation data
    # set up model checkpoint
    check = make_checkpoint(wdir + 'NN_classifier' + str(6) + "_"+ name_ici + '.h5');
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));
    # train and evaluate
    h = net.fit(X, Y, epochs=epoch, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    np.save(wdir + 'h_acc_' + str(np.max(h.history['val_acc'])) + "_"+ name_ici +  '.npy', h.history['val_acc']);
    np.save(wdir + 'h_loss' + str(6) + "_"+ name_ici + '.npy', h.history['val_loss']);
    dump(h.history, open(wdir + 'hist' + str(6) + "_"+ name_ici +  '.p', 'wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    net3 = make_classifier(input_size=n_feat);
    net3.load_weights(wdir + 'NN_classifier' + str(6) + "_"+ name_ici +  '.h5')
    return (net3, h);


net_retrain, h = train_speck_distinguisher(X_train_proba_feat.shape[1], X_train_proba_feat,
                                                   Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                   bs=5000,
                                                   epoch=20, name_ici="test")