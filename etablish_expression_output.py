import itertools
import sys
import warnings
import random
import sklearn
import sklearn.neural_network
from src.nn.models.Linear_binarized import Linear_bin
from src.nn.models.Model_AE import AE_binarize
from src.nn.nn_model_ref_v2 import NN_Model_Ref_v2
from alibi.explainers import CEM
from sympy import *
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import os
from src.nn.nn_model_ref import NN_Model_Ref
from sympy.logic import SOPform, POSform
from sympy import symbols
from alibi.explainers import AnchorTabular
from src.data_cipher.create_data import Create_data_binary
from src.utils.initialisation_run import init_all_for_run, init_cipher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from src.utils.config import Config
import argparse
from src.utils.utils import str2bool, two_args_str_int, two_args_str_float, str2list, transform_input_type
import torch.nn.utils.prune as prune
import math

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
import numpy as np
from pickle import dump

#NN
from tqdm import tqdm

import matplotlib.pyplot as plt

from matplotlib import colors

def get_final_expression_0_1(df_final):
    df_final_1 = df_final[df_final['Output'] == 1].values
    conditions_or_1 = []
    for conditions_or in df_final_1:
        str_1 = ""
        already_seen = []
        for el1 in conditions_or[1:]:
            if el1 is not None:
                if el1 not in already_seen:
                    already_seen.append(el1)
                    str_1 += el1 + " & "
        element_final_1 = str_1[:-2]
        if element_final_1 not in conditions_or_1:
            conditions_or_1.append(element_final_1)
    df_final_0 = df_final[df_final['Output'] == 0].values
    conditions_or_0 = []
    for conditions_or in df_final_0:
        str_0 = ""
        already_seen = []
        for el0 in conditions_or[1:]:
            if el0 is not None:
                if el0 not in already_seen:
                    already_seen.append(el0)
                    str_0 += el0 + " & "
        conditions_or_0.append(str_0[:-2])
        element_final_0 = str_0[:-2]
        if element_final_0 not in conditions_or_0:
            conditions_or_0.append(element_final_0)
    return conditions_or_1, conditions_or_0


def get_expression_filter(df2, df2_name):
    output_name = ["Filter_" + str(i) for i in range(args.out_channel0)]
    input_name = ["DL[i-1]", "V0[i-1]", "V1[i-1]", "DL[i]", "V0[i]", "V1[i]", "DL[i+1]", "V0[i+1]", "V1[i+1]"]
    input_name_all = ["DL[i-1]", "DV[i-1]", "V0[i-1]", "V1[i-1]", "DL[i]", "DV[i]", "V0[i]", "V1[i]", "DL[i+1]",
                      "DV[i+1]", "V0[i+1]", "V1[i+1]"]

    df2.columns = output_name
    df2_name.columns = input_name_all
    df_m = pd.concat([df2_name, df2], axis=1)
    df_m.to_csv(path_save_model + "table_of_truth_final_with_pad.csv")
    df_m2 = df_m.drop(df_m.index[df_m["DL[i-1]"] == "PAD"])
    df_m_f = df_m2.drop(df_m2.index[df_m2["DL[i+1]"] == "PAD"])
    # print (df_m_f)
    df_m_f = df_m_f.reset_index()
    print(df_m_f.head(5))
    df_m_f.to_csv(path_save_model + "table_of_truth_final_without_pad.csv")

    dictionnaire_res_fin_expression = {}
    dictionnaire_res_fin_expression_POS = {}
    dictionnaire_perfiler = {}
    dictionnaire_perfiler_POS = {}
    doublon = []
    expPOS_tot = []
    cpteur = 0
    dictionnaire_feature_name = {}

    for index_f in range(args.out_channel0):
        # for index_f in range(3):
        offset_feat = 15 * index_f
        print(output_name[index_f])
        # if "F"+str(index_f) in list(dico_important.keys()):
        index_intere = df_m_f.index[df_m_f[output_name[index_f]] == 1].tolist()
        print()
        if len(index_intere) == 0:
            print("Empty")
            for time in range(1, 15):
                dictionnaire_feature_name["Feature_" + str(index_f + time + offset_feat)] = 0
        else:
            dictionnaire_res_fin_expression[output_name[index_f]] = []
            dictionnaire_res_fin_expression_POS[output_name[index_f]] = []
            condtion_filter = []
            for col in input_name:
                s = df_m_f[col].values
                my_dict = {"0": 0, "1": 1, 0: 0, 1: 1}
                s2 = np.array([my_dict[zi] for zi in s])
                condtion_filter.append(s2[index_intere])
            condtion_filter2 = np.array(condtion_filter).transpose()
            condtion_filter3 = [x.tolist() for x in condtion_filter2]
            assert len(condtion_filter3) == len(index_intere)
            assert len(condtion_filter3[0]) == 9
            symbols_str = ""
            for input_name_ici in input_name:
                symbols_str += input_name_ici + ", "
            w1, x1, y1, w2, x2, y2, w3, x3, y3 = symbols(symbols_str[:-2])
            minterms = condtion_filter3
            exp = SOPform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
            expPOS = POSform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
            if exp in doublon:
                print(exp, "DOUBLON")
                for time in range(1, 15):
                    dictionnaire_feature_name["Feature_" + str(index_f + time + offset_feat)] = str(expPOS).replace("i",
                                                                                                                    str(
                                                                                                                        time))
            elif str(exp) == 'True':
                print(exp, "True")
                for time in range(1, 15):
                    dictionnaire_feature_name["Feature_" + str(index_f + time + offset_feat)] = 1
            else:
                print(exp)
                for time in range(1, 15):
                    dictionnaire_feature_name["Feature_" + str(index_f + time + offset_feat)] = str(expPOS).replace("i",
                                                                                                                    str(
                                                                                                                        time))
                doublon.append(exp)
                dictionnaire_res_fin_expression[output_name[index_f]].append(exp)
                expV2 = str(exp).split(" | ")
                dictionnaire_perfiler[output_name[index_f]] = [str(exp)] + [x.replace("(", "").replace(")", "") for x in
                                                                            expV2]
                print()

                print(expPOS)
                expPOS_tot.append(str(expPOS))
                dictionnaire_res_fin_expression_POS[output_name[index_f]].append(expPOS)
                expV2POS = str(expPOS).split(" & ")
                dictionnaire_perfiler_POS[output_name[index_f]] = [str(expV2POS)] + [x.replace("(", "").replace(")", "")
                                                                                     for x in expV2POS]
                # dictionnaire_res_fin_expression["Filter " + str(index_f)].append(exp)
        print()

    del df_m_f, df_m

    df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler, orient='index').T
    row = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
    df_filtre.to_csv(path_save_model + "dictionnaire_perfiler.csv")
    df_row = pd.DataFrame(row)
    df_row.to_csv(path_save_model + "clause_unique.csv")
    df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression, orient='index').T
    df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter.csv")

    df_filtre = pd.DataFrame.from_dict(dictionnaire_perfiler_POS, orient='index').T
    row3 = pd.unique(df_filtre[[index_f for index_f in df_filtre.columns]].values.ravel('K'))
    df_filtre.to_csv(path_save_model + "dictionnaire_perfiler_POS.csv")
    df_row = pd.DataFrame(row3)
    df_row.to_csv(path_save_model + "clause_unique_POS.csv")
    df_expression_bool_m = pd.DataFrame.from_dict(dictionnaire_res_fin_expression_POS, orient='index').T
    df_expression_bool_m.to_csv(path_save_model + "expression_bool_per_filter_POS.csv")

    df_expression_bool = pd.DataFrame.from_dict(dico_important, orient='index').T
    df_expression_bool.to_csv(path_save_model + "time_important_per_filter.csv")

    return dictionnaire_feature_name,

def get_truth_table_embedding(nn_model_ref, dimension_embedding = 16, bz = 500):
    arr2 = generate_binary(dimension_embedding)
    l = []
    for x in arr2:
        l += [np.array([int(d) for d in x])]
    l2 = np.array(l)
    # l3 = l2.reshape(-1, nbre_input, nbre_temps_chaque_input)
    # l4 = np.transpose(l3, axes=(0,2,1))
    dico_tt_embeding_output = {}
    dico_tt_embeding_output_name = {}
    dico_tt_embeding_feature = {}
    dico_tt_embeding_feature_name = {}
    end_ind_bz = l2.shape[0] // bz + 1
    for index_end_bz in range(end_ind_bz):
        input_array_embedding = l2[index_end_bz * bz:(index_end_bz + 1) * bz]
        x_input_f2 = torch.Tensor(input_array_embedding)
        outputs = torch.sigmoid(nn_model_ref.net.fc3(x_input_f2.to(device)))
        preds = (outputs.squeeze(1) > nn_model_ref.t.to(device)).int().cpu().detach().numpy() * 1
        outputs_feature = nn_model_ref.net.decoder(x_input_f2.to(device))
        preds_feat = (outputs_feature.squeeze(1) > nn_model_ref.t.to(device)).int().cpu().detach().numpy() * 1
        for index_input in range(len(input_array_embedding)):
            input_name2 = input_array_embedding[index_input]
            input_name3 = '_'.join(map(str, input_name2))
            dico_tt_embeding_output[input_name3] = preds[index_input]
            dico_tt_embeding_output_name[input_name3] = input_name2
            preds_feat_str = []
            for index_feat, value_feat in enumerate(preds_feat[index_input]):
                if value_feat:
                    preds_feat_str.append("Feature_"+str(index_feat))
            dico_tt_embeding_feature[input_name3] = preds_feat_str
            dico_tt_embeding_feature_name[input_name3] = input_name2
    del l2, l, x_input_f2
    return dico_tt_embeding_output, dico_tt_embeding_output_name, dico_tt_embeding_feature, dico_tt_embeding_feature_name

def get_truth_table_input_feature(nn_model_ref):
    nbre_input = 3
    nbre_temps_chaque_input = 3
    arr2 = generate_binary(nbre_input * nbre_temps_chaque_input)
    l = []
    for x in arr2:
        l += [np.array([int(d) for d in x])]
    l2 = np.array(l)
    l3 = l2.reshape(-1, nbre_input, nbre_temps_chaque_input)
    l4 = np.transpose(l3, axes=(0, 2, 1))
    V0 = l4[:, 1, :]
    V1 = l4[:, 2, :]
    Dv = V0 ^ V1
    x_input_f = np.insert(l4, 1, Dv, 1)
    l5 = x_input_f[:, :, 1:]
    rest2 = np.zeros((512, 4, 7))
    rest = np.zeros((512, 4, 6))
    # debut
    rest[:, :, :2] = l5
    # fin
    rest2[:, :, 5:] = l5
    x_input_f1b = np.append(rest, x_input_f, axis=2)
    x_input_f2 = np.append(x_input_f1b, rest2, axis=2)
    x_input_f2 = torch.Tensor(x_input_f2)
    df_dico_name_tot = {}
    df_dico_second_tot = {}
    nn_model_ref.net(x_input_f2.to(device))
    for index in range(nn_model_ref.net.x_input.shape[0]):
        res = []
        for index_x, x in enumerate(nn_model_ref.net.classify[index]):
            res.append(x.int().detach().cpu().numpy())
        res2 = np.array(res).transpose()
        df_dico_second_tot, df_dico_name_tot = incremente_dico(nn_model_ref, index, df_dico_second_tot, res2,
                                                               df_dico_name_tot)
    return df_dico_second_tot, df_dico_name_tot


def prunning_model(nn_model_ref, global_sparsity = 0.95, flag2 = True, phases = ["val"]):
    parameters_to_prune = []
    for name, module in nn_model_ref.net.named_modules():
        if len(name):
            if name not in ["layers_batch", "layers_conv"]:
                flag = True
                for layer_forbidden in args.layers_NOT_to_prune:
                    if layer_forbidden in name:
                        flag = False
                if flag:
                    parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=global_sparsity,
    )
    tot_sparsity = 0
    tot_weight = 0
    for name, module in nn_model_ref.net.named_modules():
        if len(name):
            if name not in ["layers_batch", "layers_conv"]:
                flag = True
                for layer_forbidden in args.layers_NOT_to_prune:
                    if layer_forbidden in name:
                        flag = False
                if flag:
                    tot_sparsity += 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
                    tot_weight += float(module.weight.nelement()) - float(torch.sum(module.weight == 0))
                    if args.logs_layers:
                        print(
                        "Sparsity in {}.weight: {:.2f}%".format(str(name),
                            100. * float(torch.sum(module.weight == 0))
                            / float(module.weight.nelement())
                            )
                        )
    if flag2:
        nn_model_ref.eval_all(phases)
        print(nn_model_ref.net.fc1.weight_mask.detach().cpu().int().numpy()[0].tolist())
        print(np.sum(nn_model_ref.net.fc1.weight_mask.detach().cpu().int().numpy()[0]))

        print(nn_model_ref.net.fc2.weight_mask.detach().cpu().int().numpy()[0].tolist())
        print(np.sum(nn_model_ref.net.fc2.weight_mask.detach().cpu().int().numpy()[0]))

    print(ok)
    #return nn_model_ref








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


def generate_binary(n):

  # 2^(n-1)  2^n - 1 inclusive
  bin_arr = range(0, int(math.pow(2,n)))
  bin_arr = [bin(i)[2:] for i in bin_arr]

  # Prepending 0's to binary strings
  max_len = len(max(bin_arr, key=len))
  bin_arr = [i.zfill(max_len) for i in bin_arr]

  return bin_arr

def incremente_dico(nn_model_ref, index_sample, df_dico_second_tot, res2, df_dico_name_tot):
    for index_interet in range(5):
        input_name = nn_model_ref.net.x_input[index_sample][:, 4+index_interet:7+index_interet].int().detach().cpu().numpy().transpose()
        input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
        input_name3 = '_'.join(map(str, input_name2))
        if input_name3 in list(df_dico_second_tot.keys()):
            for k in range(len(res2[5+index_interet])):
                assert (res2[5+index_interet][k] == df_dico_second_tot[input_name3][k])
        else:
            df_dico_second_tot[input_name3] = res2[5+index_interet]
            df_dico_name_tot[input_name3] = input_name2
    input_name = nn_model_ref.net.x_input[index_sample][:, :2].int().detach().cpu().numpy().transpose()
    input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
    input_name3 = '_'.join(map(str, input_name2))
    input_name4 = "PAD_PAD_PAD_PAD_" + input_name3
    input_name6 = np.append(np.array(["PAD", "PAD", "PAD", "PAD"]), input_name2, axis=0)
    if input_name4 in list(df_dico_second_tot.keys()):
        for k in range(len(res2[0])):
            assert (res2[0][k] == df_dico_second_tot[input_name4][k])
    else:
        df_dico_second_tot[input_name4] = res2[0]
        df_dico_name_tot[input_name4] = input_name6
    input_name = nn_model_ref.net.x_input[index_sample][:, -2:].int().detach().cpu().numpy().transpose()
    input_name2 = np.hstack([input_name[i, :] for i in range(input_name.shape[0])])
    input_name3 = '_'.join(map(str, input_name2))
    input_name4 =  input_name3 + "_PAD_PAD_PAD_PAD"
    input_name6 = np.append(input_name2, np.array(["PAD", "PAD", "PAD", "PAD"]), axis=0)
    if input_name4 in list(df_dico_second_tot.keys()):
        for k in range(len(res2[0])):
            assert (res2[-1][k] == df_dico_second_tot[input_name4][k])
    else:
        df_dico_second_tot[input_name4] = res2[-1]
        df_dico_name_tot[input_name4] = input_name6
    return df_dico_second_tot, df_dico_name_tot



def DPLL(exp, M):
    print(exp)
    if len(exp) ==0:
        return (M)
    if len(exp) >0:
        #int_random =random.randint(0,len(exp)-1)
        clause = exp[0].replace(" ", "")
        if '|' not in clause:
            M = increment(M, clause)
            exp = exp[1:].copy()
            return DPLL(exp, M)
        else:
            vars = clause.split('|')
            #int_random = random.randint(0, len(vars) - 1)
            #for var in vars:
            var = vars[0]
            M = increment(M, var.replace(" ", "").replace(")", "").replace("(", ""))
            #exp = exp[1:].copy()
            exp = nettoyer(exp, var.replace(" ", "").replace(")", "").replace("(", "")).copy()
            return DPLL(exp, M)

def increment(M, el_c):
    if "V0" in el_c:
        if "~" in el_c:
            F = 4
        else:
            F = 1
    elif "V1" in el_c:
        if "~" in el_c:
            F = 5
        else:
            F = 2
    elif "DL" in el_c:
        if "~" in el_c:
            F = 3
        else:
            F = 0
    if "[i]" in el_c:
        offset = 1
    elif "[i+1]" in el_c:
        offset = 2
    elif "[i-1]" in el_c:
        offset = 0
    M[F][offset] = 1
    return M

def nettoyer(exp, var):
    exp2 = []
    flag_neg = "~" in var
    if not flag_neg:
        var_neg = "~" + var
    for var2 in exp:
        if not flag_neg:
            if var not in var2 or var_neg in var2:
                exp2.append(var2)
            #if var_neg in var2:
            #    exp2.append(var2.replace(var_neg, "").replace(" ", "").replace(")", "").replace("(", "").replace("|", ""))
            else:
                var3 = var2 + " | "
                var4 = " | " + var2
                var5 = var.replace(var3, "").replace(var4, "")
                if "|" in var:
                    exp2.append(var5)
                else:
                    exp2.append(var5.replace("(", "").replace(")", "").replace(" ", ""))
        else:
            if var not in var2:
                exp2.append(var2)
            else:

                var3 = var + "|"
                var4 = "|" + var
                var5 = var2.replace(" ", "").replace(var3, "").replace(var4, "")


                if "|" in var5:
                    exp2.append(var5.replace(" ", ""))
                else:
                    exp2.append(var5.replace("(", "").replace(")", "").replace(" ", ""))



    exp2.sort(key=lambda x: len(x.split("|")), reverse=False)
    return exp2

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
parser.add_argument("--hidden2", default=config.train_nn.hidden1, type=two_args_str_int)
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
args.logs_tensorboard = args.logs_tensorboard.replace("test", "output_table_of_truth")

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
print("TABLE OF TRUTH")





def get_all_masks_from_1clausefilter_allclauseinput(dico_clause, nbre_clause_input, str_filter, mask_filtre, impossibillty):
    all_element_str_filter = str_filter.split("&")
    dico_element = {}
    for index_f, fl in enumerate(all_element_str_filter):
        fl_clean = fl.replace("(", "").replace(")", "").replace(" ", "")
        index_fl_ici = fl_clean.split("[")[-1].split("]")[0]
        flag_inv = '~' in fl_clean
        dico_element[index_f] = [fl_clean, index_fl_ici, flag_inv]
    nbre_element_filter = len(all_element_str_filter)
    lst_all_possible0 = list(itertools.combinations(range(len(all_element_str_filter)), 2))


    dico_all_possible = {}
    for (index_el1, index_el2) in lst_all_possible0:
        flag1 = dico_element[index_el1][2]
        flag2 = dico_element[index_el2][2]
        flag3 = False
        #if flag1 and flag2:
        #    flag3 = True
        if (not flag1) and (not flag2):
            flag3 = True
        d1 = dico_element[index_el1][1]
        d2 = dico_element[index_el2][1]
        if len(d1) == 1:
            d1_int = 0
        else:
            if "-" in d1:
                d1_int = -1 * int(d1.replace("l-", "").replace(" ", ""))
            else:
                d1_int = int(d1.replace("l+", "").replace(" ", ""))
        if len(d2) == 1:
            d2_int = 0
        else:
            if "-" in d2:
                d2_int = -1 * int(d2.replace("l-", "").replace(" ", ""))
            else:
                d2_int = int(d2.replace("l+", "").replace(" ", ""))
        delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))
        if delta>2:
            flag3=False
        dico_all_possible[(index_el1, index_el2)] = [delta, flag3]



    lst_all_possible = list(itertools.product(range(nbre_clause_input), repeat=nbre_element_filter))

    lst_all_possible_f = lst_all_possible.copy()



    #filtre possibilites:
    #print(dico_all_possible)
    #print(impossibillty)


    #print(len(dico_all_possible))

    for pos in list(dico_all_possible.keys()):
        if dico_all_possible[pos][1]:
            delta = dico_all_possible[pos][0]
            for pos2 in list(impossibillty.keys()):
                flag_impossible = delta in impossibillty[pos2]['oppose:non']
                #print("il faut suppremer de possibilites tous les proposetions avec ", pos2, "en posiotion", pos)
                if flag_impossible:
                    lst_all_possible_f = [tuplet for tuplet in lst_all_possible_f if ((tuplet[pos[0]] != pos2[0]) and (tuplet[pos[1]] != pos2[1]))]

    #print(len(lst_all_possible_f))


    for possible_index, possible in enumerate(lst_all_possible_f):
        mask = [""]
        for index_possition, value_position in enumerate(possible):
            el = dico_element[index_possition]
            cl = dico_clause[value_position]
            mask2 = []
            if el[-1]:
                for m in mask:
                    for ccc in cl[1].split(" | "):
                        m2 = m
                        m2 += ccc + " & "
                        mask2.append(m2)

            else:
                for m in mask:
                    m += cl[0] + " & "
                mask2.append(m)
            mask = mask2
            for m in range(len(mask)):
                mask[m] = mask[m].replace("i", el[1])
        for m in range(len(mask)):
            mask[m] = mask[m][:-3]
        for i in range(9):
            for m in range(len(mask)):
                mask[m] = mask[m].replace(str(i)+"+1", str(i+1)).replace(str(i)+"-1", str(i-1)).replace("+0", "")


        #sanity check
        for m in range(len(mask)):
            mask_el = np.array(mask[m].split(" & "))
            mask_unique_el = np.unique(mask_el)
            mask_unique_el_pos = [el for el in mask_unique_el if "~" not in el]
            mask_unique_el_neg = [el.replace("~", "") for el in mask_unique_el if "~" in el]
            intersection = np.intersect1d(mask_unique_el_pos, mask_unique_el_neg)
            if (len(intersection)==0):
                mask_filtre.append(mask[m])






    return mask_filtre


exp_filter = pd.read_csv("/home/adriben/PycharmProjects/Refactoring_le_NN_avance_masque/results/table_of_truth_v2/speck/5/DL_DV_V0_V1/2020_07_27_12_02_34_340448/expression_bool_per_filter.csv")
exp_filter_2eme_couche = pd.read_csv("/home/adriben/PycharmProjects/Refactoring_le_NN_avance_masque/results/table_of_truth_v2/speck/5/DL_DV_V0_V1/2020_07_27_12_02_34_340448/expression_bool_per_filter_2emecouche.csv")


nn_model_ref = NN_Model_Ref(args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train)
nn_model_ref.load_nn()


flag2 = True
acc_retain=[]
global_sparsity = 0.2
parameters_to_prune = []
for name, module in nn_model_ref.net.named_modules():
    if len(name):
        if name not in ["layers_batch", "layers_conv"]:
            flag = True
            for layer_forbidden in args.layers_NOT_to_prune:
                if layer_forbidden in name:
                    flag = False
            if flag:
                parameters_to_prune.append((module, 'weight'))
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=global_sparsity,
)
tot_sparsity = 0
tot_weight = 0
for name, module in nn_model_ref.net.named_modules():
    if len(name):
        if name not in ["layers_batch", "layers_conv"]:
            flag = True
            for layer_forbidden in args.layers_NOT_to_prune:
                if layer_forbidden in name:
                    flag = False
            if flag:
                tot_sparsity += 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
                tot_weight += float(module.weight.nelement()) - float(torch.sum(module.weight == 0))

                if args.logs_layers:
                    print(
                    "Sparsity in {}.weight: {:.2f}%".format(str(name),
                        100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                        )
                    )
if flag2:
    nn_model_ref.eval_all(["val"])



all_masks = {}


def get_possible(all_clause_str_input):
    #print(all_clause_str_input)
    dico_allclause = {}
    for c in all_clause_str_input:
        dico_allclause[c] = {}
        all_el = c.replace("(","").replace(")","").replace(" ","").split("&")
        elemen_pos=[]
        elemen_neg = []
        position_elemen_pos = {}
        position_elemen_neg = {}
        for el in all_el:
            index_fl_ici = el.split("[")[-1].split("]")[0]
            value_fl_ici = el.split("[")[0]
            if '~' in el:
                elemen_neg.append(value_fl_ici.replace("~", ""))
                if value_fl_ici.replace("~", "") in list(position_elemen_neg.keys()):
                    position_elemen_neg[value_fl_ici.replace("~", "")].append(index_fl_ici)
                else:
                    position_elemen_neg[value_fl_ici.replace("~", "")] = [index_fl_ici]
            else:
                elemen_pos.append(value_fl_ici)
                if value_fl_ici in list(position_elemen_pos.keys()):
                    position_elemen_pos[value_fl_ici].append(index_fl_ici)
                else:
                    position_elemen_pos[value_fl_ici] = [index_fl_ici]
        dico_allclause[c]["elemen_pos"] = elemen_pos
        dico_allclause[c]["elemen_neg"] = elemen_neg
        dico_allclause[c]["position_elemen_pos"] = position_elemen_pos
        dico_allclause[c]["position_elemen_neg"] = position_elemen_neg

    impossibility = {}
    for i_c1, c1 in enumerate(all_clause_str_input):
        for i_c2, c2 in enumerate(all_clause_str_input):
            if i_c2> i_c1:
                impossibility[(i_c1, i_c2)] = {}
                elemen_pos_c1 = dico_allclause[c1]["elemen_pos"]
                elemen_neg_c1 = dico_allclause[c1]["elemen_neg"]
                elemen_pos_c2 = dico_allclause[c2]["elemen_pos"]
                elemen_neg_c2 = dico_allclause[c2]["elemen_neg"]

                # cas meme signe
                impossibility[(i_c1, i_c2)]["oppose:non"] = []
                intersect = np.intersect1d(elemen_pos_c1, elemen_neg_c2)
                if len(intersect)>0:
                    for inter_el in intersect:
                        position_elemen_pos_c1 = dico_allclause[c1]["position_elemen_pos"][inter_el]
                        position_elemen_neg_c2 = dico_allclause[c2]["position_elemen_neg"][inter_el]
                        for d1 in position_elemen_pos_c1:
                            if len(d1)==1:
                                d1_int = 0
                            else:
                                if "-" in d1:
                                    d1_int = -1*int(d1.replace("i-","").replace(" ",""))
                                else:
                                    d1_int =  int(d1.replace("i+", "").replace(" ", ""))
                            for d2 in position_elemen_neg_c2:
                                if len(d2) == 1:
                                    d2_int = 0
                                else:
                                    if "-" in d2:
                                        d2_int = -1 * int(d2.replace("i-", "").replace(" ", ""))
                                    else:
                                        d2_int = int(d2.replace("i+", "").replace(" ", ""))
                                delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))
                                if delta not in impossibility[(i_c1, i_c2)]["oppose:non"]:
                                    if delta>0:
                                        impossibility[(i_c1, i_c2)]["oppose:non"].append(delta)

                """intersect = np.intersect1d(elemen_neg_c1, elemen_pos_c2)
                if len(intersect) > 0:
                    for inter_el in intersect:
                        position_elemen_neg_c1 = dico_allclause[c1]["position_elemen_neg"][inter_el]
                        position_elemen_pos_c2 = dico_allclause[c2]["position_elemen_pos"][inter_el]
                        for d1 in position_elemen_neg_c1:
                            if len(d1) == 1:
                                d1_int = 0
                            else:
                                if "-" in d1:
                                    d1_int = -1 * int(d1.replace("i-", "").replace(" ", ""))
                                else:
                                    d1_int = int(d1.replace("i+", "").replace(" ", ""))
                            for d2 in position_elemen_pos_c2:
                                if len(d2) == 1:
                                    d2_int = 0
                                else:
                                    if "-" in d2:
                                        d2_int = -1 * int(d2.replace("i-", "").replace(" ", ""))
                                    else:
                                        d2_int = int(d2.replace("i+", "").replace(" ", ""))
                                delta = max(abs(d2_int) - abs(d1_int), abs(d1_int) - abs(d2_int))

                                if delta not in impossibility[(i_c1, i_c2)]["oppose:non"]:
                                    if delta>0:
                                        impossibility[(i_c1, i_c2)]["oppose:non"].append(delta)"""



    return impossibility







for num_filter, filter in enumerate(exp_filter_2eme_couche.columns):
    if "Filter" in filter:
        #if int(filter.split("_")[1]) not in [3,5,6,7]:
        print()
        #all_masks[filter] = []
        M = []
        index_f = filter.split('_')[1]
        filter2 = "Filter "+str(index_f)
        str_input = exp_filter[filter2].values[0]
        all_clause_str_input = str_input.split("|")


        impossibillty = get_possible(all_clause_str_input)

        dico_clause = {}
        for index_c, cl in enumerate(all_clause_str_input):
            cl_clean = cl.replace("(", "").replace(")", "").replace(" ", "").replace("&", " & ")
            cl_clean_inv = ("~" + cl_clean.replace(" & ", " & ~")).replace("~~", "").replace(" & ", " | ")
            dico_clause[index_c] = [cl_clean, cl_clean_inv]
        nbre_clause_input = len(all_clause_str_input)
        str_filter_all = exp_filter_2eme_couche[filter].values[0]
        str_filter_all_lst = str_filter_all.split("|")
        for str_filter_i, str_filter in enumerate(str_filter_all_lst):
            print(str_filter_i+1, "/", len(str_filter_all_lst))
            M = get_all_masks_from_1clausefilter_allclauseinput(dico_clause, nbre_clause_input, str_filter, M, impossibillty)

        for clause_ici_index, clause_ici in enumerate(M):
            element_clause = clause_ici.split("&")
            M = 0.5 * np.ones((3, 9), dtype=np.uint8)
            for el_c in element_clause:
                if "DL[l]" in el_c:
                    if "~" in el_c:
                        M[0][0] = 0
                    else:
                        M[0][0] = 1
                for l_str in range(1,9):
                    if "DL[l+"+str(l_str)+"]" in el_c:
                        if "~" in el_c:
                            M[0][l_str] = 0
                        else:
                            M[0][l_str] = 1

                if "V0[l]" in el_c:
                    if "~" in el_c:
                        M[1][0] = 0
                    else:
                        M[1][0] = 1
                for l_str in range(1,9):
                    if "V0[l+"+str(l_str)+"]" in el_c:
                        if "~" in el_c:
                            M[1][l_str] = 0
                        else:
                            M[1][l_str] = 1

                if "V1[l]" in el_c:
                    if "~" in el_c:
                        M[2][0] = 0
                    else:
                        M[2][0] = 1
                for l_str in range(1, 9):
                    if "V1[l+" + str(l_str) + "]" in el_c:
                        if "~" in el_c:
                            M[2][l_str] = 0
                        else:
                            M[2][l_str] = 1
            image = M
            row_labels = ["DL", "V0", "V1"]
            col_labels = ["l"] + ["l+"+str(i) for i in range(1,9)]

            cmap = colors.ListedColormap(['black', 'grey', 'white'])
            bounds = [0, 0.25, 0.75, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            # tell imshow about color map so that only set colors are used
            img = plt.imshow(image, interpolation='nearest', origin='lower',
                             cmap=cmap, norm=norm)

            # make a color bar
            #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 0.25, 0.75, 1])

            plt.xticks(range(9), col_labels)
            plt.yticks(range(3), row_labels)
            plt.title('Mask ' + str(clause_ici) + ' clause Filtre numeros ' + str(filter))
            path = "./results/imgs/" + str(filter)
            try:
                os.mkdir(path)
            except OSError as exc:  # Python >2.5
                pass
            plt.savefig(path + "/mask_"+str(clause_ici)+".png")
            plt.clf()
        print("Numeros filtre ", num_filter," nom filtre:", filter," nbre de masks ", len(M))
        del M


