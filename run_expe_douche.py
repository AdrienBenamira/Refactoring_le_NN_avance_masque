import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


command = "python3 prunning.py --model_to_prune results/models_trained/speck/6/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_V0&inv(V1)_inv(V0)&inv(V1)/Gohr_baseline_best_nbre_sampletrain_10000000.pth"
os.system(command)

command = "python3 prunning.py --model_to_prune results/models_trained/speck/6/ctdata0l^ctdata1l_ctdata0r^ctdata1r^ctdata0l^ctdata1l_ctdata0l^ctdata0r_ctdata1l^ctdata1r/Gohr_baseline_best_nbre_sampletrain_10000000.pth"
os.system(command)

command = "python3 prunning.py --model_to_prune results/models_trained/speck/6/ctdata0l_ctdata0r_ctdata1l_ctdata1r/Gohr_baseline_best_nbre_sampletrain_10000000.pth"
os.system(command)