import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


for model in ["baseline"]:
    for round in ([5, 6, 7, 8]):
        command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r, inv(V0)&inv(V1)]' " + " --type_model " + str(model)
        os.system(command)


for model in ["baseline","cnn_attention", "multihead"]:
    for round in ([5, 6, 7, 8]):
        command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]' " + " --type_model " + str(model)
        os.system(command)


