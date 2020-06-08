import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]



for round in ([5, 6]):
    command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, inv(DeltaV), inv(V0)&inv(V1), inv(V0)&V1]'"
    os.system(command)

for round in ([5, 6]):
    command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]'"
    os.system(command)

for round in ([5, 6]):
    command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1, ctdata1r]'"
    os.system(command)

