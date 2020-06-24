import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


command = "python3 main.py --nombre_round_eval 4'"
os.system(command)

command = "python3 main.py --nombre_round_eval 7'"
os.system(command)

command = "python3 main.py --nombre_round_eval 8'"
os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5"
os.system(command)






