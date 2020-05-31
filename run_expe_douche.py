import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


command = "python3 main.py --inputs_type '[ctdata0r^ctdata1r]'"
os.system(command)

command = "python3 main.py --inputs_type '[ctdata0r^ctdata1r^ctdata0l^ctdata1l]'"
os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l^ctdata0r]' "
os.system(command)

command = "python3 main.py --inputs_type '[ctdata1l^ctdata1r]'"
os.system(command)

