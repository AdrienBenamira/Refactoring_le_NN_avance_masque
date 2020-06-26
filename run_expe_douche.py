import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]




command = "python3 main.py --nombre_round_eval 5"
os.system(command)

command = "python3 main.py --nombre_round_eval 6"
os.system(command)

command = "python3 main.py --nombre_round_eval 7"
os.system(command)

command = "python3 main.py --nombre_round_eval 8"
os.system(command)

command = "python3 main.py --nombre_round_eval 4"
os.system(command)

command = "python3 main.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

command = "python3 main.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

command = "python3 main.py --nombre_round_eval 7 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

command = "python3 main.py --nombre_round_eval 8 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

command = "python3 main.py --nombre_round_eval 4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]'"
os.system(command)

"""
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5 & python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 6 --device 2"
os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 4 & python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 7 --device 2"
os.system(command)

command = "python3 main.py --type_model baseline_bin_v4 --nombre_round_eval 4 & python3 main.py --type_model baseline_bin_v4 --nombre_round_eval 5 --device 2"
os.system(command)

command = "python3 main.py --type_model baseline_bin_v4 --nombre_round_eval 6 & python3 main.py --type_model baseline_bin_v4 --nombre_round_eval 7 --device 2"
os.system(command)

command = "python3 main.py --type_model baseline_bin_v4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5 & python3 main.py --type_model baseline_bin_v4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 6 --device 2"
os.system(command)

command = "python3 main.py --type_model baseline_bin_v4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 4 & python3 main.py --type_model baseline_bin_v4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 7 --device 2"
os.system(command)
"""
#command = "python3 main.py --type_model baseline_bin_v4 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 8 & python3 main.py --type_model baseline_bin_v4 --nombre_round_eval 8 --device 2"
#os.system(command)




