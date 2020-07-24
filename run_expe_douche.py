import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'
#["baseline", "deepset", "cnn_attention", "multihead"]


command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class --make_data_equilibre_3class No"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 5 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)

command = "python3 main_3class.py --nombre_round_eval 6 --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_3class"
os.system(command)




command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)

command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 5"
os.system(command)
command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 6"
os.system(command)
#command = "python3 main.py --type_model baseline_bin_v2 --nombre_round_eval 7"
#os.system(command)