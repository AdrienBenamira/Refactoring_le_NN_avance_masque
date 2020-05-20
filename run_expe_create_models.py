import os

#'[ctdata0l, ctdata0r, ctdata1l, ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]', '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l]'


for curriculum in ["No"]:
    for continuous in ["No", "Yes"]:
        for model in ["baseline"]:
            for round in ([5, 6, 7, 8]):
                command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l, ctdata0r, ctdata1l, ctdata1r]' " + " --type_model " + str(model) + " --countinuous_learning " + str(continuous) + " --curriculum_learning " + str(curriculum)
                os.system(command)


for curriculum in ["No"]:
    for continuous in ["No", "Yes"]:
        for model in ["baseline"]:
            for round in ([5, 6, 7, 8]):
                command = "python3 main.py --nombre_round_eval " + str(round) + " --inputs_type '[ctdata0l^ctdata1l, ctdata0r^ctdata1r^ctdata0l^ctdata1l, ctdata0l^ctdata0r, ctdata1l^ctdata1r]' " + " --type_model " + str(model) + " --countinuous_learning " + str(continuous) + " --curriculum_learning " + str(curriculum)
                os.system(command)
