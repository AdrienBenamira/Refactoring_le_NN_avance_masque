from src.classifiers.nn_classifier_keras import train_speck_distinguisher


class Get_masks:


    def __init__(self, args, create_data_g, path_save_model, generator_data, get_masks_gen, nn_model_ref):
        self.args = args
        self.create_data_g = create_data_g
        self.path_save_model = path_save_model
        self.nn_model_ref = nn_model_ref
        self.get_masks_gen = get_masks_gen
        self.generator_data = generator_data
        self.X_train_proba = generator_data.X_proba_train
        self.Y_train_proba = generator_data.Y_create_proba_train
        self.X_eval_proba = generator_data.X_proba_val
        self.Y_eval_proba = generator_data.Y_create_proba_val
        if args.retrain_nn_ref:
            self.retrain_classifier_final()



    def retrain_classifier_final(self, args, nn_model_ref):
        nn_model_ref.epochs = args.num_epch_2
        nn_model_ref.batch_size_2 = args.batch_size_2
        nn_model_ref.net.freeze()
        X_train_proba_feat, X_eval_proba_feat = nn_model_ref.all_intermediaire, nn_model_ref.all_intermediaire_val
        Y_train_proba = generator_data.Y_create_proba_train
        Y_eval_proba = generator_data.Y_create_proba_val
        print("START RETRAIN LINEAR NN GOHR ")
        print()
        net_retrain, h = train_speck_distinguisher(args, X_train_proba_feat.shape[1], X_train_proba_feat,
                                                   Y_train_proba, X_eval_proba_feat, Y_eval_proba,
                                                   bs=args.batch_size_2,
                                                   epoch=args.num_epch_2, name_ici="retrain_nn_gohr",
                                                   wdir=self.path_save_model)


    def classifier_nn(self):
        net2, h = train_speck_distinguisher(self.args, len(self.get_masks_gen.masks[0]), self.X_train_proba,
                                            self.Y_train_proba, self.X_eval_proba, self.Y_eval_proba,
                                            bs=self.args.batch_size_our,
                                            epoch=self.args.num_epch_our, name_ici="our_model",
                                            wdir=self.path_save_model)

    def classifier_lgbm(self):
        pass

    def classifier_RF(self):
        pass

