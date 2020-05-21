from __future__ import print_function, division
import time
import shap
import numpy as np
from sklearn.decomposition import PCA
from utils.train_Gohr_nn import load_nn



def upgrade_mask_middle1(args, shap_values_sample, masks, listethr,
                         liste_all_mask, bits_faible_masks, bol_thr=False):
    pca = PCA(n_components=3)
    nbit = 16 * len(masks)

    if shap_values_sample.shape[0]>3:

        X_reduced = pca.fit(shap_values_sample).transform(shap_values_sample)
        all_v = abs(pca.components_[0])*100*pca.explained_variance_ratio_[0]
        for kkkk in range(1, 3):
            all_v += abs(pca.components_[kkkk])*100*pca.explained_variance_ratio_[kkkk]
        #print(all_v)
        for thrh in listethr:
            mask = np.zeros(nbit)
            if bol_thr:
                all_vmask = all_v > thrh
                all_vmask2 = []
                for i in range(args.nbre_bit_faible):
                    index_recall = list(all_v).index(np.sort(all_v[all_vmask])[:args.nbre_bit_faible][i])
                    all_vmask2.append(index_recall)
                all_vmask2 = np.array(all_vmask2)
            else:
                all_vmask = np.argsort(all_v)[::-1][:int(thrh)]
                all_vmask2 = np.argsort(all_v)[::-1][int(thrh)-args.nbre_bit_faible:int(thrh)]
            mask[all_vmask] = 1
            verify_mm = []
            masks_for_moment = []
            for index_m in range(len(masks)):
                masks_for_moment.append(int("".join(str(int(x)) for x in mask[16*index_m:16*index_m+16]), 2))
                verify_mm.append(masks_for_moment[-1])
            if tuple(verify_mm) not in liste_all_mask:
                liste_all_mask.append(tuple(verify_mm))
                for index_m in range(len(masks)):
                    masks[index_m].append(masks_for_moment[index_m])
                bits_faible_masks.append(all_vmask2)

    return liste_all_mask, masks, bits_faible_masks

def upgrade_mask_middle2(args, X_train, index_interettrain, masks, listethr2,
                         liste_all_mask, bits_faible_masks, bol_thr=False):
    nbit = 16 * len(masks)
    pca = PCA(n_components=3)
    X_reduced = pca.fit(X_train[index_interettrain]).transform(X_train[index_interettrain])
    all_v = abs(pca.components_[0]) * 100 * pca.explained_variance_ratio_[0]
    for kkkk in range(1, 3):
        all_v += abs(pca.components_[kkkk]) * 100 * pca.explained_variance_ratio_[kkkk]
    # print(all_v)
    for thrh in listethr2:
        mask = np.zeros(nbit)
        if bol_thr:
            all_vmask = all_v > thrh
            all_vmask2 = []
            for i in range(args.nbre_bit_faible):
                index_recall = list(all_v).index(np.sort(all_v[all_vmask])[:args.nbre_bit_faible][i])
                all_vmask2.append(index_recall)
            all_vmask2 = np.array(all_vmask2)
        else:
            all_vmask = np.argsort(all_v)[::-1][:int(thrh)]
            all_vmask2 = np.argsort(all_v)[::-1][int(thrh) - args.nbre_bit_faible:int(thrh)]
        mask[all_vmask] = 1
        verify_mm = []
        masks_for_moment = []
        for index_m in range(len(masks)):
            masks_for_moment.append(int("".join(str(int(x)) for x in mask[16 * index_m:16 * index_m + 16]), 2))
            verify_mm.append(masks_for_moment[-1])
        if tuple(verify_mm) not in liste_all_mask:
            liste_all_mask.append(tuple(verify_mm))
            for index_m in range(len(masks)):
                masks[index_m].append(masks_for_moment[index_m])
            bits_faible_masks.append(all_vmask2)
    return liste_all_mask, masks, bits_faible_masks



def upgrade_mask(args, index_interet, k, X_train, index_interettrain, X_eval, nombre_sample_shapley, liste_all_mask, masks, max_int_bit, net, bits_faible_masks, listethr=[2, 5], listethr2 = [1, 1.5 ]):
    background = X_train[index_interettrain]
    test_images = X_eval[index_interet][(k + 1) * nombre_sample_shapley:(k + 2) * nombre_sample_shapley]
    start = time.time()
    e = shap.GradientExplainer(net, background)
    #e = PyTorchDeepExplainer(model, background)
    #shap_values = e.shap_values(test_images)
    print("temps train", time.time()-start)
    start = time.time()
    shap_values_sample = e.shap_values(test_images)[0]
    print("temps val", time.time()-start)
    #shap_values_sample_resphae = shap_values_sample.reshape(100, 4, 16)

    #test_images_res = test_images.reshape(100, 4, 16)



    liste_all_mask, masks,bits_faible_masks =upgrade_mask_middle1(args, shap_values_sample, masks, args.listethr,
                         liste_all_mask, bits_faible_masks, bol_thr=False)
    liste_all_mask, masks,bits_faible_masks =upgrade_mask_middle1(args, shap_values_sample, masks, args.listethr3,
                         liste_all_mask, bits_faible_masks, bol_thr=True)




    if X_train[index_interettrain].shape[0] > 0:

        liste_all_mask, masks,bits_faible_masks =upgrade_mask_middle2(args, X_train, index_interettrain, masks, args.listethr2,
                             liste_all_mask, bits_faible_masks, bol_thr=False)
        liste_all_mask, masks, bits_faible_masks = upgrade_mask_middle2(args, X_train, index_interettrain, masks,
                                                                        args.listethr4,
                                                                        liste_all_mask, bits_faible_masks,
                                                                        bol_thr=True)




    liste_all_mask, masks,bits_faible_masks = update_masks(shap_values_sample, liste_all_mask,
                                                    masks, bits_faible_masks, args, max_int_bit=max_int_bit)




    return liste_all_mask, masks, bits_faible_masks




def getting_masks_general(args, path_file_models, rng, liste_all_mask, masks, bits_faible_masks):
    for k in range(args.nbre_de_mask_init):

        k = 0

        net, X_deltaout, Y_tf, X_eval, Y_eval, ctdata0l, ctdata0r, ctdata1l, ctdata1r = load_nn(args, args.nbre_sample_DDT,
                                                                                                args.nbre_sampleval,
                                                                                                args.nombre_round_eval,
                                                                                                path_file_models, rng,
                                                                                                depth=2)





        pred_net_eval = net.predict(X_eval[:args.max_limit_int]).squeeze(1)
        pred_net_train = net.predict(X_deltaout[:args.max_limit_int]).squeeze(1)

        index_interet = Y_eval[:args.max_limit_int] == 1
        index_interettrain = Y_tf[:args.max_limit_int] == 1



        liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                        X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                        liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr,
                                                        args.listethr2)


        liste_max_min2 = [tuple(map(float, sub.split(', '))) for sub in args.liste_max_min]

        for (valmax, valimin) in liste_max_min2:
            print(valmax, valimin)


            index_interet = np.logical_and(pred_net_eval > valimin, valmax > pred_net_eval)
            index_interettrain = np.logical_and(pred_net_train > valimin, valmax > pred_net_train)
            liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                            X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                            liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr, args.listethr2)


            index_interet = np.logical_and(pred_net_eval > valimin, Y_eval[:args.max_limit_int ]==1)
            index_interettrain = np.logical_and(pred_net_train > valimin, Y_tf[:args.max_limit_int ]==1)
            liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                            X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                            liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr, args.listethr2)

            index_interet = np.logical_and(pred_net_eval > valimin, Y_eval[:args.max_limit_int] != 1)
            index_interettrain = np.logical_and(pred_net_train > valimin, Y_tf[:args.max_limit_int] != 1)
            liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                            X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                            liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr, args.listethr2)

            index_interet = np.logical_and(pred_net_eval < valmax, Y_eval[:args.max_limit_int] != 1)
            index_interettrain = np.logical_and(pred_net_train < valmax, Y_tf[:args.max_limit_int] != 1)
            liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                            X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                            liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr, args.listethr2)

            index_interet = np.logical_and(pred_net_eval < valmax, Y_eval[:args.max_limit_int] == 1)
            index_interettrain = np.logical_and(pred_net_train < valmax, Y_tf[:args.max_limit_int] == 1)
            liste_all_mask, masks, bits_faible_masks = upgrade_mask(args, index_interet, k, X_deltaout[:args.max_limit_int], index_interettrain,
                                                            X_eval[:args.max_limit_int], args.nombre_sample_shapley,
                                                            liste_all_mask, masks, args.max_int_bit, net, bits_faible_masks, args.listethr, args.listethr2)



        print("iterations: ", k + 1, "Nbre de masks:", len(masks[0]))

        del X_deltaout, Y_tf, X_eval, Y_eval, ctdata0l, ctdata0r, ctdata1l, ctdata1r

        #assert len(ml_list)==len(mr_list)==len(liste_all_mask)
        for i in range(len(args.inputs_type) - 1):
            assert len(masks[i]) == len(masks[i + 1])

        assert len(masks[0]) == len(bits_faible_masks)


    return masks, bits_faible_masks




def update_masks(shap_values_sample, liste_all_mask, masks, bits_faible_masks, args,max_int_bit = [18]):
    nbit= 16*len(masks)

    for max_int in max_int_bit:
        mask_tot = mask = np.zeros(nbit)
        for i in range(shap_values_sample.shape[0]):
            arg_sorted_array = np.argsort(abs(shap_values_sample[i]))[::-1]
            arg_sorted_array = arg_sorted_array[:max_int]
            mask = np.zeros(nbit)
            mask[arg_sorted_array] = 1
            mask_tot[arg_sorted_array] += 1 / shap_values_sample.shape[0]
            # print(mask[:4], mask[4:8], mask[8:12], mask[12:16], mask[16:20], mask[16:20], mask[24:28], mask[28:])
        # print(mask_tot[:4], mask_tot[4:8], mask_tot[8:12], mask_tot[12:16], mask_tot[16:20], mask_tot[16:20], mask_tot[24:28], mask_tot[28:])

        arg_sorted_array = np.argsort(mask_tot)[::-1]
        arg_sorted_array = arg_sorted_array[:max_int]
        all_vmask2 = np.argsort(mask_tot)[::-1][max_int - args.nbre_bit_faible:max_int]

        mask = np.zeros(nbit)
        mask[arg_sorted_array] = 1
        # print(mask[:4], mask[4:8], mask[8:12], mask[12:16], mask[16:20], mask[16:20], mask[24:28], mask[28:])
        verify_mm = []
        masks_for_moment = []
        for index_m in range(len(masks)):
            masks_for_moment.append(int("".join(str(int(x)) for x in mask[16 * index_m:16 * index_m + 16]), 2))
            verify_mm.append(masks_for_moment[-1])
        if tuple(verify_mm) not in liste_all_mask:
            liste_all_mask.append(tuple(verify_mm))
            for index_m in range(len(masks)):
                masks[index_m].append(masks_for_moment[index_m])
            bits_faible_masks.append(all_vmask2)


        shap_values_sample_sum = np.sum(abs(shap_values_sample), axis=0)
        arg_sorted_array = np.argsort(shap_values_sample_sum)[::-1]
        arg_sorted_array = arg_sorted_array[:max_int]
        all_vmask2 = np.argsort(shap_values_sample_sum)[::-1][max_int - args.nbre_bit_faible:max_int]

        mask = np.zeros(nbit)
        mask[arg_sorted_array] = 1
        # print(mask[:4], mask[4:8], mask[8:12], mask[12:16], mask[16:20], mask[16:20], mask[24:28], mask[28:])
        verify_mm = []
        masks_for_moment = []
        for index_m in range(len(masks)):
            masks_for_moment.append(int("".join(str(int(x)) for x in mask[16 * index_m:16 * index_m + 16]), 2))
            verify_mm.append(masks_for_moment[-1])
        if tuple(verify_mm) not in liste_all_mask:
            liste_all_mask.append(tuple(verify_mm))
            for index_m in range(len(masks)):
                masks[index_m].append(masks_for_moment[index_m])
            bits_faible_masks.append(all_vmask2)


        shap_values_sample_mean = np.mean(abs(shap_values_sample), axis=0)
        arg_sorted_array = np.argsort(shap_values_sample_mean)[::-1]
        arg_sorted_array = arg_sorted_array[:max_int]
        all_vmask2 = np.argsort(shap_values_sample_mean)[::-1][max_int - args.nbre_bit_faible:max_int]

        mask = np.zeros(nbit)
        mask[arg_sorted_array] = 1
        # print(mask[:4], mask[4:8], mask[8:12], mask[12:16], mask[16:20], mask[16:20], mask[24:28], mask[28:])
        verify_mm = []
        masks_for_moment = []
        for index_m in range(len(masks)):
            masks_for_moment.append(int("".join(str(int(x)) for x in mask[16 * index_m:16 * index_m + 16]), 2))
            verify_mm.append(masks_for_moment[-1])
        if tuple(verify_mm) not in liste_all_mask:
            liste_all_mask.append(tuple(verify_mm))
            for index_m in range(len(masks)):
                masks[index_m].append(masks_for_moment[index_m])
            bits_faible_masks.append(all_vmask2)


        shap_values_sample_med = np.median(abs(shap_values_sample), axis=0)
        arg_sorted_array = np.argsort(shap_values_sample_med)[::-1]
        arg_sorted_array = arg_sorted_array[:max_int]
        all_vmask2 = np.argsort(shap_values_sample_med)[::-1][max_int - args.nbre_bit_faible:max_int]

        mask = np.zeros(nbit)
        mask[arg_sorted_array] = 1
        verify_mm = []
        masks_for_moment = []
        for index_m in range(len(masks)):
            masks_for_moment.append(int("".join(str(int(x)) for x in mask[16 * index_m:16 * index_m + 16]), 2))
            verify_mm.append(masks_for_moment[-1])
        if tuple(verify_mm) not in liste_all_mask:
            liste_all_mask.append(tuple(verify_mm))
            for index_m in range(len(masks)):
                masks[index_m].append(masks_for_moment[index_m])
            bits_faible_masks.append(all_vmask2)



    return liste_all_mask, masks, bits_faible_masks