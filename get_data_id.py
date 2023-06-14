import pickle
import os
import numpy as np
import random


def load_pickle(path, filename):
    with open(f"{path}/{filename}.pickle", "rb") as f:
        data = pickle.load(f)
        return data


def save_pickle(data, path, filename):
    with open(f"{path}/{filename}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def shuffle_lists(featllist, labellist, total_epoch, epoch, thirdparty=None):
    # seed = 100
    np.random.seed(1)
    seed_set = np.random.randint(100, size=total_epoch)
    seed_set = seed_set.tolist()

    if labellist is None:
        np.random.seed(seed_set[epoch])
        random.shuffle(featllist)
        return featllist
    elif labellist is not None and thirdparty is None:
        combined = list(zip(featllist, labellist))
        np.random.seed(seed_set[epoch])
        random.shuffle(combined)
        featllist, labellist = zip(*combined)
        return featllist, labellist
    else:
        combined = list(zip(featllist, labellist, thirdparty))
        np.random.seed(seed_set[epoch])
        random.shuffle(combined)
        featllist, labellist, thirdparty = zip(*combined)
        return featllist, labellist, thirdparty


def get_test_train(fold):
    len_fold = len(fold)
    fold_unit = int(len_fold / 5)

    test_list = fold[:fold_unit]
    train_list = fold[fold_unit:]

    return test_list, train_list

    # if len(fold) != 928:
    #     test_list = fold[:234]
    #     train_list = fold[234:]

    #     return test_list, train_list

    # test_list = fold[:232]
    # train_list = fold[232:]

    # return test_list, train_list


PATH = '/home/jinmyeong/code/HEARD'
os.chdir(PATH)

PHEME_path = 'data'
filename = 'PAN_to_PHEME'

PHEME_data = load_pickle(path=PHEME_path, filename=filename)

eid_list = list(PHEME_data.keys())

random.shuffle(eid_list)

# fold 나누기 -> 총 5개의 fold로 나누기
len_eid_list = len(eid_list)
fold_unit = int(len_eid_list / 6)

val_eid_list = eid_list[:fold_unit]

fold_0 = eid_list[fold_unit:fold_unit * 2]
fold_1 = eid_list[fold_unit * 2:fold_unit * 3]
fold_2 = eid_list[fold_unit * 3:fold_unit * 4]
fold_3 = eid_list[fold_unit * 4:fold_unit * 5]
fold_4 = eid_list[fold_unit * 5:]

fold_0_test, fold_0_train = get_test_train(fold_0)
fold_1_test, fold_1_train = get_test_train(fold_1)
fold_2_test, fold_2_train = get_test_train(fold_2)
fold_3_test, fold_3_train = get_test_train(fold_3)
fold_4_test, fold_4_train = get_test_train(fold_4)

data_ids = {"val": val_eid_list, "fold0": {'test': fold_0_test, 'train': fold_0_train}, "fold1": {'test': fold_1_test, 'train': fold_1_train}, "fold2": {
    'test': fold_2_test, 'train': fold_2_train}, "fold3": {'test': fold_3_test, 'train': fold_3_train}, "fold4": {'test': fold_4_test, 'train': fold_4_train}, }

save_pickle(data=data_ids, path=PHEME_path, filename='PAN_data_ids')
