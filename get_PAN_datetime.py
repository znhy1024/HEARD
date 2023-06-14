import pickle
import os

PATH = '/home/jinmyeong/code/HEARD'
os.chdir(PATH)


def load_pickle(path, filename):
    with open(f"{path}/{filename}.pickle", "rb") as f:
        data = pickle.load(f)
        return data


def save_pickle(data, path, filename):
    with open(f"{path}/{filename}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


PATH = '/home/jinmyeong/code/HEARD'
os.chdir(PATH)

path = 'data'
filename = 'PAN_to_PHEME'
PAN_to_PHEME = load_pickle(path=path, filename=filename)

datetime = '06-13-24 00:00:01'

for eid in PAN_to_PHEME:
    merge_times = PAN_to_PHEME[eid]['merge_seqs']['merge_times']
    for merge_time_idx in range(len(merge_times)):
        for time_idx in range(len(merge_times[merge_time_idx])):
            merge_times[merge_time_idx][time_idx] = datetime

save_pickle(PAN_to_PHEME, path, filename)
