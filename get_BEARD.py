from data_process import get_timeline, timeline_convert_merge_post, compute_tfidf
from preprocess_pheme import preprocess_PHEME
import pickle


def add_timeline(data):
    for eid, value in data.items():
        info = data[eid]['info']
        tids, timeline, texts = get_timeline(info=info)

        data[eid]['timeline'] = timeline
        data[eid]['texts'] = texts
    return data


def save_pickle(data, path, filename):
    with open(f"{path}/{filename}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path, filename):
    with open(f"{path}/{filename}.pickle", "rb") as f:
        data = pickle.load(f)
        return data


# data = preprocess_PHEME()
data = load_pickle(path='data', filename='PAN_to_rawPHEME')
data = add_timeline(data)
data = timeline_convert_merge_post(data)
data = compute_tfidf(data)

save_pickle(data=data, path='data', filename='PAN_to_PHEME')
