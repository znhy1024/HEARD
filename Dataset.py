
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class EDataset(Dataset):

    def __init__(self, data, config):

        self.interval = config["models"][config["active_model"]
                                         ]["hyperparameters"]["interval"]
        self.max_seq_len = config["models"][config["active_model"]
                                            ]["hyperparameters"]["max_seq_len"]
        self.max_post_len = config["models"][config["active_model"]
                                             ]["hyperparameters"]["max_post_len"]

        self.data = data
        self.fold_x = list(self.data.keys())

    def __len__(self):
        return len(self.fold_x)

    def __label_convert__(self, label):
        if label == 1:
            target = [0, 1]
        else:
            target = [1, 0]
        return target

    def __time_convert__(self, merge_times):

        merge_times = [[datetime.strptime(
            time, '%m-%d-%y %H:%M:%S').timestamp() for time in merge_time]for merge_time in merge_times]
        # ---
        start_timestamp = merge_times[0][-1]
        time_last_arrival = [0.0]
        time_since_starts = []

        for x in merge_times:
            tmp = (x[-1]-start_timestamp)/self.interval
            assert tmp >= 0
            time_since_starts.append(tmp)

        for x in zip(time_since_starts[1:], time_since_starts[:-1]):
            tmp = x[0]-x[1]
            assert tmp >= 0
            time_last_arrival.append(tmp)

        return time_since_starts, time_last_arrival

    def __index_convert__(self, merge_tids):
        last_post_index = [0]*len(merge_tids)
        for i, seq in enumerate(merge_tids):
            last_post_index[i] = len(seq)+last_post_index[i-1]
        return last_post_index

    def __getitem__(self, idx):

        eid = self.fold_x[idx]
        seqs = self.data[eid]['merge_seqs']
        label = self.data[eid]['label']

        all_lens = list(map(len, seqs['merge_times']))
        seq_length = sum(all_lens)

        merge_times = seqs['merge_times'][:self.max_seq_len]
        merge_tids = seqs['merge_tids'][:self.max_seq_len]
        text_vec_seq_t = seqs['merge_vecs'][:self.max_seq_len]
        try:
            time_since_starts, time_last_arrival = self.__time_convert__(
                merge_times[:self.max_seq_len])
        except Exception as e:
            print(eid)
            print(merge_times)
            print(e)

        last_post_index = self.__index_convert__(merge_tids)
        target = self.__label_convert__(int(label))
        text_vec_seq = text_vec_seq_t
        return label, target, text_vec_seq, last_post_index, time_since_starts, time_last_arrival, seq_length, eid, merge_tids


class HDataLoader(object):
    def __init__(self, config):

        self.device = config["models"][config["active_model"]]["device"]

        self.batchsize = config["models"][config["active_model"]
                                          ]["hyperparameters"]["batch_size"]
        self.text_feats = config["models"][config["active_model"]
                                           ]["hyperparameters"]["in_feats_RD"]

        self.data = pickle.load(
            open(config["models"][config["active_model"]]["data"], 'rb'))

        self.dataids = pickle.load(
            open(config["models"][config["active_model"]]["data_ids"], 'rb'))

        self.load_data()
        self.init_dataset(config)

    def load_data(self):
        self.val_set = {}
        val_ids = self.dataids['val']
        for eid in val_ids:
            self.val_set[eid] = self.data[eid]

        def inner_function():
            folds = {}
            for fold in self.dataids.keys():
                if fold == 'val':
                    continue
                else:
                    train_ids = self.dataids[fold]['train']
                    folds[fold] = {'train': {}, 'test': {}}
                    for eid in train_ids:
                        folds[fold]['train'][eid] = self.data[eid]
                    test_ids = self.dataids[fold]['test']
                    for eid in test_ids:
                        folds[fold]['test'][eid] = self.data[eid]
            return folds
        self.folds = inner_function()

    def pad_batch_fn(self, batch_data):
        sorted_batch = sorted(
            batch_data, key=lambda x: len(x[2]), reverse=True)
        eids = [seq[7] for seq in sorted_batch]
        label_seqs = torch.LongTensor([int(seq[0]) for seq in sorted_batch])
        target_seqs = torch.LongTensor([seq[1] for seq in sorted_batch])
        text_seqs = [seq[2] for seq in sorted_batch]
        seqs_length = torch.LongTensor(list(map(len, text_seqs)))
        real_lengths = torch.LongTensor([seq[6] for seq in sorted_batch])
        post_since_start_seqs = [seq[3] for seq in sorted_batch]
        last_arrival_time_seqs = [seq[5] for seq in sorted_batch]
        times_since_start_seqs = [seq[4] for seq in sorted_batch]
        all_tids = [seq[8] for seq in sorted_batch]
        end_time_seqs = torch.FloatTensor(
            [seq[-1] for seq in times_since_start_seqs])
        posts_length, max_post_len = [[] for x in text_seqs], None

        text_seqs_tensor = torch.zeros(
            len(sorted_batch), seqs_length.max(), self.text_feats).float()

        arrival_time_seqs_tensor = torch.zeros(
            len(sorted_batch), seqs_length.max()).float()
        times_since_seqs_tensor = torch.zeros(
            len(sorted_batch), seqs_length.max()).float()
        post_since_start_seqs_tensor = torch.zeros(
            len(sorted_batch), seqs_length.max()).long()
        posts_length_tensor = torch.ones(
            len(sorted_batch), seqs_length.max()).long()

        for idx, (text_seq, time_seq, seqlen, timstamp_seq, index_seq, posts_len) in enumerate(zip(text_seqs, last_arrival_time_seqs, seqs_length, times_since_start_seqs, post_since_start_seqs, posts_length)):

            text_seqs_tensor[idx, :seqlen, :] = torch.FloatTensor(text_seq)

            arrival_time_seqs_tensor[idx,
                                     :seqlen] = torch.FloatTensor(time_seq)
            times_since_seqs_tensor[idx,
                                    :seqlen] = torch.FloatTensor(timstamp_seq)
            post_since_start_seqs_tensor[idx,
                                         :seqlen] = torch.LongTensor(index_seq)

        return label_seqs, target_seqs, text_seqs_tensor, post_since_start_seqs_tensor, arrival_time_seqs_tensor, end_time_seqs, seqs_length, times_since_seqs_tensor, posts_length_tensor, max_post_len, real_lengths, eids, all_tids

    def init_dataset(self, config):
        self.val_set = EDataset(self.val_set, config)
        self.train_set = {}
        self.test_set = {}
        for fold, data in self.folds.items():
            self.train_set[fold] = EDataset(data['train'], config)
            self.test_set[fold] = EDataset(data['test'], config)

    def get_loaders(self):
        val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False,
                                num_workers=5, collate_fn=self.pad_batch_fn, drop_last=True)
        folds_loader = {}
        for fold in self.train_set:
            folds_loader[fold] = {}
            folds_loader[fold]['train'] = DataLoader(
                self.train_set[fold], batch_size=self.batchsize, shuffle=True, num_workers=5, collate_fn=self.pad_batch_fn, drop_last=True)
            folds_loader[fold]['test'] = DataLoader(
                self.test_set[fold], batch_size=1, shuffle=False, num_workers=5, collate_fn=self.pad_batch_fn, drop_last=True)
            folds_loader[fold]['num'] = (
                len(self.train_set[fold]), len(self.test_set[fold]))
        return len(self.val_set), val_loader, folds_loader


def get_dataloader(config):

    handle = HDataLoader(config)
    val_len, val_loader, folds_loader = handle.get_loaders()
    return val_len, val_loader, folds_loader
