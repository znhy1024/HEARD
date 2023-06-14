import os
import time
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from Dataset import get_dataloader
from Model import HEARD
from copy import deepcopy
from tqdm import tqdm
import wandb


class EarlyStopping:
    def __init__(self, stop_rate, patience, model_dir):

        self.stop_rate = stop_rate
        self.patience = patience
        self.count = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

        self.best_params = {'loss': None}
        self.model_dir = model_dir
        for mdir in self.model_dir.values():
            if not os.path.exists(mdir):
                os.makedirs(mdir)

    def __call__(self, val_loss, model, rd_lr):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.best_score, model)
        elif score < self.best_score:
            self.count += 1
            if rd_lr < self.stop_rate or self.count > self.patience:
                self.early_stop = True
                print(
                    'Least Validation loss ({:.6f}).'.format(-self.val_loss_min))
        else:
            self.best_score = score
            self.save_checkpoint(self.best_score, model)
            self.count = 0

    def save_checkpoint(self, val_loss, model):

        self.val_loss_min = val_loss
        self.best_params['loss'] = deepcopy(model.state_dict())
        torch.save(model.state_dict(),
                   self.model_dir['loss']+'best_model'+'.m')


class Train(object):

    def __init__(self, config):

        self.config = config

        self.device = config["models"][config["active_model"]]["device"]
        self.dataset = config["models"][config["active_model"]]["dataset"]
        self.model_dir = config["models"][config["active_model"]
                                          ]["model_dir"]+config["active_model"]+'/'

        self.decay_patience = config["models"][config["active_model"]
                                               ]["hyperparameters"]["decay_patience"]
        self.stop_rate = config["models"][config["active_model"]
                                          ]["early_stop_lr"]
        self.stop_patience = config["models"][config["active_model"]
                                              ]["early_stop_patience"]

        self.interval = config["models"][config["active_model"]
                                         ]["hyperparameters"]["interval"]

        self.val_len, self.val_loader, self.folds_loader = get_dataloader(
            config)

    def set_optimizer(self, config, model):
        optimizer = torch.optim.Adam([
            {'params': model.parameters(
            ), 'lr': config["models"][config["active_model"]]["hyperparameters"]["learning_rate"]['RD']}
        ], weight_decay=config["models"][config["active_model"]]["hyperparameters"]["weight_decay"])
        return optimizer

    def set_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               'max',
                                                               patience=self.decay_patience,
                                                               factor=0.5,
                                                               threshold=1e-3)
        return scheduler

    def train_HEARD(self, fold, train_loader, val_loader):
        names = ['class{}'.format(i) for i in range(2)]
        n_epochs = self.config["models"][self.config["active_model"]
                                         ]["hyperparameters"]["epochs"]

        model = HEARD(self.config).to(self.device)
        optimizer = self.set_optimizer(self.config, model)
        scheduler = self.set_scheduler(optimizer)
        model.train()

        model_dir = {"loss": self.model_dir+self.dataset+'/'+f'{fold}/'}
        early_stopping = EarlyStopping(
            self.stop_rate, self.stop_patience, model_dir)
        for epoch in tqdm(range(n_epochs)):
            st = time.time()
            avg_loss = []
            val_loss = []

            model.train()
            preds = torch.LongTensor()
            trues = torch.LongTensor()
            early_rate = torch.FloatTensor()
            print(f'start training {epoch} epoch: ' +
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            for Batch_data in tqdm(train_loader):

                label_seqs, post_indexes, real_lens = Batch_data[0], Batch_data[3], Batch_data[10]
                outputs = model(Batch_data)
                loss, stop_points, stop_preds, _ = model.compute_log_likelihood(
                    Batch_data)

                optimizer.zero_grad()
                loss.backward()
                avg_loss.append(loss.item())
                nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
                optimizer.step()
                trues = torch.cat((trues, label_seqs.cpu().data))
                preds = torch.cat((preds, stop_preds.cpu().data))

                stops_index = post_indexes.cpu().data[torch.arange(
                    0, len(stop_preds)), stop_points]*1.0
                er = stops_index / real_lens
                early_rate = torch.cat((early_rate, er.cpu().data))

            evals = classification_report(
                trues, preds, target_names=names, output_dict=True, zero_division=0
            )
            train_acc = evals['accuracy']
            train_recall = evals['macro avg']['recall']
            train_f1 = evals['macro avg']['f1-score']
            train_precision = evals['macro avg']['precision']
            lr_n = optimizer.param_groups[0]['lr']
            print("Epoch {:05d} | Train_Loss {:.4f}| Train_Acc {:.4f} | ER : {:.4f}| Train_R {:.4f} | Train_P {:.4f} | Train_F {:.4f} | lr : {}|".format(
                epoch, np.mean(avg_loss), train_acc, torch.mean(early_rate).cpu().data, train_recall, train_precision, train_f1, lr_n))

            wandb.log(
                {
                    "Train_Loss": np.mean(avg_loss),
                    "Train_epoch": epoch,
                    "Train_Acc": train_acc,
                    "ER": torch.mean(early_rate).cpu().data,
                    "Train_recall": train_recall,
                    "Train_precision": train_precision,
                    "Train_F1": train_f1,
                    "learning_rate": lr_n,
                }
            )
            print(f'[+]Start eval {epoch} epoch: ' +
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                preds = torch.LongTensor()
                trues = torch.LongTensor()
                early_rate = torch.FloatTensor()

                for Batch_data in tqdm(val_loader):

                    label_seqs, post_indexes, real_lens = Batch_data[0], Batch_data[3], Batch_data[10]
                    outputs = model(Batch_data, if_dp=False)
                    loss, stop_points, stop_preds, _ = model.compute_log_likelihood(
                        Batch_data, if_dp=False)
                    val_loss.append(loss.item())
                    trues = torch.cat((trues, label_seqs[0:1].cpu().data))
                    preds = torch.cat((preds, stop_preds[0:1].cpu().data))

                    stops_index = post_indexes.cpu(
                    ).data[0, stop_points[0]]*1.0

                    er = stops_index / real_lens
                    early_rate = torch.cat((early_rate, er.cpu().data))

                evals = classification_report(
                    trues, preds, target_names=names, output_dict=True, zero_division=0
                )
                val_acc = evals['accuracy']
                val_recall = evals['macro avg']['recall']
                val_f1 = evals['macro avg']['f1-score']
                val_precision = evals['macro avg']['precision']
                print("Epoch {:05d} | Val_Acc {:.4f} | ER : {:.4f} | Val_R {:.4f} | Val_P {:.4f} | Val_F {:.4f}".format(epoch,
                                                                                                                        val_acc, torch.mean(early_rate).cpu().data, val_recall, val_precision, val_f1))
                wandb.log(
                    {
                        "Val_Loss": np.mean(val_loss),
                        "Val_epoch": epoch,
                        "Val_Acc": val_acc,
                        "ER": torch.mean(early_rate).cpu().data,
                        "Val_recall": val_recall,
                        "Val_precision": val_precision,
                        "Val_F1": val_f1,
                    }
                )
            et = time.time()
            scheduler.step(val_acc)
            print(f'It takes {et-st}s for an epoch.')
            lr_n = optimizer.param_groups[0]['lr']
            early_stopping(np.mean(val_loss), model, lr_n)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return model, early_stopping.best_params

    def inverse_count(self, preds):
        counts = 0
        if len(preds) == 1:
            return 0.0
        for pi, pred in enumerate(preds):
            if pi == len(preds)-1:
                break
            next_pred = preds[pi+1]
            counts += int(pred != next_pred)
        stable = counts / (len(preds)-1)
        return stable

    def compute_early_stable(self, accs):
        all_eids = list(accs.keys())
        for eid in all_eids:
            record = accs[eid]
            stop_index = int(record['stop_point'].item())
            ts = record['time']
            preds = record['preds']
            autostop = record['delta_N_at_stops']

            assert stop_index >= 0
            if autostop != 0:
                early = np.Inf
            else:
                early = ts[stop_index]

            stable = self.inverse_count(preds[stop_index:])
            accs[eid]['early'] = early*self.interval
            accs[eid]['stable'] = stable
        return accs

    @torch.no_grad()
    def test_HEARD(self, test_loader, model_paras):
        predicts = {}
        torch.cuda.empty_cache()
        with torch.no_grad():
            model = HEARD(self.config)
            model.load_state_dict(torch.load(model_paras))
            model.to(self.device)
            model.eval()
            preds = torch.LongTensor()
            trues = torch.LongTensor()
            early_rate = torch.FloatTensor()
            for Batch_data in test_loader:
                label_seqs, seq_lengths, post_indexes, real_lens, eids, tids = Batch_data[
                    0], Batch_data[6], Batch_data[3], Batch_data[10], Batch_data[11], Batch_data[12]

                outputs = model(Batch_data, if_dp=False)

                eid = eids[0]
                timestamp = Batch_data[7][0].cpu().data.tolist()
                predicts[eid] = {'preds': outputs[0][2][:, 0].cpu().data.tolist(
                ), 'time': timestamp, 'label': label_seqs.cpu().data.tolist()[0]}

                _, stop_points, stop_preds, delta_N_at_stops = model.compute_log_likelihood(
                    Batch_data, if_dp=False)

                trues = torch.cat((trues, label_seqs[0:1].cpu().data))
                preds = torch.cat((preds, stop_preds[0:1].cpu().data))

                stops_index = post_indexes.cpu().data[0, stop_points[0]]*1.0
                er = stops_index / real_lens
                predicts[eid]['stop'] = stops_index
                predicts[eid]['stop_point'] = stop_points[0]
                predicts[eid]['delta_N_at_stops'] = delta_N_at_stops[0]
                early_rate = torch.cat((early_rate, er.cpu().data))

            evals = classification_report(
                trues, preds, output_dict=True, zero_division=0
            )

            test_acc = evals['accuracy']
            test_recall = evals['macro avg']['recall']
            test_f1 = evals['macro avg']['f1-score']
            test_precision = evals['macro avg']['precision']

            new_accs = self.compute_early_stable(predicts)
            all_stable = []
            for eid in new_accs:
                stable = new_accs[eid]['stable']
                all_stable.append(stable)
            avg_stable = np.mean(all_stable)
            avg_er = torch.mean(early_rate).cpu().data.item()
            sea = (1-avg_er+1-avg_stable+test_acc)/3
            print("Test_Acc {:.4f} | ER : {:.4f} | Test_R {:.4f} | Test_P {:.4f} | Test_F {:.4f} | SEA {:.4f}".format(
                test_acc, torch.mean(early_rate).cpu().data, test_recall, test_precision, test_f1, sea))

            wandb.log(
                {
                    "SEA": sea,
                    "Test_Acc": test_acc,
                    "ER": torch.mean(early_rate).cpu().data,
                    "Test_recall": test_recall,
                    "Test_precision": test_precision,
                    "Test_F1": test_f1,
                }
            )
        return test_acc, test_recall, test_precision, test_f1, torch.mean(early_rate).cpu().data.item(), sea
