import numpy as np
from numpy.core.numeric import Inf

import torch
import torch.nn as nn
import torch.nn.functional as F

# RD Modules
class RDLSTMCell(nn.Module):
    def __init__(self, hidden_dim, batch_size,dp_rate,device):
        super(RDLSTMCell, self).__init__()

        self.device = device

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.dp_rate = dp_rate

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 4, bias=True).to(self.device)
        nn.init.orthogonal_(self.linear.weight)

        self.dropout = nn.Dropout(self.dp_rate)
    
    def init_states(self):
       
        self.h_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        
    def forward(
            self, rnn_input,
            hidden_i_minus, cell_i_minus,if_dp=True):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_i_minus), dim=dim_of_hidden)

        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, = output_i.chunk(
            4, dim_of_hidden)

        gate_input = torch.sigmoid(gate_input)

        gate_forget = torch.sigmoid(gate_forget)

        gate_output = torch.sigmoid(gate_output)

        gate_pre_c = torch.tanh(gate_pre_c)-1

        if if_dp:
            cell_i = self.dropout(gate_forget) * cell_i_minus + self.dropout(gate_input) * gate_pre_c
            h_i = self.dropout(gate_output)*torch.tanh(cell_i)
        else:
            cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c
            h_i = gate_output*torch.tanh(cell_i)

        return h_i, (h_i,cell_i)

# HP Modules
class CTLSTMCell(nn.Module):
    def __init__(self, hidden_dim, batch_size,dp_rate,device):
        super(CTLSTMCell, self).__init__()

        self.device = device

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.dp_rate = dp_rate

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 7, bias=True).to(self.device)
    
    def init_states(self):

        self.h_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_bar = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)

    def forward(
            self, rnn_input,
            hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_t_i_minus), dim=dim_of_hidden)

        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(
            7, dim_of_hidden)

        gate_input = torch.sigmoid(gate_input)

        gate_forget = torch.sigmoid(gate_forget)

        gate_output = torch.sigmoid(gate_output)

        gate_pre_c = torch.tanh(gate_pre_c)

        gate_input_bar = torch.sigmoid(gate_input_bar)

        gate_forget_bar = torch.sigmoid(gate_forget_bar)

        gate_decay = F.softplus(gate_decay)
      

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime,if_predict=False):
        
        if not if_predict:
            if dtime.dim() < cell_i.dim():
                dtime = dtime.unsqueeze(cell_i.dim()-1).expand_as(cell_i)
        else:
            dtime = dtime.unsqueeze(0).unsqueeze(0)
            cell_bar_i = cell_bar_i.unsqueeze(-1)
            cell_i = cell_i.unsqueeze(-1)
            gate_decay = gate_decay.unsqueeze(-1)
            gate_output = gate_output.unsqueeze(-1)

        #Eq(7)
        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * dtime)
        
        #Eq(4b)
        hidden_t_ip1_minus = gate_output * torch.tanh(cell_t_ip1_minus)
        
        return cell_t_ip1_minus, hidden_t_ip1_minus

class HEARD(nn.Module):
    def __init__(self,config):
        super(HEARD, self).__init__()

        self.device = config["models"][config["active_model"]]["device"]
        self.batch_size = config["models"][config["active_model"]]["hyperparameters"]["batch_size"]
        self.hid_feats_RD = config["models"][config["active_model"]]["hyperparameters"]["hidden_size_RD"]
        self.hid_feats_HC = config["models"][config["active_model"]]["hyperparameters"]["hidden_size_HC"]
        self.in_feats_RD = config["models"][config["active_model"]]["hyperparameters"]["in_feats_RD"]
        self.in_feats_HC = config["models"][config["active_model"]]["hyperparameters"]["in_feats_HC"]
        self.sample_integral = config["models"][config["active_model"]]["hyperparameters"]["sample_integral"]
        self.sample_pred = config["models"][config["active_model"]]["hyperparameters"]["sample_pred"]
        beta_HC = config["models"][config["active_model"]]["hyperparameters"]["beta"]["HC"]
        beta_T = config["models"][config["active_model"]]["hyperparameters"]["beta"]["T"]
        beta_N = config["models"][config["active_model"]]["hyperparameters"]["beta"]["N"]
        fc_dp = config["models"][config["active_model"]]["hyperparameters"]["fc_dropout"]
        lstm_dp = config["models"][config["active_model"]]["hyperparameters"]["lstm_dropout"]

        self.HCLSTM = CTLSTMCell(self.hid_feats_HC,self.batch_size,lstm_dp,self.device)

        self.EmbeddingRD = nn.Linear(self.in_feats_RD,self.hid_feats_RD,bias=False).to(self.device)
        self.EmbeddingHC = nn.Linear(self.in_feats_HC,self.hid_feats_HC).to(self.device)
        nn.init.orthogonal_(self.EmbeddingRD.weight)
        nn.init.orthogonal_(self.EmbeddingHC.weight)

        self.fcRD=nn.Linear(self.hid_feats_RD,2).to(self.device)
        nn.init.orthogonal_(self.fcRD.weight)
        self.fcHC=nn.Linear(self.hid_feats_HC,1).to(self.device)
        nn.init.orthogonal_(self.fcHC.weight)
        self.fcN=nn.Linear(self.hid_feats_HC,2).to(self.device)
        nn.init.orthogonal_(self.fcN.weight)

        self.dropout = nn.Dropout(fc_dp)

        self.sigmoid = nn.Sigmoid()

        self.beta_HC = torch.Tensor([beta_HC]).to(self.device)
        self.beta_T = torch.Tensor([beta_T]).to(self.device)
        self.beta_N = torch.Tensor([beta_N]).to(self.device)
        
        self.num_layer = config["models"][config["active_model"]]["hyperparameters"]["lstm_layers"]
        self.layers = nn.ModuleList().to(self.device)
        for l in range(self.num_layer):
            self.layers.append(RDLSTMCell(self.hid_feats_RD,self.batch_size,lstm_dp,self.device))
        
    def forward(self, data,if_dp=True):
        
        h_d_RDs,c_d_RDs=[],[]
        for layer in self.layers:
            layer.init_states()
            h_d_RDs.append(layer.h_d)
            c_d_RDs.append(layer.c_d)

        self.HCLSTM.init_states()

        h_d_HC,c_d_HC,c_bar_HC = self.HCLSTM.h_d,self.HCLSTM.c_d,self.HCLSTM.c_bar
        num_reverse = torch.zeros(self.batch_size,1,dtype=torch.long).to(self.device)

        label_seqs,text_seqs_tensor, \
        time_seqs_tensor, seqs_length,_,_,posts_length,max_post_len,real_lens,post_indexes = self.__data_adapt__(data,if_dp)

        batch_length = time_seqs_tensor.size()[1]
        seqs_length = seqs_length.unsqueeze(1)

        h_list_RD, c_list_RD, pred_list_RD,prob_list_RD,h_list_RD_tmp = [], [], [], [],[]
        h_list_HC, c_list_HC, c_bar_list_HC, o_list_HC, delta_list_HC, reverse_list_HC, if_reverse_HC = [], [], [], [], [], [],[]

        for t in range(batch_length):

            text_input = text_seqs_tensor[:,t,:]
            text_seq_emb = self.EmbeddingRD(text_input)

            h_d_RD =  text_seq_emb
            for li,layer in enumerate(self.layers):
                h_d_RD,(_,c_d_RD) = layer(h_d_RD,h_d_RDs[li],c_d_RDs[li])
                h_d_RDs[li],c_d_RDs[li] = h_d_RD,c_d_RD

            if if_dp:
                pred_RD = self.fcRD(self.dropout(h_d_RD))
            else:
                pred_RD = self.fcRD(h_d_RD)
            pred_RD_prob = F.softmax(pred_RD, dim=1)
            label_RD = torch.argmax(pred_RD_prob, dim=1)
            h_list_RD.append(h_d_RD)
            c_list_RD.append(c_d_RD)
            prob_list_RD.append(pred_RD)

            if t == 0:
                if_reverse_t = torch.zeros_like(label_RD).long().unsqueeze(1)
            else:
                if_reverse_t = (label_RD != pred_list_RD[-1]).long().unsqueeze(1)

            pred_list_RD.append(label_RD)
            num_reverse = num_reverse+if_reverse_t
           
            reverse_list_HC.append(num_reverse)
            if_reverse_HC.append(if_reverse_t)
            num_reverse_n = num_reverse * 1.0
            
            reverse_seq_emb = self.EmbeddingHC(num_reverse_n)
            reverse_seq_emb = reverse_seq_emb.squeeze(1)
            c, c_bar_HC, delta_t,o_t = self.HCLSTM(reverse_seq_emb,h_d_HC,c_d_HC,c_bar_HC)
            c_d_HC, h_d_HC = self.HCLSTM.decay(c, c_bar_HC, delta_t, o_t, time_seqs_tensor[:,t])

            h_list_HC.append(c_d_HC)
            c_list_HC.append(c)
            c_bar_list_HC.append(c_bar_HC)
            o_list_HC.append(o_t)
            delta_list_HC.append(delta_t)

        h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverce_seq_HC\
             = torch.stack(h_list_HC),torch.stack(c_list_HC),torch.stack(c_bar_list_HC),\
                 torch.stack(o_list_HC),torch.stack(delta_list_HC),torch.stack(reverse_list_HC),torch.stack(if_reverse_HC)

        h_seq_RD, c_seq_RD, pred_seq_RD,prob_seq_RD = torch.stack(h_list_RD),torch.stack(c_list_RD),torch.stack(pred_list_RD),torch.stack(prob_list_RD)

        self.output_RD = (h_seq_RD, c_seq_RD, pred_seq_RD,prob_seq_RD)
        self.output_HC = (h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverce_seq_HC)

        return self.output_RD,self.output_HC,label_seqs

    def __data_adapt__(self, data,if_dp=True):
        label_seqs,target_seqs,\
        text_seqs_tensor,\
        post_since_start_seqs_tensor,time_seqs_tensor,\
        last_time_seqs, seqs_length,timstamp_seqs_tensor,posts_length,max_post_len,real_lens,_,_ = data
        
        if not if_dp:
            label_seqs, seqs_length, real_lens= label_seqs.repeat(self.batch_size,),seqs_length.repeat(self.batch_size,),real_lens.repeat(self.batch_size,)
            post_since_start_seqs_tensor = post_since_start_seqs_tensor.repeat(self.batch_size,1)
            if max_post_len:
                text_seqs_tensor = text_seqs_tensor.repeat(self.batch_size,1,1,1)
                posts_length = posts_length.repeat(self.batch_size,1)
            else:
                text_seqs_tensor = text_seqs_tensor.repeat(self.batch_size,1,1)

            time_seqs_tensor, timstamp_seqs_tensor,target_seqs = \
                time_seqs_tensor.repeat(self.batch_size,1),\
                timstamp_seqs_tensor.repeat(self.batch_size,1),target_seqs.repeat(self.batch_size,1)
                
        label_seqs,text_seqs_tensor, time_seqs_tensor,seqs_length,timstamp_seqs_tensor,target_seqs, real_lens,post_since_start_seqs_tensor = \
        label_seqs.to(self.device),text_seqs_tensor.to(self.device),time_seqs_tensor.to(self.device),\
        seqs_length.to(self.device),timstamp_seqs_tensor.to(self.device),target_seqs.to(self.device), real_lens.to(self.device),post_since_start_seqs_tensor.to(self.device)

        return label_seqs,text_seqs_tensor, time_seqs_tensor, seqs_length,timstamp_seqs_tensor,target_seqs,posts_length,max_post_len,real_lens,post_since_start_seqs_tensor

    def __label_convert__(self,label_seqs):
        targets_seqs = torch.zeros((self.batch_size,2),dtype=torch.float)
        for idx,target in enumerate(label_seqs):
            if target.item() == 1:
                targets_seqs[idx,:] = torch.FloatTensor([0,1])
            else:
                targets_seqs[idx,:] = torch.FloatTensor([1,0])
        targets_seqs = targets_seqs.to(self.device)
        return targets_seqs

    def compute_lambda(self,h_seq_HC):

        lambda_k  = F.softplus(self.fcHC(h_seq_HC.transpose(0, 1)))
        return lambda_k

    def compute_sim_lambda(self,timestamps,gates_states):

        c, c_bar_HC, o_t, delta_t= gates_states
        if not timestamps:
            time_diffs = torch.FloatTensor(np.float32(np.array(sorted(
                                np.random.exponential(
                                    scale=1.0,
                                    size=(self.sample_pred,)))))).to(self.device)
        else:
            diff_time = (timestamps[1] - timestamps[0])
            sample_time = diff_time * \
                        torch.rand([self.sample_integral], device=self.device).squeeze(0)
            interval_left_shift = (timestamps[0] + 1)
            time_diffs = sample_time / interval_left_shift
        c_p_HC, h_p_HC = self.HCLSTM.decay(c, c_bar_HC, delta_t, o_t, time_diffs,True)
        lambda_pred  = F.softplus(self.fcHC(h_p_HC.transpose(1, -1))).transpose(1, -1).squeeze(0)
        return lambda_pred,time_diffs

    def coumpute_loss_step(self,idx,step,f_step,seq_len,label_seqs,final_N,prob_seq_RD,
                            timstamp_seqs_tensor,
                            N_i,h_step,gates_states):

        c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC = gates_states
        c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC = c_seq_HC.transpose(0,1),c_bar_seq_HC.transpose(0,1),o_seq_HC.transpose(0,1),delta_seq_HC.transpose(0,1)

        step_loss,term1,term2,term3,term4,term5,log_likelihood_HC= [0.0]*7
        delta_N = np.Inf
        target_RD = label_seqs[idx]
        output_RD = prob_seq_RD[idx,step]

        if step == 0:
            term3 = F.binary_cross_entropy_with_logits(output_RD,target_RD)         
        else:

            gates_states_step = (c_seq_HC[idx,step,:],c_bar_seq_HC[idx,step,:],o_seq_HC[idx,step,:],delta_seq_HC[idx,step,:])
            timestamps = timstamp_seqs_tensor[idx, :step+1]

            lambda_inf_step,time_diffs = self.compute_sim_lambda(None,gates_states_step)

            cum_num = (torch.arange(time_diffs.size()[0]+1)[1:]*1.0).to(self.device)
            time_density_term2 = torch.exp((-1.0*torch.cumsum(lambda_inf_step,dim=1) / cum_num[None, :])*time_diffs[None,:])
            time_density = time_density_term2 * lambda_inf_step

            time_pred_step = torch.mean(time_diffs[None, :]*time_density,dim=1)*time_diffs[-1]
            time_true = timestamps[step].to(self.device)
            term1 = torch.sqrt((time_true - time_pred_step).abs()**2)

            delta_N_i = self.fcN(h_step)
            target_N_i = self.__label_convert__(N_i)[0:1,:]
            term2 = F.binary_cross_entropy_with_logits(delta_N_i,target_N_i)

            term3 = F.binary_cross_entropy_with_logits(output_RD,target_RD)
            
            term4 = -torch.log(1.0-(f_step)/(1.0*seq_len))

            delta_N = torch.sum(lambda_inf_step,dim=1)*time_diffs[-1] / self.sample_pred 
            term5 = torch.sqrt((delta_N - final_N).abs()**2)

        step_loss = self.beta_HC*(term1 + term2) + term3 + self.beta_T*term4 + self.beta_N*term5
        return step_loss,delta_N,term1,term2,term3,term4,term5
            
    def compute_log_likelihood(self,Batch_data,if_dp=True):

        label_seqs,_,_, seqs_length,timstamp_seqs_tensor,_,posts_length,max_post_len,real_lens,post_indexes = self.__data_adapt__(Batch_data,if_dp)
        target_seqs = self.__label_convert__(label_seqs)
        h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverse_seq_HC= self.output_HC
        prob_seq_RD = self.output_RD[3].transpose(0,1)
        reverse_seq_HC = reverse_seq_HC.transpose(0,1)
        if_reverse_seq_HC = if_reverse_seq_HC.transpose(0,1)
        h_seq_HC = h_seq_HC.transpose(0,1)
        delta_N_at_stops = []

        stop_points = torch.LongTensor([-1]*seqs_length.size()[0]).to(self.device)
        batch_loss = torch.zeros(seqs_length.size()[0]).to(self.device)
        for idx, seq_len in enumerate(seqs_length):
            seq_loss = []
            seq_delta = []
            delta_N_at_stop = None
            term1s,term2s,term3s,term4s,term5s = [],[],[],[],[]
            stop_points[idx] = seq_len-1
            for step in range(seq_len):
                if step == seq_len-1 and seq_len>1:
                    seq_loss.append(100.0)
                    break
                h_step = h_seq_HC[idx,step,:].unsqueeze(0)

                final_N = (reverse_seq_HC[idx,-1]-reverse_seq_HC[idx,step]).to(self.device)
                if seq_len>1:
                    N_i = (reverse_seq_HC[idx,step+1]-reverse_seq_HC[idx,step]).to(self.device)
                else:
                    N_i = None
                f_seq_len = real_lens[idx]
                f_step = post_indexes.cpu().data[idx,step]*1.0
                loss_step,delta_N,term1,term2,term3,term4,term5 = self.coumpute_loss_step(idx,step,f_step,f_seq_len,target_seqs,final_N,
                                                prob_seq_RD,timstamp_seqs_tensor,N_i,h_step,
                                                (c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC))
                seq_delta.append(delta_N)
                seq_loss.append(loss_step)
                term1s.append(term1)
                term2s.append(term2)
                term3s.append(term3)
                term4s.append(term4)
                term5s.append(term5)
                if (delta_N < 1.0):
                    stop_points[idx] = step
                    seq_loss.pop()
                    delta_N_at_stop = 0
                    break

            if delta_N_at_stop != None:
                delta_N_at_stops.append(delta_N_at_stop)
            else:
                delta_N_at_stops.append(1)
            batch_loss[idx] = (sum(seq_loss) / len(seq_loss))+term3s[-1]

        loss_batch = torch.mean(batch_loss)

        stop_preds = torch.LongTensor([-1]*seqs_length.size()[0]).to(self.device)
        for idx, stop_point in enumerate(stop_points):
            if stop_point.item() == -1:
                stop_points[idx] = seqs_length[idx]-1
            prob_stop_RD = prob_seq_RD[idx,stop_points[idx]]
            pred = torch.argmax(self.sigmoid(prob_stop_RD))
            stop_preds[idx] = pred
        return loss_batch,stop_points,stop_preds,delta_N_at_stops
