"""Traffic data loading, support-only normalization, and window construction.

Portions of this file are adapted from TPB:
https://github.com/zhyliu00/TPB

The original TPB code is released under the MIT License.
See third_party/TPB_LICENSE for the original license notice.
"""

import random

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from utils import generate_dataset, get_normalized_adj

random.seed(7)


class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo


class traffic_dataset(Dataset):
    def __init__(self, data_args, task_args, data_list=None, stage='source', test_data='metr-la', add_target=True, target_days=3, norm_scope='full'):
        super(traffic_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.predefined_data_list = data_list
        self.norm_scope = norm_scope
        self.load_data(stage, test_data)

        print("[INFO] Dataset init finished!")


    # according to the stage, output x_list and y_list, both of them are dict
    def load_data(self, stage, test_data):
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}
        self.batchnum_list = {}

        data_keys = np.array(self.data_args['data_keys'])
        if(self.predefined_data_list != None):
            data_keys = self.predefined_data_list
            if(self.add_target):
                data_keys += [self.test_data]

        if stage == 'source' or stage == 'pretrain' or self.stage == 'cluster' or self.stage == 'source_train':
            self.data_list = data_keys
        elif stage == 'target' or stage == 'target_maml':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        else:
            print("stage is : {}".format(stage))
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))


        for dataset_name in self.data_list:
            print("dataset_name : {}".format(dataset_name))
            A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
            edge_index, edge_attr, node_feature = self.get_attr_func(
            self.data_args[dataset_name]['adjacency_matrix_path']
            )

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            X = np.load(self.data_args[dataset_name]['dataset_path'])
            # Convert [T, N, F] to [N, F, T]; retain speed and the last time feature.

            X = X.transpose((1, 2, 0))
            X = torch.tensor(X,dtype=torch.double)

            # [N, 2, L]
            X = torch.cat((X[:,0, :].unsqueeze(1), X[:,-1,:].unsqueeze(1)), dim = 1)

            # Interpolation. Chengdu and Shenzhen interpolated to 5min level.
            interp = False
            if(dataset_name in ['chengdu_m','shenzhen']):
                interp = True

            if(interp):
                interp_X = torch.nn.functional.interpolate(X, size = 2 * X.shape[-1] - 1,mode='linear',align_corners=True)
                interp_X = torch.cat((interp_X[:,:,:1],interp_X),dim=-1)
                interp_X[:,1,0] = ((interp_X[:,1,1] - 1) + 2016 ) % 2016 # 2016 is the week slot
                X = interp_X

            X = X.numpy()
            # mean and std 
            X[:,0,:] = X[:,0,:].astype(np.float64)
            norm_values = X[:, 0, :]

            if self.norm_scope == 'target_support' and dataset_name == self.test_data:
                norm_values = X[:, 0, :288 * self.target_days]

            means = np.expand_dims(np.mean(norm_values), 0)
            stds = np.expand_dims(np.std(norm_values), 0)
            stds = np.maximum(stds, 1e-6)

            self.means_list[dataset_name], self.stds_list[dataset_name] = means, stds
            X[:, 0, :] = (X[:, 0, :] - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)

            # [N, 2, L] and 0 is normalized
            if stage == 'source' or stage == 'dann' or stage == 'pretrain' or stage == 'source_train':
                if(dataset_name == self.test_data):
                    X = X[:, :, :288 * self.target_days]
                else:
                    X = X

            # target, small sample to finetune, 288 = 24 * 12 is one day data.
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]

            # test, choose rest of data
            elif stage == 'test':
                X = X[:, :, 288 * self.target_days:]

            # X : [N, 2, L]

            if(self.stage == 'cluster'):
                self.x_list[dataset_name] = X
                self.y_list[dataset_name] = []
                continue

            his_num = self.task_args['his_num']
            pred_num = self.task_args['pred_num']


            if(self.stage == 'pretrain'):
                # Pretraining windows advance by three hours at five-minute resolution.
                inter_step = 12 * 3 
            elif(self.stage == 'source_train'):
                inter_step = 12 * 24
            else:
                inter_step = 12
            x_inputs, y_outputs = generate_dataset(X, his_num, pred_num, means, stds, inter_step)
            print('{} : x shape : {}, y shape : {}'.format(dataset_name, x_inputs.shape, y_outputs.shape))
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs


        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            self.pretrain_batchnum = 0
            batch_size = self.task_args['batch_size']
            for dataset_name in self.data_list:
                this_data_total_batches = int(self.x_list[dataset_name].shape[0] // batch_size)
                self.batchnum_list[dataset_name] = this_data_total_batches
                self.pretrain_batchnum += this_data_total_batches

            self.pretrain_which_data = torch.zeros((self.pretrain_batchnum))
            self.pretrain_which_pos = torch.zeros((self.pretrain_batchnum))
            cur = 0
            for idx, dataset_name in enumerate(self.data_list):
                self.pretrain_which_data[cur : cur + self.batchnum_list[dataset_name]] = int(idx)
                self.pretrain_which_pos[cur : cur + self.batchnum_list[dataset_name]] = torch.arange(cur, cur + self.batchnum_list[dataset_name]) - cur
                cur += self.batchnum_list[dataset_name]
            self.random_permutation =torch.randperm(self.pretrain_batchnum)

    def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature


    def __getitem__(self, index):
        """Return a PyG data batch and its normalized adjacency matrix.

        data.x has shape [B, N, his_num, F]; data.y contains raw speed targets
        with shape [B, N, pred_num]. The data object includes node count,
        edge indices, city name, and normalization statistics.
        """

        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            # need query *batch_size* continuous batches
            idx = self.random_permutation[index]
            select_dataset = self.data_list[self.pretrain_which_data[idx].detach().cpu().numpy().astype(int)]
            pos = self.pretrain_which_pos[idx].detach().cpu().numpy().astype(int)
            batch_size = self.task_args['batch_size']
            indices = torch.tensor(list(range(pos,pos+batch_size)))
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        # if 'source', randomly choose a city and random choose a batch
        elif (self.stage == 'source'):
            select_dataset = random.choice(self.data_list)
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        # if 'target_maml', choose the first city and randomly choose a batch
        else:
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        x_data = x_data.float()
        y_data = y_data.float()
        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data,means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]

        return data_i, A_wave


    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        if self.stage == 'pretrain' or self.stage == 'source_train':
            return self.pretrain_batchnum
        if self.stage == 'target_maml' or self.stage == 'test':
            return int(self.x_list[self.data_list[0]].shape[0] //  self.task_args['batch_size'])
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length
