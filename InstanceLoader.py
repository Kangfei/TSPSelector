import numpy as np
import os

import torch
import pickle
from torch.utils.data.dataset import Dataset
from transform import default_val_transforms
from scipy import sparse

'''
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.decomposition import PCA

def edge_sampler(edge_index, edge_attr, prob = 0.5):
    edge_num = edge_attr.shape[0]
    p = 1.0 / edge_attr
    p = p / np.sum(p)
    sample_size = int(edge_num * prob)
    idx = np.random.choice(edge_num, size= sample_size, replace = False, p = p)
    edge_attr = edge_attr[idx]
    # edge attribute normalization
    #edge_attr = edge_attr / np.sum(edge_attr)
    return edge_index[:, idx], edge_attr


class GeoInstanceDataset(GeoDataset):
    def __init__(self, num_node_feats, path, labels):
        """
        path: the  directory of the input of instances (edges and weight matrix)
        labels: dictionary of instance -> label
        """
        self.num_node_feats = num_node_feats
        self.path = path
        self.labels = labels
        self.file_list = []
        self.label_map = {'eax': 0,
                          'eax.restart': 1,
                          'lkh': 2,
                          'lkh.restart': 3,
                          'maos': 4}
        for key in labels.keys():
            dataset = key.strip().split('_')[0]
            instance_id = key.strip().split('_')[1]
            if dataset == 'morphed':
                node_num = instance_id.strip().split('-')[0]
                tmp1, tmp2 = instance_id.strip().split('---')[0], instance_id.strip().split('---')[1]
                instance_id = node_num + "---" + tmp1 + '.tsp---' + tmp2 + '.tsp'
            #full_instance_dir = os.path.join(path, dataset, instance_id) + '.pickle'
            # load the normalized tsp problem
            full_instance_dir = os.path.join(path, dataset, instance_id) + '_norm.pickle'
            self.file_list.append((key, full_instance_dir))
        self.num = len(self.file_list)

    def __getitem__(self, index):
        key, full_instance_dir = self.file_list[index]
        with open(full_instance_dir, 'rb') as in_file:
            data = pickle.load(in_file)
            x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
            # x : (N, 2)

            edge_index, edge_attr = edge_sampler(edge_index, edge_attr)

            # TODO: normalize x

            #x = x / x.max(axis = 0)
            #x = torch.zeros((x.shape[0], self.num_node_feats), dtype=torch.float)

            label = np.zeros(shape = (1), dtype = np.int)
            idx = self.label_map[self.labels[key]]
            label[0] = idx
            label = torch.LongTensor(label)
            x = torch.FloatTensor(x)
            edge_index = torch.LongTensor(edge_index)
            edge_attr = torch.FloatTensor(edge_attr)

            #print(x.shape, edge_index.shape, edge_attr.shape)
            data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = label)
            return data

    def __len__(self):
        return self.num

    def _download(self):
        pass

    def _process(self):
        pass



class InstanceDataset(Dataset):
    def __init__(self, num_node_feats, path, labels):
        """
        path: the  directory of the input of instances (edges and weight matrix)
        labels: dictionary of instance -> label
        """
        self.num_node_feats = num_node_feats
        self.path = path
        self.labels = labels
        self.file_list = []
        self.label_map = {'eax': 0,
                          'eax.restart': 1,
                          'lkh': 2,
                          'lkh.restart': 3,
                          'maos': 4}
        for key in labels.keys():
            dataset = key.strip().split('_')[0]
            instance_id = key.strip().split('_')[1]
            if dataset == 'morphed':
                node_num = instance_id.strip().split('-')[0]
                tmp1, tmp2 = instance_id.strip().split('---')[0], instance_id.strip().split('---')[1]
                instance_id = node_num + "---" + tmp1 + '.tsp---' + tmp2 + '.tsp'
            #full_instance_dir = os.path.join(path, dataset, instance_id) + '.pickle'
            # load the normalized tsp problem
            full_instance_dir = os.path.join(path, dataset, instance_id) + '_norm.pickle'
            self.file_list.append((key, full_instance_dir))
        self.num = len(self.file_list)

    def __getitem__(self, index):
        key, full_instance_dir = self.file_list[index]
        with open(full_instance_dir, 'rb') as in_file:
            data = pickle.load(in_file)
            A = data['adj'] # A： （N, N） distance matrix
            x = data['x']
            A = self.hilbert_matrix_reorder(x,  A)

            label = np.zeros(shape = (1), dtype = np.int)
            idx = self.label_map[self.labels[key]]
            label[0] = idx
            label = torch.LongTensor(label)
            A = torch.FloatTensor(A)
            #padding to fixed size
            image = self.padding(A)
            #image = self.toTSPImage(x)

            # repeat to 3 channels
            # image = image.repeat(1, 3)
            # image = image.view((3, image.shape[0], image.shape[0]))

            # for 1 channel
            image = image.view((1, image.shape[0], image.shape[0]))
            #print(image.shape)
            return image, label

    def padding(self, A):
        delta_width = 2000 - A.shape[0]
        delta_height = 2000 - A.shape[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, delta_width - pad_width, pad_height, delta_height - pad_height)
        m = torch.nn.ZeroPad2d(padding=padding)
        return m(A)

    def matrix_reorder(self, x, A):
        N = A.shape[0]
        pca = PCA(n_components=1)
        x = pca.fit_transform(x).squeeze()
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        for i in range(N):
            A[:, i] = A[ranks, i]
        for i in range(N):
            A[i, :] = A[i, ranks]
        return A

    def hilbert_matrix_reorder(self, x, A):
        N = A.shape[0]
        x_norm = x * 1000
        x_norm = x_norm.astype(int)
        hilbert_curve = HilbertCurve(10, 2)
        temp = np.zeros(shape=(N))
        for i in range(N):
            temp[i] = hilbert_curve.distance_from_coordinates([x_norm[i][0], x_norm[i][1]])

        temp = temp.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))
        for i in range(N):
            A[:, i] = A[ranks, i]
        for i in range(N):
            A[i, :] = A[i, ranks]
        return A

    def toTSPImage(self, x, num_grid = 128):
        image = torch.zeros((num_grid, num_grid), dtype=torch.float)
        for i in range(x.shape[0]):
            idx_x = int(x[i][0] * num_grid) - 1
            idx_y = int(x[i][1] * num_grid) - 1

            image[idx_x][idx_y] = image[idx_x][idx_y] + 1.0
        return image

    def __len__(self):
        return self.num
'''

'''
Dataset of TSP image with image rotation and flip data argumentation
'''
class AugmentInstanceDataset(Dataset):
    def __init__(self, num_node_feats, path, labels):
        """
        path: the  directory of the input of instances (edges and weight matrix)
        labels: dictionary of instance -> label
        """
        self.num_node_feats = num_node_feats
        self.path = path
        self.labels = labels
        self.file_list = []
        self.label_map = {'eax': 0,
                          'eax.restart': 1,
                          'lkh': 2,
                          'lkh.restart': 3,
                          'maos': 4}

        for key in labels.keys():
            dataset = key.strip().split('_')[0]
            instance_id = key.strip().split('_')[1]
            if dataset == 'morphed':
                node_num = instance_id.strip().split('-')[0]
                tmp1, tmp2 = instance_id.strip().split('---')[0], instance_id.strip().split('---')[1]
                instance_id = node_num + "---" + tmp1 + '.tsp---' + tmp2 + '.tsp'
            #full_instance_dir = os.path.join(path, dataset, instance_id) + '.pickle'
            # load the normalized rotated tsp image
            for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
                for num_grid in [256]:
                    full_instance_dir = os.path.join(path, dataset, instance_id) + '_{0}_{1}_image.pickle'.format(num_grid, angle)
                    self.file_list.append((key, full_instance_dir))
            # load the normalized flipped tsp image
            full_instance_dir = os.path.join(path, dataset, instance_id) + '_{0}_flip0_image.pickle'.format(num_grid)
            self.file_list.append((key, full_instance_dir))
            full_instance_dir = os.path.join(path, dataset, instance_id) + '_{0}_flip1_image.pickle'.format(num_grid)
            self.file_list.append((key, full_instance_dir))
        self.num = len(self.file_list)

    def __getitem__(self, index):
        key, full_instance_dir = self.file_list[index]
        with open(full_instance_dir, 'rb') as in_file:
            data = pickle.load(in_file)
            A = data['adj'] # A： （N, N） distance matrix
            A = A.todense()

            label = np.zeros(shape = (1), dtype = np.int)
            idx = self.label_map[self.labels[key]]
            label[0] = idx
            label = torch.LongTensor(label)
            image = torch.FloatTensor(A)
            #image = self.padding(image)

            # repeat to 3 channels
            image = image.repeat(1, 3)
            image = image.view((3, image.shape[0], image.shape[0]))

            # for 1 channel
            #image = image.view((1, image.shape[0], image.shape[0]))

            #print(image.shape)
            return image, label

    def padding(self, A):
        delta_width = 256 - A.shape[0]
        delta_height = 256 - A.shape[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, delta_width - pad_width, pad_height, delta_height - pad_height)
        m = torch.nn.ZeroPad2d(padding=padding)
        return m(A)

    def __len__(self):
        return self.num



class ArgumentDataset(Dataset):
    def __init__(self, args, path, labels, transform = default_val_transforms):
        self.path = path
        self.labels = labels
        self.label_type = args.loss_type
        self.transform = transform
        self.all_coordinates = []
        self.label_map = {'GA-EAX': 0,
                          'GA-EAX-restart': 1,
                          'LKH': 2,
                          'LKH-restart': 3,
                          'LKH-crossover': 4,
                          'MAOS': 5}
        for key in labels.keys():
            dataset = key.strip().split('/')[2]
            instance_id = key.strip().split('/')[3]

            full_instance_dir = os.path.join(path, dataset, instance_id) + '.coo.pickle'
            with open(full_instance_dir, 'rb') as in_file:
                data = pickle.load(in_file)
                x = data['x']
                self.all_coordinates.append((key, x))
                in_file.close()
        self.num = len(self.all_coordinates)

    def __getitem__(self, index):
        key, x = self.all_coordinates[index]

        algorithm_to_median = self.labels[key][1]

        run_time = self.to_time_tensor(algorithm_to_median)

        # generate the label
        if self.label_type == 'nll':
            label = self.to_nll_label_tensor(key)
            label = torch.LongTensor(label)
        elif self.label_type == 'sce':
            label = self.to_sce_label_tensor(run_time)
            label = torch.FloatTensor(label)
        elif self.label_type == 'bce':
            label = self.to_bce_label_tensor(run_time)
            label = torch.FloatTensor(label)
        elif self.label_type == 'mse':
            label = self.to_mse_label_tensor(run_time)
            label = torch.FloatTensor(label)

        # generate the weights
        weights = self.to_weights_tensor(run_time)


        # generate the image
        image = self.transform(x)
        # repeat to 3 channels
        image = image.repeat(1, 3)
        image = image.view((3, image.shape[0], image.shape[0]))

        return image, label, weights, run_time


    def __len__(self):
        return self.num

    def to_time_tensor(self, algorithm_to_median):
        run_time = np.zeros(shape=(len(self.label_map)), dtype=np.float)
        for algorithm, value in algorithm_to_median.items():
            run_time[self.label_map[algorithm]] = value
        return run_time

    def to_sce_label_tensor(self, run_time, exp = 2.0):
        '''
        run_time: numpy array of running time
        '''

        baseline = run_time[np.argsort(run_time)[-3]]
        #print("baseline : {}".format(baseline))
        mask = np.where(run_time > baseline, 0, 1)
        label = 1.0 / np.power(run_time, exp)
        label = np.multiply(label, mask)
        # normalize to probability
        label = label / np.sum(label)

        return label


    def to_bce_label_tensor(self, run_time, decay_factor = 0.9):
        order = run_time.argsort()
        rank = order.argsort()
        mask = np.where(rank > 2, 0, 1)
        label = np.power(decay_factor, rank)
        label = np.multiply(label, mask)
        return label


    def to_mse_label_tensor(self, run_time):
        label = self.run_time_normalize(run_time)
        return label

    def to_nll_label_tensor(self, key):
        # labels for nll loss
        label = np.zeros(shape=(1), dtype=np.int)
        idx = self.label_map[self.labels[key][0]]
        label[0] = idx
        return label


    def run_time_normalize(self, run_time):
        min_time, max_time = np.min(run_time), np.max(run_time)
        res = run_time - min_time / (max_time - min_time)
        return res

    def to_weights_tensor(self, run_time, exp = 0.9):
        weights = np.power(run_time, exp)
        return weights