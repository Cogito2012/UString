from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import networkx
import itertools


class DADDataset(Dataset):
    def __init__(self, data_path, feature, phase='training', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = os.path.join(data_path, feature + '_features')
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s"%(filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 100 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 100 x 19 x 6
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        if labels[1] > 0:
            toa = [90.0]
        else:
            toa = [self.n_frames + 1]
        
        graph_edges, edge_weights = generate_st_graph(detections)

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  100 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            graph_edges = torch.Tensor(graph_edges).long().to(self.device)
            edge_weights = torch.Tensor(edge_weights).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            video_id = str(data['ID'])[5:11]  # e.g.: b001_000490_*
            return features, labels, graph_edges, edge_weights, toa, detections, video_id
        else:
            return features, labels, graph_edges, edge_weights, toa


class A3DDataset(Dataset):
    def __init__(self, data_path, feature, phase='train', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        self.files_list, self.labels_list = self.read_datalist(data_path, phase)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def read_datalist(self, data_path, phase):
        # load training set
        list_file = os.path.join(data_path, self.feature + '_features', '%s.txt' % (phase))
        assert os.path.exists(list_file), "file not exists: %s"%(list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = [], []
        for line in fid.readlines():
            filename, label = line.rstrip().split(' ')
            data_files.append(filename)
            data_labels.append(int(label))
        fid.close()

        return data_files, data_labels

    def get_toa(self, clip_id):
        # handle clip id like "uXXC8uQHCoc_000011_0" which should be "uXXC8uQHCoc_000011"
        clip_id = clip_id if len(clip_id.split('_')[-1]) > 1 else clip_id[:-2]
        label_file = os.path.join(self.data_path, 'frame_labels', clip_id + '.txt')
        assert os.path.exists(label_file)
        f = open(label_file, 'r')
        label_all = []
        for line in f.readlines():
            label = int(line.rstrip().split(' ')[1])
            label_all.append(label)
        f.close()
        label_all = np.array(label_all, dtype=np.int32)
        toa = np.where(label_all == 1)[0][0]
        toa = max(1, toa)  # time-of-accident should not be equal to zero
        return toa

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.feature + '_features', self.files_list[index])
        assert os.path.exists(data_file), "file not exists: %s"%(data_file)
        data = np.load(data_file)
        features = data['features']
        label = self.labels_list[index]
        label_onehot = np.array([0, 1]) if label > 0 else np.array([1, 0])
        # get time of accident
        file_id = self.files_list[index].split('/')[1].split('.npz')[0]
        if label > 0:
            toa = [self.get_toa(file_id)]
        else:
            toa = [self.n_frames + 1]

        # construct graph
        attr = 'positive' if label > 0 else 'negative'
        dets_file = os.path.join(self.data_path, 'detections', attr, file_id + '.pkl')
        assert os.path.exists(dets_file), "file not exists: %s"%(dets_file)
        with open(dets_file, 'rb') as f:
            detections = pickle.load(f)
            detections = np.array(detections)  # 100 x 19 x 6
            graph_edges, edge_weights = generate_st_graph(detections)
        f.close()

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)          #  100 x 20 x 4096
            label_onehot = torch.Tensor(label_onehot).to(self.device)  #  2
            graph_edges = torch.Tensor(graph_edges).long().to(self.device)
            edge_weights = torch.Tensor(edge_weights).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            # file_id = file_id if len(file_id.split('_')[-1]) > 1 else file_id[:-2]
            # video_path = os.path.join(self.data_path, 'video_frames', file_id, 'images')
            # assert os.path.exists(video_path), video_path
            return features, label_onehot, graph_edges, edge_weights, toa, detections, file_id
        else:
            return features, label_onehot, graph_edges, edge_weights, toa


class CrashDataset(Dataset):
    def __init__(self, data_path, feature, phase='train', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 50
        self.n_obj = 19
        self.fps = 10.0
        self.dim_feature = self.get_feature_dim(feature)
        self.files_list, self.labels_list = self.read_datalist(data_path, phase)
        self.toa_dict = self.get_toa_all(data_path)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def read_datalist(self, data_path, phase):
        # load training set
        list_file = os.path.join(data_path, self.feature + '_features', '%s.txt' % (phase))
        assert os.path.exists(list_file), "file not exists: %s"%(list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = [], []
        for line in fid.readlines():
            filename, label = line.rstrip().split(' ')
            data_files.append(filename)
            data_labels.append(int(label))
        fid.close()
        return data_files, data_labels

    def get_toa_all(self, data_path):
        toa_dict = {}
        annofile = os.path.join(data_path, 'videos', 'Crash-1500.txt')
        annoData = self.read_anno_file(annofile)
        for anno in annoData:
            labels = np.array(anno['label'], dtype=np.int)
            toa = np.where(labels == 1)[0][0]
            toa = min(max(1, toa), self.n_frames-1) 
            toa_dict[anno['vid']] = toa
        return toa_dict

    def read_anno_file(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist! %s"%(anno_file)
        result = []
        with open(anno_file, 'r') as f:
            for line in f.readlines():
                items = {}
                items['vid'] = line.strip().split(',[')[0]
                labels = line.strip().split(',[')[1].split('],')[0]
                items['label'] = [int(val) for val in labels.split(',')]
                assert sum(items['label']) > 0, 'invalid accident annotation!'
                others = line.strip().split(',[')[1].split('],')[1].split(',')
                items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
                result.append(items)
        f.close()
        return result

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.feature + '_features', self.files_list[index])
        assert os.path.exists(data_file), "file not exists: %s"%(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 50 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 50 x 19 x 6
            vid = str(data['ID'])
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        if labels[1] > 0:
            toa = [self.toa_dict[vid]]
        else:
            toa = [self.n_frames + 1]

        graph_edges, edge_weights = generate_st_graph(detections)

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  50 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            graph_edges = torch.Tensor(graph_edges).long().to(self.device)
            edge_weights = torch.Tensor(edge_weights).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            return features, labels, graph_edges, edge_weights, toa, detections, vid
        else:
            return features, labels, graph_edges, edge_weights, toa


def generate_st_graph(detections):
    # create graph edges
    num_frames, num_boxes = detections.shape[:2]
    num_edges = int(num_boxes * (num_boxes - 1) / 2)
    graph_edges = []
    edge_weights = np.zeros((num_frames, num_edges), dtype=np.float32)
    for i in range(num_frames):
        # generate graph edges (fully-connected)
        edge = generate_graph_from_list(range(num_boxes))
        graph_edges.append(np.transpose(np.stack(edge).astype(np.int32)))  # 2 x 171
        # compute the edge weights by distance
        edge_weights[i] = compute_graph_edge_weights(detections[i, :, :4], edge)  # 171,

    return graph_edges, edge_weights

       
def generate_graph_from_list(L, create_using=None):
   G = networkx.empty_graph(len(L),create_using)
   if len(L)>1:
       if G.is_directed():
            edges = itertools.permutations(L,2)
       else:
            edges = itertools.combinations(L,2)
       G.add_edges_from(edges)
   graph_edges = list(G.edges())

   return graph_edges


def compute_graph_edge_weights(boxes, edges):
    """
    :param: boxes: (19, 4)
    :param: edges: (171, 2)
    :return: weights: (171,)
    """
    N = boxes.shape[0]
    assert len(edges) == N * (N-1) / 2
    weights = np.ones((len(edges),), dtype=np.float32)
    for i, edge in enumerate(edges):
        c1 = [0.5 * (boxes[edge[0], 0] + boxes[edge[0], 2]),
              0.5 * (boxes[edge[0], 1] + boxes[edge[0], 3])]
        c2 = [0.5 * (boxes[edge[1], 0] + boxes[edge[1], 2]),
              0.5 * (boxes[edge[1], 1] + boxes[edge[1], 3])]
        d = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
        weights[i] = np.exp(-d)
    # normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)  # N*(N-1)/2,
    else:
        weights = np.ones((len(edges),), dtype=np.float32)

    return weights


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['a3d', 'dad', 'crash'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    p = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)
    elif p.dataset == 'a3d':
        train_data = A3DDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = A3DDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    elif p.dataset == 'crash':
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    for e in range(2):
        print('Epoch: %d'%(e))
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('label dim:', batch_ys.size())
                print('graph edges dim:', graph_edges.size())
                print('edge weights dim:', edge_weights.size())
                print('time of accidents dim:', batch_toas.size())

    for e in range(2):
        print('Epoch: %d'%(e))
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas, detections, video_ids) in \
            tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('label dim:', batch_ys.size())
                print('graph edges dim:', graph_edges.size())
                print('edge weights dim:', edge_weights.size())
                print('time of accidents dim:', batch_toas.size())

