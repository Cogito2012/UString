from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse

from torch.utils.data import DataLoader
from src.GraphModels import BayesGCRNN
from src.eval_tools import evaluation, print_results, vis_results
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['a3d', 'dad', 'crash'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=64,
                        help='The number of iteration to perform a evaluation process. Default: 64')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The dimension of hidden states in RNN. Default: 256')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--use_mask', action='store_true',
                        help='Apply masking on input features. Default: False')
    parser.add_argument('--remove_saa', action='store_true',
                        help='Use self-attention aggregation layer. Default: False')
    parser.add_argument('--uncertainty_ranking', action='store_true',
                        help='Use uncertainty ranking loss. Default: False')
    parser.add_argument('--loss_alpha', type=float, default=0.001,
                        help='The weighting factor of posterior and prior losses. Default: 1e-3')
    parser.add_argument('--loss_beta', type=float, default=10,
                        help='The weighting factor of auxiliary loss. Default: 10')
    parser.add_argument('--loss_yita', type=float, default=10,
                        help='The weighting factor of uncertainty ranking loss. Default: 10')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'],
                        help='The state of running the model. Default: train')
    parser.add_argument('--evaluate_all', action='store_true',
                        help='Whether to evaluate models of all epoches. Default: False')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--model_file', type=str, default='./output_debug/bayes_gcrnn/vgg16/dad/snapshot/gcrnn_model_90.pth',
                        help='The trained GCRNN model file for demo test only.')
    parser.add_argument('--output_dir', type=str, default='./output_debug/bayes_gcrnn/vgg16',
                        help='The directory of src need to save in the training.')
    return parser.parse_args()


def test_flops():
    ### --- CONFIG PATH ---
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from src.DataLoader import DADDataset
    test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)
    num_samples = len(test_data)
    print("Number of testing samples: %d"%(num_samples))
    
    # building model
    model = BayesGCRNN(test_data.dim_feature, p.hidden_dim, p.latent_dim, 
                       n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps, 
                       with_saa=(not p.remove_saa), uncertain_ranking=p.uncertainty_ranking, use_mask=p.use_mask)

    # model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)

    # if multiGPU:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        for batch_xs, batch_ys, graph_edges, edge_weights, batch_toas, detections, video_ids in testdata_loader:
            # run forward inference
            # losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas, graph_edges, 
            #         hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False, eval_uncertain=True)
            from thop import profile
            hidden_in=None
            edge_weights=edge_weights
            npass=10
            nbatch=len(testdata_loader)
            testing=False
            eval_uncertain=True
            flops, params = profile(model, inputs=(batch_xs, batch_ys, batch_toas, graph_edges, 
                    hidden_in, edge_weights, npass, nbatch, testing, eval_uncertain,))
            break

if __name__ == "__main__":
    p = get_parser()
    p.uncertainty_ranking = True
    p.gpus = "0"
    test_flops()