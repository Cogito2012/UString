#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse

from torch.utils.data import DataLoader

from src.GraphModels import VGRNN
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
ROOT_PATH = os.path.dirname(__file__)
 

def evaluation(all_pred, all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, np.mean(new_Time) * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print("Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 5))

    ### visualize

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()

    return AP



def test_all(testdata_loader, model, time=90, gpu_ids=[0]):
    
    all_pred = []
    all_labels = []
    loss_val, loss_kld_val, loss_acc_val = 0, 0, 0
    for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(testdata_loader):
        # ipdb.set_trace()
        with torch.no_grad():
            kld_loss, acc_loss, pred_scores, prior_means, _ = model(batch_xs, batch_ys, graph_edges, hidden_in=None, edge_weights=edge_weights)
        loss = kld_loss + p.loss_weight * acc_loss

        loss_val += loss.mean().item()
        loss_kld_val += kld_loss.mean().item()
        loss_acc_val += acc_loss.mean().item()

        num_frames = batch_xs.size()[1]
        assert num_frames >= time
        batch_size = batch_xs.size()[0]
        pred_frames = np.zeros((batch_size, time), dtype=np.float32)
        # run inference
        with torch.no_grad():
            for t in range(time):
                latent = prior_means[t]
                latent = latent.view(latent.size(0), -1)
                pred = model.module.predictor(latent) if len(gpu_ids)>1 else model.predictor(latent)  # 10 x 2
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
        # gather results and ground truth
        all_pred.append(pred_frames)
        label_onehot = batch_ys.cpu().numpy()
        label = np.reshape(label_onehot[:, 1], [batch_size,])
        all_labels.append(label)

    num_batch = i + 1
    loss_val /= num_batch
    loss_kld_val /= num_batch
    loss_acc_val /= num_batch

    # evaluation
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    print('----------------------------------')
    print("Starting evaluation...")
    AP = evaluation(all_pred, all_labels, total_time=time)
    print('----------------------------------')
    
    return loss_val, loss_kld_val, loss_acc_val, AP


def train_eval():
    # hyperparameters

    h_dim = p.hidden_dim  # 32
    z_dim = p.latent_dim  # 16
    x_dim = p.feature_dim  # 4096

    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    # model snapshots
    model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # tensorboard logging
    logs_dir = os.path.join(p.output_dir, p.dataset, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    # gpu options
    # ipdb.set_trace()
    gpu_ids = [int(id) for id in p.gpus.split(',')]

    # building model
    model = VGRNN(x_dim, h_dim, z_dim, p.num_rnn, conv=p.conv_type, bias=True, loss_func=p.loss_func)
    device = torch.device('cuda:%d'%(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.train() # set the model into training status
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, 'testing', toTensor=True, device=device)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        train_data = A3DDataset(data_path, 'train', toTensor=True, device=device)
        test_data = A3DDataset(data_path, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=int(p.batch_size / len(gpu_ids)), shuffle=True, drop_last=True)

    iter_cur = 0
    for k in range(p.epoch):
        for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(traindata_loader):
            # ipdb.set_trace()
            optimizer.zero_grad()
            kld_loss, acc_loss, _, _, hidden_st = model(batch_xs, batch_ys, graph_edges, edge_weights=edge_weights)

            loss = kld_loss + p.loss_weight * acc_loss
            loss.mean().backward()
            optimizer.step()
            
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            
            print('----------------------------------')
            print('epoch: %d, iter: %d' % (k, iter_cur))
            print('kld_loss = %.6f' % (kld_loss.mean().item()))
            print('%s_loss = %.6f (*factor=%.3f)' % (p.loss_func, acc_loss.mean().item(), p.loss_weight))
            print('loss = %.6f' % (loss.mean().item()))
            info = {'loss': loss.mean().item(),
                    'loss_kld': kld_loss.mean().item(),
                    'loss_%s'%(p.loss_func): acc_loss.mean().item()}
            logger.add_scalars("losses/train", info, iter_cur)
            
            iter_cur += 1
            # test and evaluate the model
            if iter_cur % p.test_iter == 0:
                model.eval()
                loss_val, loss_kld_val, loss_acc_val, AP = test_all(testdata_loader, model, time=90, gpu_ids=gpu_ids)
                model.train()
                # keep track of validation losses
                info = {'loss': loss_val, 'loss_kld': loss_kld_val, 'loss_acc': loss_acc_val}
                logger.add_scalars("losses/val", info, iter_cur)
                # logger.add_scalars('accuracy/valAP', AP, iter_cur)

        # save model
        model_file = os.path.join(model_dir, 'vgrnn_model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
        print('Model has been saved as: %s'%(model_file))

    logger.close()


def test_eval():
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    # result path
    result_dir = os.path.join(p.output_dir, p.dataset, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if p.visualize:
        vis_dir = os.path.join(result_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    # gpu options
    # ipdb.set_trace()
    gpu_ids = [int(id) for id in p.gpus.split(',')]

    # building model
    model = VGRNN(p.feature_dim, p.hidden_dim, p.latent_dim, p.num_rnn, conv=p.conv_type, bias=True)
    device = torch.device('cuda:%d'%(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # load the trained model weights
    assert os.path.exists(p.model_file)
    checkpoint = torch.load(p.model_file)
    model.load_state_dict(checkpoint['model'])
    print('Model weights are loaded.')
    model.eval()

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        phase = 'testing'
        test_data = DADDataset(data_path, phase, toTensor=True, device=device, vis=True)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        phase = 'test'
        test_data = A3DDataset(data_path, phase, toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)
    num_samples = len(test_data)
    print("Number of testing samples: %d"%(num_samples))

    all_pred, all_labels = [], []
    print('----------------------------------')
    print("Starting inference...")
    for i, (batch_xs, batch_ys, graph_edges, edge_weights, toa, detections, video_ids) in enumerate(testdata_loader):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _, _, pred_scores, prior_means, _ = model(batch_xs, batch_ys, graph_edges, hidden_in=None, edge_weights=edge_weights)

        num_frames = batch_xs.size()[1]
        batch_size = batch_xs.size()[0]
        pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
        # run inference
        with torch.no_grad():
            for t in range(90):
                latent = prior_means[t]
                latent = latent.view(latent.size(0), -1)
                pred = model.predictor(latent)  # 10 x 2
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
        label_onehot = batch_ys.cpu().numpy()
        labels = np.reshape(label_onehot[:, 1], [batch_size,])

        torch.cuda.synchronize()
        time_ellapsed = (time.time()-start) / batch_size

        # visualize
        if p.visualize:
            vis_results(pred_frames, toa, labels, vis_dir)
            print('Batch %d visualized.'%(i))
            continue
        # evaluation
        print("Batch %d processed. Time=%.3f s per video."%(i, time_ellapsed))
        all_pred.append(pred_frames)
        all_labels.append(labels)

    # evaluation
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    print('----------------------------------')
    print("Starting evaluation...")
    AP = evaluation(all_pred, all_labels, total_time=90)
    print('----------------------------------')



def vis_results(pred_frames, toa, labels, vis_dir):
    for n in range(batch_size):
        if labels[n] == 1:
            # plot the probability predictions
            plt.figure(figsize=(14, 5))
            plt.plot(pred_frames[n, :], linewidth=3.0)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.xlim(0, 100)
            plt.xlabel('Frame (FPS=20)')
            plt.grid(True)
            plt.tight_layout()
            plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            plt.savefig(os.path.join(vis_dir, video_ids[n] + '.png'))
            # # video/frames files
            # visualize_on_video(p.data_path, dataset=p.dataset, format='gif')
            # pos_neg = 'positive' if labels[1] > 0 else 'negative'
            # video_path = os.path.join(data_path, 'videos', phase, pos_neg, video_ids[n] + '.mp4')
            # assert os.path.exists(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['a3d', 'dad'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=200,
                        help='The number of training epoches. Default: 200')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size in training process. Default: 16')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='VGG16', choices=['VGG16', 'ResNet152', 'C3D', 'I3D', 'TSN'],
                        help='The name of feature embedding methods. Default: VGG16')
    parser.add_argument('--conv_type', type=str, default='GCN', choices=['GCN', 'SAGE', 'GIN'],
                        help='The types of graph convolutional neural networks. Default: GCN')
    parser.add_argument('--test_iter', type=int, default=20,
                        help='The number of iteration to perform a evaluation process.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='The dimension of hidden states in RNN. Default: 128')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='The dimension of latent space. Default: 64')
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help='The dimension of node features in graph. Default: 4096')
    parser.add_argument('--loss_func', type=str, default='exp', choices=['exp', 'bernoulli'],
                        help='The functions of loss for accident prediction. Default: exp')
    parser.add_argument('--loss_weight', type=float, default=0.1,
                        help='The weighting factor of the two loss functions. Default: 0.1')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'],
                        help='The state of running the model. Default: train')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--model_file', type=str, default='./output/dad/snapshot/vgrnn_model_90.pth',
                        help='The trained VGRNN model file for demo test only.')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='The directory of src need to save in the training.')

    p = parser.parse_args()
    if p.phase == 'test':
        test_eval()
    else:
        train_eval()
