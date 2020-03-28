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

from src.GraphModels import BayesGCRNN
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
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
    for Th in np.arange(np.min(all_pred), 1.0, 0.001):
        if length is not None and Th <= 0:
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
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    mTTA = np.mean(new_Time)
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))]
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80 * 5))

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='TTA Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()

    if mTTA == np.nan:
        mTTA = 0
    if TTA_R80 == np.nan:
        TTA_R80 = 0
    return AP, mTTA, TTA_R80


def average_losses(losses_all):
    total_loss, cross_entropy, log_posterior, log_prior = 0, 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        log_posterior += losses['log_posterior']
        log_prior += losses['log_prior']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['log_posterior'] = log_posterior / len(losses_all)
    losses_mean['log_prior'] = log_prior / len(losses_all)
    return losses_mean


def test_all(testdata_loader, model, time=90, gpu_ids=[0]):
    
    all_pred = []
    all_labels = []
    losses_all = []
    metrics = {}
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(testdata_loader):
            # run forward inference
            losses, pred_scores, hiddens = model(batch_xs, batch_ys, graph_edges, hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False)
            losses_all.append(losses)

            num_frames = batch_xs.size()[1]
            assert num_frames >= time
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, time), dtype=np.float32)
            # run inference
            for t in range(time):
                pred = pred_scores[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)

    loss_val = average_losses(losses_all)

    # evaluation
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    print('----------------------------------')
    print("Starting evaluation...")
    metrics['AP'], metrics['mTTA'], metrics['TTA_R80'] = evaluation(all_pred, all_labels, total_time=time)
    print('----------------------------------')
    
    return loss_val, metrics


def write_scalars(logger, cur_epoch, cur_iter, losses):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    log_prior = losses['log_prior'].mean().item()
    log_posterior = losses['log_posterior'].mean().item()
    # print info
    print('----------------------------------')
    print('epoch: %d, iter: %d' % (cur_epoch, cur_iter))
    print('total loss = %.6f' % (total_loss))
    print('cross_entropy = %.6f' % (cross_entropy))
    print('log_posterior = %.6f' % (log_posterior))
    print('log_prior = %.6f' % (log_prior))
    # write to tensorboard
    logger.add_scalars("train/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("train/losses/cross_entropy", {'cross_entropy': cross_entropy}, cur_iter)
    logger.add_scalars("train/losses/log_posterior", {'log_posterior': log_posterior}, cur_iter)
    logger.add_scalars("train/losses/log_prior", {'log_prior': log_prior}, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    # write to tensorboard
    logger.add_scalars("test/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("test/accuracy/AP", {'AP': metrics['AP']}, cur_iter)
    logger.add_scalars("test/accuracy/time-to-accident", {'mTTA': metrics['mTTA'], 
                                                          'TTA_R80': metrics['TTA_R80']}, cur_iter)


def write_weight_histograms(writer, net, epoch):
    writer.add_histogram('histogram/w1_mu', net.predictor.l1.weight_mu, epoch)
    writer.add_histogram('histogram/w1_rho', net.predictor.l1.weight_rho, epoch)
    writer.add_histogram('histogram/w2_mu', net.predictor.l2.weight_mu, epoch)
    writer.add_histogram('histogram/w2_rho', net.predictor.l2.weight_rho, epoch)
    writer.add_histogram('histogram/b1_mu', net.predictor.l1.bias_mu, epoch)
    writer.add_histogram('histogram/b1_rho', net.predictor.l1.bias_rho, epoch)
    writer.add_histogram('histogram/b2_mu', net.predictor.l2.bias_mu, epoch)
    writer.add_histogram('histogram/b2_rho', net.predictor.l2.bias_rho, epoch)


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar', device=torch.device('cuda')):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def train_eval():
    # hyperparameters

    h_dim = p.hidden_dim  # 32
    z_dim = p.latent_dim  # 16
    x_dim = p.feature_dim  # 4096

    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset, p.feature_name + '_features')
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
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # building model
    model = BayesGCRNN(x_dim, h_dim, z_dim, p.num_rnn)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 17, gamma=0.1, last_epoch=-1)

    # resume training 
    start_epoch = 0
    if p.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, filename=p.model_file)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train() # set the model into training status

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
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=True, drop_last=True)

    # write histograms
    write_weight_histograms(logger, model, 0)
    iter_cur = 0
    for k in range(p.epoch):
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        # adjust learning rate
        scheduler.step()
        for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(traindata_loader):
            # ipdb.set_trace()
            optimizer.zero_grad()
            losses, predictions, hidden_st = model(batch_xs, batch_ys, graph_edges, edge_weights=edge_weights, npass=2, nbatch=len(traindata_loader), loss_w=p.loss_weight)
            # backward
            losses['total_loss'].mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # write the losses info
            write_scalars(logger, k, iter_cur, losses)
            
            iter_cur += 1
            # test and evaluate the model
            if iter_cur % p.test_iter == 0:
                model.eval()
                loss_val, metrics = test_all(testdata_loader, model, time=90, gpu_ids=gpu_ids)
                model.train()
                # keep track of validation losses
                write_test_scalars(logger, k, iter_cur, loss_val, metrics)

        # save model
        model_file = os.path.join(model_dir, 'bayesian_gcrnn_model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'logger': logger}, model_file)
        print('Model has been saved as: %s'%(model_file))
        
        # write histograms
        write_weight_histograms(logger, model, k+1)

    logger.close()


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
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'i3d'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=20,
                        help='The number of iteration to perform a evaluation process.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='The dimension of hidden states in RNN. Default: 128')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='The dimension of latent space. Default: 64')
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help='The dimension of node features in graph. Default: 4096')
    parser.add_argument('--loss_weight', type=float, default=0.0001,
                        help='The weighting factor of the two loss functions. Default: 0.1')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'],
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

    p = parser.parse_args()
    if p.phase == 'test':
        raise NotImplementedError
        # test_eval()
    else:
        train_eval()
