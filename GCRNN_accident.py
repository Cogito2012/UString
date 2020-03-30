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
from src.GraphModels import GCRNN
from src.eval_tools import evaluation, print_results, vis_results
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)
 

def test_all(testdata_loader, model, time=90):
    
    all_pred = []
    all_labels = []
    loss_val, loss_acc_val, loss_aux_val = 0, 0, 0
    for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(testdata_loader):
        with torch.no_grad():
            acc_loss, aux_loss, pred_scores, hiddens = model(batch_xs, batch_ys, graph_edges, 
                    hidden_in=None, edge_weights=edge_weights)
            loss = acc_loss + p.loss_weight * aux_loss

        loss_val += loss.mean().item()
        loss_acc_val += acc_loss.mean().item()
        loss_aux_val += aux_loss.mean().item()

        num_frames = batch_xs.size()[1]
        assert num_frames >= time
        batch_size = batch_xs.size()[0]
        pred_frames = np.zeros((batch_size, time), dtype=np.float32)
        # run inference
        with torch.no_grad():
            for t in range(time):
                pred = pred_scores[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
        # gather results and ground truth
        all_pred.append(pred_frames)
        label_onehot = batch_ys.cpu().numpy()
        label = np.reshape(label_onehot[:, 1], [batch_size,])
        all_labels.append(label)

    num_batch = i + 1
    loss_val /= num_batch
    loss_acc_val /= num_batch
    loss_aux_val /= num_batch

    # evaluation
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    print('----------------------------------')
    print("Starting evaluation...")
    AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, total_time=time)
    print('----------------------------------')
    
    return loss_val, loss_acc_val, loss_aux_val, AP, mTTA, TTA_R80


def test_all_vis(testdata_loader, model, time=90, vis=True, multiGPU=False, device=torch.device('cuda')):
    
    if multiGPU:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()

    all_pred = []
    all_labels = []
    vis_data = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, toa, detections, video_ids) in enumerate(testdata_loader):
            with torch.no_grad():
                acc_loss, aux_loss, pred_scores, hiddens = model(batch_xs, batch_ys, graph_edges, 
                        hidden_in=None, edge_weights=edge_weights)

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
            if vis:
                # gather data for visualization
                vis_data.append({'pred_frames': pred_frames, 'label': label,
                                'toa': toa, 'detections': detections, 'video_ids': video_ids})

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    return all_pred, all_labels, vis_data


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def train_eval():
    # hyperparameters
    if p.feature_name == 'vgg16':
        feature_dim = 4096 
    if p.feature_name == 'i3d':
        feature_dim = 2048

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
    model = GCRNN(feature_dim, p.hidden_dim, p.latent_dim, p.num_rnn)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # resume training 
    start_epoch = 0
    if p.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)

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
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    iter_cur = 0
    for k in range(p.epoch):
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_ys, graph_edges, edge_weights) in enumerate(traindata_loader):
            # ipdb.set_trace()
            optimizer.zero_grad()
            acc_loss, aux_loss, predictions, hidden_st = model(batch_xs, batch_ys, graph_edges, edge_weights=edge_weights)
            # backward
            loss = acc_loss + p.loss_weight * aux_loss
            loss.mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            print('----------------------------------')
            print('epoch: %d, iter: %d' % (k, iter_cur))
            print('loss = %.6f' % (loss.mean().item()))
            print('loss_acc = %.6f' % (acc_loss.mean().item()))
            print('loss_aux = %.6f' % (aux_loss.mean().item()))
            info = {'loss': loss.mean().item(), 'loss_acc': acc_loss.mean().item(), 'loss_aux': aux_loss.mean().item()}
            logger.add_scalars("losses/train", info, iter_cur)
            lr = optimizer.param_groups[0]['lr']
            logger.add_scalar("others/learning_rate", lr, iter_cur)
            
            iter_cur += 1
            # test and evaluate the model
            if iter_cur % p.test_iter == 0:
                model.eval()
                loss_val, loss_acc_val, loss_aux_val, AP, mTTA, TTA_R80 = test_all(testdata_loader, model, time=90)
                model.train()
                # keep track of validation losses
                info_losses = {'loss_total': loss_val, 'loss_acc': loss_acc_val, 'loss_aux': loss_aux_val}
                logger.add_scalars("losses/val_total", info_losses, iter_cur)
                logger.add_scalars("accuracy/val", {'AP': AP}, iter_cur)
                logger.add_scalars("time-to-accident/val", {'mTTA': mTTA, 'TTA_R80': TTA_R80}, iter_cur)

        # save model
        model_file = os.path.join(model_dir, 'gcrnn_model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
        print('Model has been saved as: %s'%(model_file))

        # adjust learning rate, using indicator as monitor
        indicator = 2 * AP * mTTA / (AP + mTTA)
        scheduler.step(indicator)
    logger.close()


def test_eval():
    # hyperparameters
    if p.feature_name == 'vgg16':
        feature_dim = 4096 
    if p.feature_name == 'i3d':
        feature_dim = 2048

    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset, p.feature_name + '_features')
    # result path
    result_dir = os.path.join(p.output_dir, p.dataset, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # visualization results
    # ipdb.set_trace()
    p.visualize = False if p.evaluate_all else p.visualize
    vis_dir = None
    if p.visualize:
        epoch_str = p.model_file.split("_")[-1].split(".pth")[0]
        vis_dir = os.path.join(result_dir, 'vis_epoch' + epoch_str)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    
    # building model
    model = GCRNN(feature_dim, p.hidden_dim, p.latent_dim, p.num_rnn)

    # start to evaluate
    if p.evaluate_all:
        model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
        assert os.path.exists(model_dir)
        AP_all, mTTA_all, TTA_R80_all = [], [], []
        modelfiles = sorted(os.listdir(model_dir))
        for filename in modelfiles:
            epoch_str = filename.split("_")[-1].split(".pth")[0]
            print("Evaluation for epoch: " + epoch_str)
            model_file = os.path.join(model_dir, filename)
            model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
            # run model inference
            all_pred, all_labels, _ = test_all_vis(testdata_loader, model, time=90, vis=False, device=device)
            # evaluate results
            AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, total_time=90)
            AP_all.append(AP)
            mTTA_all.append(mTTA)
            TTA_R80_all.append(TTA_R80)
        # print results to file
        print_results(AP_all, mTTA_all, TTA_R80_all, result_dir)
    else:
        model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)
        # run model inference
        all_pred, all_labels, vis_data = test_all_vis(testdata_loader, model, time=90, vis=True, device=device)
        # save predictions
        result_file = os.path.join(vis_dir, "..", "pred_res")
        np.savez(result_file, pred=all_pred, label=all_labels, total_time=90, vis_dir=vis_dir)
        # evaluate results
        AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, total_time=90)
        # visualize
        vis_results(vis_data, p.batch_size, vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['a3d', 'dad'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-4')
    parser.add_argument('--epoch', type=int, default=40,
                        help='The number of training epoches. Default: 40')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'i3d'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=64,
                        help='The number of iteration to perform a evaluation process.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The dimension of hidden states in RNN. Default: 256')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--loss_weight', type=float, default=0.1,
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
    parser.add_argument('--model_file', type=str, default='./output/dad/snapshot/gcrnn_model_90.pth',
                        help='The trained GCRNN model file for demo test only.')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='The directory of src need to save in the training.')

    p = parser.parse_args()
    if p.phase == 'test':
        test_eval()
    else:
        train_eval()
