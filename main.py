#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil

from torch.utils.data import DataLoader
from src.Models import UString
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


def average_losses(losses_all):
    total_loss, cross_entropy, log_posterior, log_prior, aux_loss, rank_loss = 0, 0, 0, 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        log_posterior += losses['log_posterior']
        log_prior += losses['log_prior']
        aux_loss += losses['auxloss']
        rank_loss += losses['ranking']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['log_posterior'] = log_posterior / len(losses_all)
    losses_mean['log_prior'] = log_prior / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    losses_mean['ranking'] = rank_loss / len(losses_all)
    return losses_mean


def test_all(testdata_loader, model):
    
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas) in enumerate(testdata_loader):
            # run forward inference
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas, graph_edges, 
                    hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False)
            # make total loss
            losses['total_loss'] = p.loss_alpha * (losses['log_posterior'] - losses['log_prior']) + losses['cross_entropy']
            losses['total_loss'] += p.loss_beta * losses['auxloss']
            losses['total_loss'] += p.loss_yita * losses['ranking']
            losses_all.append(losses)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            # run inference
            for t in range(num_frames):
                pred = all_outputs[t]['pred_mean']
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    
    return all_pred, all_labels, all_toas, losses_all


def test_all_vis(testdata_loader, model, vis=True, multiGPU=False, device=torch.device('cuda')):
    
    if multiGPU:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()

    all_pred = []
    all_labels = []
    all_toas = []
    vis_data = []
    all_uncertains = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas, detections, video_ids) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            # run forward inference
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas, graph_edges, 
                    hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False, eval_uncertain=True)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            pred_uncertains = np.zeros((batch_size, num_frames, 2), dtype=np.float32)
            # run inference
            for t in range(num_frames):
                # prediction
                pred = all_outputs[t]['pred_mean']  # B x 2
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
                # uncertainties
                aleatoric = all_outputs[t]['aleatoric']  # B x 2 x 2
                aleatoric = aleatoric.cpu().numpy() if aleatoric.is_cuda else aleatoric.detach().numpy()
                epistemic = all_outputs[t]['epistemic']  # B x 2 x 2
                epistemic = epistemic.cpu().numpy() if epistemic.is_cuda else epistemic.detach().numpy()
                pred_uncertains[:, t, 0] = aleatoric[:, 0, 0] + aleatoric[:, 1, 1]
                pred_uncertains[:, t, 1] = epistemic[:, 0, 0] + epistemic[:, 1, 1]

            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)
            all_uncertains.append(pred_uncertains)

            if vis:
                # gather data for visualization
                vis_data.append({'pred_frames': pred_frames, 'label': label, 'pred_uncertain': pred_uncertains,
                                'toa': toas, 'detections': detections, 'video_ids': video_ids})

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    all_uncertains = np.vstack((np.vstack(all_uncertains[:-1]), all_uncertains[-1]))

    return all_pred, all_labels, all_toas, all_uncertains, vis_data


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    log_prior = losses['log_prior'].mean().item()
    log_posterior = losses['log_posterior'].mean().item()
    aux_loss = losses['auxloss'].mean().item()
    rank_loss = losses['ranking'].mean().item()
    # print info
    print('----------------------------------')
    print('epoch: %d, iter: %d' % (cur_epoch, cur_iter))
    print('total loss = %.6f' % (total_loss))
    print('cross_entropy = %.6f' % (cross_entropy))
    print('log_posterior = %.6f' % (log_posterior))
    print('log_prior = %.6f' % (log_prior))
    print('aux_loss = %.6f' % (aux_loss))
    print('rank_loss = %.6f' % (rank_loss))
    # write to tensorboard
    logger.add_scalars("train/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("train/losses/cross_entropy", {'cross_entropy': cross_entropy}, cur_iter)
    logger.add_scalars("train/losses/log_posterior", {'log_posterior': log_posterior}, cur_iter)
    logger.add_scalars("train/losses/log_prior", {'log_prior': log_prior}, cur_iter)
    logger.add_scalars("train/losses/complexity_cost", {'complexity_cost': log_posterior-log_prior}, cur_iter)
    logger.add_scalars("train/losses/aux_loss", {'aux_loss': aux_loss}, cur_iter)
    logger.add_scalars("train/losses/rank_loss", {'rank_loss': rank_loss}, cur_iter)
    # write learning rate
    logger.add_scalars("train/learning_rate/lr", {'lr': lr}, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    # write to tensorboard
    loss_info = {'total_loss': total_loss, 'cross_entropy': cross_entropy}
    aux_loss = losses['auxloss'].mean().item()
    loss_info.update({'aux_loss': aux_loss})
    logger.add_scalars("test/losses/total_loss", loss_info, cur_iter)
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


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
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
    ### --- CONFIG PATH ---
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
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        train_data = A3DDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = A3DDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)
    
    # building model
    model = UString(train_data.dim_feature, p.hidden_dim, p.latent_dim, 
                       n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps, 
                       with_saa=True, uncertain_ranking=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train() # set the model into training status

    # resume training 
    start_epoch = -1
    if p.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)

    # write histograms
    write_weight_histograms(logger, model, 0)
    iter_cur = 0
    best_metric = 0
    for k in range(p.epoch):
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas) in enumerate(traindata_loader):
            # ipdb.set_trace()
            optimizer.zero_grad()
            losses, all_outputs, hidden_st = model(batch_xs, batch_ys, batch_toas, graph_edges, edge_weights=edge_weights, npass=2, nbatch=len(traindata_loader), eval_uncertain=True)
            complexity_loss = losses['log_posterior'] - losses['log_prior']
            losses['total_loss'] = p.loss_alpha * complexity_loss + losses['cross_entropy']
            losses['total_loss'] += p.loss_beta * losses['auxloss']
            losses['total_loss'] += p.loss_yita * losses['ranking']
            # backward
            losses['total_loss'].mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # write the losses info
            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, iter_cur, losses, lr)
            
            iter_cur += 1
            # test and evaluate the model
            if iter_cur % p.test_iter == 0:
                model.eval()
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
                model.train()
                loss_val = average_losses(losses_all)
                print('----------------------------------')
                print("Starting evaluation...")
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'] = evaluation(all_pred, all_labels, all_toas, fps=test_data.fps)
                print('----------------------------------')
                # keep track of validation losses
                write_test_scalars(logger, k, iter_cur, loss_val, metrics)

        # save model
        model_file = os.path.join(model_dir, 'bayesian_gcrnn_model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            # update best model file
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))
        print('Model has been saved as: %s'%(model_file))

        scheduler.step(losses['log_posterior'])
        # write histograms
        write_weight_histograms(logger, model, k+1)
    logger.close()


def update_final_model(src_file, dest_file):
    # source file must exist
    assert os.path.exists(src_file), "src file does not exist!"
    # destinate file should be removed first if exists
    if os.path.exists(dest_file):
        if not os.path.samefile(src_file, dest_file):
            os.remove(dest_file)
    # copy file
    shutil.copyfile(src_file, dest_file)


def test_eval():
    ### --- CONFIG PATH ---
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    # result path
    result_dir = os.path.join(p.output_dir, p.dataset, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # visualization results
    p.visualize = False if p.evaluate_all else p.visualize
    vis_dir = None
    if p.visualize:
        vis_dir = os.path.join(result_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        test_data = A3DDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)
    num_samples = len(test_data)
    print("Number of testing samples: %d"%(num_samples))
    
    # building model
    model = UString(test_data.dim_feature, p.hidden_dim, p.latent_dim, 
                       n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps, 
                       with_saa=True, uncertain_ranking=True)

    # start to evaluate
    if p.evaluate_all:
        model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
        assert os.path.exists(model_dir)
        Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all = [], [], [], [], [], []
        modelfiles = sorted(os.listdir(model_dir))
        for filename in modelfiles:
            epoch_str = filename.split("_")[-1].split(".pth")[0]
            print("Evaluation for epoch: " + epoch_str)
            model_file = os.path.join(model_dir, filename)
            model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
            # run model inference
            all_pred, all_labels, all_toas, all_uncertains, _ = test_all_vis(testdata_loader, model, vis=False, device=device)
            # evaluate results
            AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, all_toas, fps=test_data.fps)
            mUncertains = np.mean(all_uncertains, axis=(0, 1))
            all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
            AP_video = average_precision_score(all_labels, all_vid_scores)
            APvid_all.append(AP_video)
            # save
            Epochs.append(epoch_str)
            AP_all.append(AP)
            mTTA_all.append(mTTA)
            TTA_R80_all.append(TTA_R80)
            Unc_all.append(mUncertains)
        # print results to file
        print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all, result_dir)
    else:
        result_file = os.path.join(vis_dir, "..", "pred_res.npz")
        if not os.path.exists(result_file):
            model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)
            # run model inference
            all_pred, all_labels, all_toas, all_uncertains, vis_data = test_all_vis(testdata_loader, model, vis=True, device=device)
            # save predictions
            np.savez(result_file[:-4], pred=all_pred, label=all_labels, toas=all_toas, uncertainties=all_uncertains, vis_data=vis_data)
        else:
            print("Result file exists. Loaded from cache.")
            all_results = np.load(result_file, allow_pickle=True)
            all_pred, all_labels, all_toas, all_uncertains, vis_data = \
                all_results['pred'], all_results['label'], all_results['toas'], all_results['uncertainties'], all_results['vis_data']
        # evaluate results
        all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
        AP_video = average_precision_score(all_labels, all_vid_scores)
        print("video-level AP=%.5f"%(AP_video))
        AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, all_toas, fps=test_data.fps)
        # evaluate uncertainties
        mUncertains = np.mean(all_uncertains, axis=(0, 1))
        print("Mean aleatoric uncertainty: %.6f"%(mUncertains[0]))
        print("Mean epistemic uncertainty: %.6f"%(mUncertains[1]))
        # visualize
        vis_results(vis_data, p.batch_size, vis_dir)


if __name__ == '__main__':
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
    parser.add_argument('--loss_alpha', type=float, default=0.001,
                        help='The weighting factor of posterior and prior losses. Default: 1e-3')
    parser.add_argument('--loss_beta', type=float, default=10,
                        help='The weighting factor of auxiliary loss. Default: 10')
    parser.add_argument('--loss_yita', type=float, default=10,
                        help='The weighting factor of uncertainty ranking loss. Default: 10')
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
        test_eval()
    else:
        train_eval()
