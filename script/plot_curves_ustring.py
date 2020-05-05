import matplotlib.pyplot as plt
import os
import numpy as np

def parse_data(logfile, method='baseline'):
    fid = open(logfile, 'r')
    epoch, iters = [], []
    losses = {'total':[], 'ce':[], 'posterior':[], 'prior':[], 'aux':[], 'rank':[]}
    accs = {'ap':[], 'mTTA':[], 'ttar80':[]}
    for line in fid.readlines():
        if "iter:" in line:
            epoch_id = int(line.strip().split(', ')[0].split('epoch: ')[-1])
            iter_id = int(line.strip().split(', ')[1].split('iter: ')[-1])
            epoch.append(epoch_id)
            iters.append(iter_id)
        if "total loss" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['total'].append(loss)
        if "cross_entropy" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['ce'].append(loss)
        if "log_posterior" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['posterior'].append(loss)
        if "log_prior" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['prior'].append(loss)
        if "aux_loss" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['aux'].append(loss)
        if "rank_loss" in line:
            loss = float(line.strip().split(' = ')[-1])
            losses['rank'].append(loss)
        if "Precision" in line:
            ap = line.strip().split(', ')[0].split('= ')[-1]
            ap = float(ap) if ap != 'nan' else 0.0
            accs['ap'].append(ap)
            mTTA = line.strip().split(', ')[-1].split('= ')[-1]
            mTTA = float(mTTA) if mTTA != 'nan' else 0.0
            accs['mTTA'].append(mTTA)
        if "Recall" in line:
            tta = line.strip().split('= ')[-1]
            tta = float(tta) if tta != 'nan' else 0.0
            accs['ttar80'].append(tta)
    fid.close()
    if method == 'baseline':
        losses['ce'] = [total - (posterior-prior)*0.001 - aux*10 - rank*10 for total, posterior, prior, aux, rank in zip(losses['total'],losses['posterior'], losses['prior'], losses['aux'], losses['rank'])]
    if method == 'noRankLoss':
        losses['ce'] = [total - (posterior-prior)*0.001 - aux*10 for total, posterior, prior, aux in zip(losses['total'],losses['posterior'], losses['prior'], losses['aux'])]
    if method == 'noBNNs':
        losses['ce'] = [total - aux*10 for total, aux in zip(losses['total'], losses['aux'])]
    
    return epoch, iters, losses, accs

def plot_metrics(logfile_baseline, logfile_noBayes, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse baseline
    epoch1, iters1, losses1, accs1 = parse_data(logfile_baseline)
    epoch2, iters2, losses2, accs2 = parse_data(logfile_noBayes)
    
    iters_len = min(len(iters1), len(iters2))
    celoss1 = losses1['ce'][0:iters_len]
    celoss2 = losses2['ce'][0:iters_len]
    
    # # plot total loss curve
    # plt.figure(figsize=(8, 4))
    # plt.plot(iters, celoss1, 'r-')
    # plt.plot(iters, celoss2, 'b-')
    # plt.grid('on')
    # plt.xlabel('iters')
    # plt.ylabel('training loss')
    # plt.tight_layout()
    # plt.grid()
    
    epoch1_np = np.array(epoch1[0:iters_len], dtype=np.int)
    epoch2_np = np.array(epoch2[0:iters_len], dtype=np.int)
    celoss1_np = np.array(celoss1, dtype=np.float32)
    celoss2_np = np.array(celoss2, dtype=np.float32)
    celoss1_e, celoss2_e = [], []
    epochs = range(max(epoch1))
    for e in epochs:
        inds = np.where(epoch1_np == e)[0]
        avg = np.mean(celoss1_np[inds])
        celoss1_e.append(avg)
        inds = np.where(epoch2_np == e)[0]
        avg = np.mean(celoss2_np[inds])
        celoss2_e.append(avg)
    
    plt.figure(figsize=(8, 4))
    fontsize = 18
    plt.plot(epochs, celoss1_e, 'r-')
    plt.plot(epochs, celoss2_e, 'b-')
    plt.xlim(0, max(epochs)-1)
    plt.ylim(10, 40)
    plt.xticks(range(0, max(epochs), 2), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.ylabel('exp. loss', fontsize=fontsize)
    plt.legend(['UString', 'UString w/o BNNs'], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    
    # plot aps
    plt.figure(figsize=(8, 4))
    steps = range(0, min(len(accs1['ap']), len(accs2['ap'])))
    ap1 = accs1['ap'][0:len(steps)]
    ap2 = accs2['ap'][0:len(steps)]
    plt.plot(steps, ap1, 'g-')
    plt.plot(steps, ap2, 'b-')
    plt.grid(True)
    plt.xlabel('test step')
    plt.ylabel('AP')
    plt.legend(['UString', 'UString w/o BNNs'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ap.png'))
    
    # plot mtta
    plt.figure(figsize=(8, 4))
    steps = range(0, min(len(accs1['mTTA']), len(accs2['mTTA'])))
    mTTA1 = accs1['mTTA'][0:len(steps)]
    mTTA2 = accs2['mTTA'][0:len(steps)]
    plt.plot(steps, mTTA1, 'g-')
    plt.plot(steps, mTTA2, 'b-')
    plt.grid(True)
    plt.xlabel('test step')
    plt.ylabel('mTTA')
    plt.legend(['UString', 'UString w/o BNNs'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mtta.png'))
    
    # plot tta
    plt.figure(figsize=(8, 4))
    steps = range(0, min(len(accs1['ttar80']), len(accs2['ttar80'])))
    TTA1 = accs1['ttar80'][0:len(steps)]
    TTA2 = accs2['ttar80'][0:len(steps)]
    plt.plot(steps, TTA1, 'g-')
    plt.plot(steps, TTA2, 'b-')
    plt.grid(True)
    plt.xlabel('test step')
    plt.ylabel('TTA@R80')
    plt.legend(['UString', 'UString w/o BNNs'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ttaR80.png'))
    plt.show()


def plot_losses(logfile_baseline, logfile_noRank, logfile_noBayes):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse baseline
    epoch1, iters1, losses1, accs1 = parse_data(logfile_baseline, 'baseline')
    epoch2, iters2, losses2, accs2 = parse_data(logfile_noRank, 'noRankLoss')
    epoch3, iters3, losses3, accs3 = parse_data(logfile_noBayes, 'noBNNs')
    
    iters_len = min([len(iters1), len(iters2), len(iters3)])
    celoss1 = losses1['ce'][0:iters_len]
    celoss2 = losses2['ce'][0:iters_len]
    celoss3 = losses3['ce'][0:iters_len]
    
    epoch1_np = np.array(epoch1[0:iters_len], dtype=np.int)
    epoch2_np = np.array(epoch2[0:iters_len], dtype=np.int)
    epoch3_np = np.array(epoch3[0:iters_len], dtype=np.int)
    celoss1_np = np.array(celoss1, dtype=np.float32)
    celoss2_np = np.array(celoss2, dtype=np.float32)
    celoss3_np = np.array(celoss3, dtype=np.float32)
    celoss1_e, celoss2_e, celoss3_e = [], [], []
    epochs = range(min([max(epoch1), max(epoch2), max(epoch3)]))
    for e in epochs:
        inds = np.where(epoch1_np == e)[0]
        avg = np.mean(celoss1_np[inds])
        celoss1_e.append(avg)
        inds = np.where(epoch2_np == e)[0]
        avg = np.mean(celoss2_np[inds])
        celoss2_e.append(avg)
        inds = np.where(epoch3_np == e)[0]
        avg = np.mean(celoss3_np[inds])
        celoss3_e.append(avg)
    
    plt.figure(figsize=(8, 4))
    fontsize = 18
    plt.plot(epochs, celoss1_e, 'r^-')
    plt.plot(epochs, celoss2_e, 'bo-')
    plt.plot(epochs, celoss3_e, 'gs-')
    plt.xlim(0, max(epochs)-5)
    plt.ylim(15, 45)
    plt.xticks(range(0, max(epochs)-4, 2), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.ylabel('exp. loss', fontsize=fontsize)
    plt.legend(['BNNs + RankLoss', 'BNNs only', 'naive NNs'], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))

if __name__ == '__main__':
    logfile_baseline = './script/savedlogs/ablations_baseline-0420.txt'
    logfile_noRank = './script/savedlogs/ablation_noRankLoss.txt'
    logfile_noBayes = './script/savedlogs/ablation_noBayes.txt'
    save_dir='./script/res/'
    # plot_metrics(logfile_baseline, logfile_Bayes, save_dir)
    plot_losses(logfile_baseline, logfile_noRank, logfile_noBayes)
