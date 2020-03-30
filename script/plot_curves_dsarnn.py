import matplotlib.pyplot as plt
import os

def parse_data(logfile):
    fid = open(logfile, 'r')
    epoch, losses, aps, mTTAs, TTA = [], [], [], [], []
    for line in fid.readlines():
        if "Loss:" in line:
            epoch_id = int(line.strip().split(' done')[0].split('Epoch: ')[-1])
            loss_val = float(line.strip().split("Loss: ")[-1])
            epoch.append(epoch_id)
            losses.append(loss_val)
        if "Precision" in line:
            ap = line.strip().split(', ')[0].split('= ')[-1]
            ap = float(ap) if ap != 'nan' else 0.0
            aps.append(ap)
            mTTA = line.strip().split(', ')[-1].split('= ')[-1]
            mTTA = float(mTTA) if mTTA != 'nan' else 0.0
            mTTAs.append(mTTA)
        if "Recall" in line:
            tta = line.strip().split('= ')[-1]
            tta = float(tta) if tta != 'nan' else 0.0
            TTA.append(tta)
    fid.close()
    return epoch, losses, aps, mTTAs, TTA

def plot_metrics(logfile, save_dir):
    epoch, losses, aps, mTTAs, TTA = parse_data(logfile)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(epoch, losses, 'r-')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    
    # plot aps
    plt.figure(figsize=(8, 4))
    ap_train = aps[1::2]
    ap_test = aps[0::2]
    plt.plot(epoch, ap_train, 'g-')
    plt.plot(epoch, ap_test, 'b-')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.ylabel('AP')
    plt.legend(['train', 'test'])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'ap_train_test.png'))
    
    # plot mtta
    plt.figure(figsize=(8, 4))
    mTTA_train = mTTAs[1::2]
    mTTA_test = mTTAs[0::2]
    plt.plot(epoch, mTTA_train, 'g-')
    plt.plot(epoch, mTTA_test, 'b-')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.ylabel('mTTA')
    plt.legend(['train', 'test'])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'mtta_train_test.png'))
    
    # plot tta
    plt.figure(figsize=(8, 4))
    TTA_train = TTA[1::2]
    TTA_test = TTA[0::2]
    plt.plot(epoch, TTA_train, 'g-')
    plt.plot(epoch, TTA_test, 'b-')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.ylabel('TTA@R80')
    plt.legend(['train', 'test'])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'tta_train_test.png'))
    # plt.show()

if __name__ == '__main__':
    logfile = './dsarnn_tf/log_retrain_dsarnn.txt'
    save_dir='./dsarnn_tf/dsarnn_plots'
    plot_metrics(logfile, save_dir)
