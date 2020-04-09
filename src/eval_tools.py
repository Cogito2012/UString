import numpy as np
import matplotlib.pyplot as plt
import os

def evaluation(all_pred, all_labels, total_time = 90, length = None):
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

    if mTTA == np.nan:
        mTTA = 0
    if TTA_R80 == np.nan:
        TTA_R80 = 0
    return AP, mTTA, TTA_R80


def print_results(AP_all, mTTA_all, TTA_R80_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for AP, mTTA, TTA_R80 in zip(AP_all, mTTA_all, TTA_R80_all):
            f.writelines('{:.3f} {:.3f} {:.3f}\n'.format(AP, mTTA, TTA_R80))
    f.close()
    

def vis_results(vis_data, batch_size, vis_dir):
    for results in vis_data:
        pred_frames = results['pred_frames']
        labels = results['label']
        toa = results['toa']
        video_ids = results['video_ids']
        detections = results['detections']
        uncertainties = results['pred_uncertain']
        for n in range(batch_size):
            if labels[n] == 1:
                pred_mean = pred_frames[n, :]  # (90,)
                pred_std_alea = 1.0 * np.sqrt(uncertainties[n, :, 0])
                pred_std_epis = 1.0 * np.sqrt(uncertainties[n, :, 1])
                # plot the probability predictions
                fig, ax = plt.subplots(1, figsize=(14, 5))
                ax.fill_between(range(1, len(pred_mean)+1), pred_mean - pred_std_alea, pred_mean + pred_std_alea, facecolor='wheat', alpha=0.5)
                ax.fill_between(range(1, len(pred_mean)+1), pred_mean - pred_std_epis, pred_mean + pred_std_epis, facecolor='yellow', alpha=0.5)
                plt.plot(range(1, len(pred_mean)+1), pred_mean, linewidth=3.0)
                plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                fontsize = 18
                plt.ylim(0, 1)
                plt.xlim(1, 100)
                plt.ylabel('Probability', fontsize=fontsize)
                plt.xlabel('Frame (FPS=20)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, video_ids[n] + '.png'))
                plt.close()