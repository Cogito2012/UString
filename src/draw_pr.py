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
            # Precision[cnt] = np.nan
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            # Recall[cnt] = np.nan
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            # Time[cnt] = np.nan
            continue
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

    return new_Precision, new_Recall, new_Time

if __name__ == "__main__":
    result_dir = './output_dev/gcrnn/vgg16/dad/test'
    # eval our own model
    result_file = os.path.join(result_dir, "pred_res.npz")
    data = np.load(result_file)
    all_pred = data['pred']
    all_labels = data['label']
    total_time = data['total_time']
    precision, recall, tta = evaluation(all_pred, all_labels, total_time)

    # eval DSARNN model
    result_file = "./output_dev/dsa_rnn_tf/pred_res_dsarcnn.npz"
    data = np.load(result_file)
    all_pred = data['pred']
    all_labels = data['label']
    total_time = data['total_time']
    precision_dsarnn, recall_dsarnn, tta_dsarnn = evaluation(all_pred, all_labels, total_time)

    # draw comparison curves
    plt.figure()
    plt.plot(recall_dsarnn, precision_dsarnn, 'b-')
    plt.plot(recall, precision, 'r-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curves')
    plt.legend(['DSA-RNN', 'Ours'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'PRCurve.png'))

    plt.figure()
    plt.plot(recall_dsarnn, tta_dsarnn*5, 'b-')
    plt.plot(recall, tta*5, 'r-')
    plt.xlabel('Recall')
    plt.ylabel('Time to Accident')
    plt.ylim([0.0, 5])
    plt.xlim([0.0, 1.0])
    plt.title('Recall Time-to-Accident Curves' )
    plt.legend(['DSA-RNN', 'Ours'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'TRCurve.png'))
    plt.show()