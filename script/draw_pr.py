import numpy as np
import matplotlib.pyplot as plt
import os


def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5) 
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # # compute AP (area under P-R curve)
    # AP = 0.0
    # if new_Recall[0] != 0:
    #     AP += new_Precision[0]*(new_Recall[0]-0)
    # for i in range(1,len(new_Precision)):
    #     AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # # transform the relative mTTA to seconds
    # mTTA = np.mean(new_Time) * total_seconds
    # print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    # sort_time = new_Time[np.argsort(new_Recall)]
    # sort_recall = np.sort(new_Recall)
    # TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    # print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return new_Precision, new_Recall, new_Time


def old_evaluation(all_pred, all_labels, total_time = 90, length = None, speedup=False):
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
    if speedup:
        threshs = np.arange(np.min(all_pred), 1.0, 0.001)
    else:
        threshs = sorted(all_pred.flatten())
    for Th in threshs:
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
    rep_index = rep_index[1:] if speedup else rep_index
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    return new_Precision, new_Recall, new_Time

if __name__ == "__main__":
    result_dir = './output_dev/'
    # eval our own model (GCN-RNN)
    result_file = os.path.join(result_dir, "../backup/gcrnn_auxloss_0331/vgg16/dad/test/pred_res.npz")
    data = np.load(result_file)
    all_pred = data['pred']
    all_labels = data['label']
    # toas = data['toas']
    toas = [90] * all_labels.shape[0]
    precision, recall, tta = evaluation(all_pred, all_labels, toas, fps=20.0)

    # eval our own model (Bayes GCN-RNN)
    result_file = os.path.join(result_dir, "../backup/bayes_gcrnn_0402/vgg16/dad/test_bk/pred_res.npz")
    data = np.load(result_file)
    all_pred = data['pred']
    all_labels = data['label']
    # toas = data['toas']
    toas = [90] * all_labels.shape[0]
    precision_bayes, recall_bayes, tta_bayes = evaluation(all_pred, all_labels, toas)

    # # eval our own model (Bayes GCN-RNN with Uranking)
    # result_file = os.path.join(result_dir, "bayes_gcrnn_ranking/vgg16/dad/test/pred_res.npz")
    # data = np.load(result_file)
    # all_pred = data['pred']
    # all_labels = data['label']
    # # total_time = data['total_time']
    # precision_bayesUr, recall_bayesUr, tta_bayesUr = evaluation(all_pred, all_labels, total_time = 90)

    # eval DSARNN model
    # result_file = "./dsarnn_tf/eval/eval_dsarcnn_demo.npz"
    result_file = "./dsarnn_tf/eval/eval_dsarcnn_retrain.npz"
    data = np.load(result_file)
    all_pred = data['pred']
    all_labels = data['label']
    # toas = data['toas']
    toas = [90] * all_labels.shape[0]
    precision_dsarnn, recall_dsarnn, tta_dsarnn = evaluation(all_pred, all_labels, toas, fps=20.0)

    # draw comparison curves
    plt.figure()
    plt.plot(recall_dsarnn, precision_dsarnn, 'b-')
    plt.plot(recall, precision, 'k-')
    plt.plot(recall_bayes, precision_bayes, 'r-')
    # plt.plot(recall_bayesUr, precision_bayesUr, 'r--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curves')
    plt.legend(['DSA-RNN', 'Ours (String)', 'Ours (UString)'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'PRCurve.png'))

    plt.figure()
    plt.plot(recall_dsarnn, tta_dsarnn*5, 'b-')
    plt.plot(recall, tta*5, 'k-')
    plt.plot(recall_bayes, tta_bayes*5, 'r-')
    # plt.plot(recall_bayesUr, tta_bayesUr*5, 'r--')
    plt.xlabel('Recall')
    plt.ylabel('Time to Accident')
    plt.ylim([0.0, 5])
    plt.xlim([0.0, 1.0])
    plt.title('Time-to-Accident Recall Curves' )
    plt.legend(['DSA-RNN', 'Ours (String)', 'Ours (UString)'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'TRCurve.png'))
    plt.show()