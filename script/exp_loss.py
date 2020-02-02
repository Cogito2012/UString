import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np

def tf_exp_loss(pred, y, time, fps=20.0, n_frames=100):
    # positive example (exp_loss)
    pos_loss = -tf.multiply(tf.exp(-(n_frames-time-1)/fps),-tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    # negative example
    neg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = pred) # Softmax loss

    loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss,y[:,1]),tf.multiply(neg_loss,y[:,0])))

    return loss, neg_loss

def _exp_loss(pred, target, time, frames=100):
    # transform the onehot target into scalar labels
    target_cls = target[:, 1].to(dtype=torch.int64)  # 1st column: negative, 2nd column: positive
    mask_pos = target[:, 1]
    mask_neg = target[:, 0]
    # positive loss
    pos_loss = torch.exp(-torch.tensor((frames - time)/20.0)) * ce_loss(pred, target_cls)
    # negative loss
    neg_loss = ce_loss(pred, target_cls)
    loss = torch.mean(torch.mul(pos_loss, mask_pos) + torch.mul(neg_loss, mask_neg))

    return loss, neg_loss

def pth_exp_loss(pred, target, time, frames=100):
    # transform the onehot target into scalar labels
    target_cls = target[:, 1].to(dtype=torch.int64)  # 1st column: negative, 2nd column: positive
    mask_pos = target[:, 1]
    mask_neg = target[:, 0]
    # positive loss
    pos_loss = torch.exp(-torch.tensor((frames - time - 1)/20.0)) * ce_loss(pred, target_cls)
    # negative loss
    neg_loss = ce_loss(pred, target_cls)
    loss = torch.mean(torch.mul(pos_loss, mask_pos) + torch.mul(neg_loss, mask_neg))

    return loss, neg_loss

def pth_exp_loss2(pred, target, time, fps=20.0, n_frames=100):
    # positive example (exp_loss)
    target_cls = target[:, 1]
    pos_loss = -torch.mul(torch.exp(-torch.tensor((n_frames-time-1)/fps)), -ce_loss(pred, target_cls.to(torch.long)))
    # negative example
    neg_loss = ce_loss(pred, target_cls.to(torch.long))

    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))

    return loss, neg_loss

def _nll_bernoulli(logits, target_adj_dense):
    temp_size = target_adj_dense.size()[0]
    temp_sum = target_adj_dense.sum()
    posw = float(temp_size * temp_size - temp_sum) / temp_sum
    norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
    nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                      , target=target_adj_dense
                                                      , pos_weight=posw
                                                      , reduction='none')
    nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
    return - nll_loss


if __name__ == '__main__':
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    # def ce_loss(x, y):
    #     LS = torch.nn.LogSoftmax(dim=1)
    #     NLL = torch.nn.NLLLoss(reduction='none')
    #     return NLL(LS(x), y)

    sess = tf.Session()

    np.random.seed(12345)
    prediction = np.random.rand(10, 2)
    label = np.random.randint(0, 2, (10, 1))
    targets = np.hstack((label, 1 - label))
    print(targets)

    # tensorflow implementation
    pred_tf = tf.constant(prediction, dtype=tf.float32)
    target_tf = tf.constant(targets, dtype=tf.float32)
    _loss_tf, _neg_loss_tf = tf_exp_loss(pred_tf, target_tf, 60)
    loss_tf, neg_loss_tf = sess.run([_loss_tf, _neg_loss_tf])

    # pytorch implementation
    pred_pth = torch.from_numpy(prediction).to(torch.float32)
    target_pth = torch.from_numpy(targets).to(torch.float32)
    _loss_pth, _neg_loss = _exp_loss(pred_pth, target_pth, 60)
    # _loss_pth, _neg_loss = pth_exp_loss2(pred_pth, target_pth, 60)
    loss_pth = _loss_pth.numpy()
    neg_loss_pth = _neg_loss.numpy()

    loss_ber = _nll_bernoulli(pred_pth, target_pth)
    print(loss_ber)

    print(loss_tf)
    print(loss_pth)

    print(neg_loss_tf)
    print(neg_loss_pth)