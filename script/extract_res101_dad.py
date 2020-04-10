from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys
# Add lib to PYTHONPATH
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '../lib', 'frcnn', 'lib')
sys.path.insert(0, lib_path)

import tensorflow as tf
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import im_detect
from model.nms_wrapper import nms
from nets.resnet_v1 import resnetv1

import numpy as np
import os, cv2
import argparse
from tqdm import tqdm

CLASSES = ('__background__', 'Car', 'Pedestrian', 'Cyclist')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dad_dir', dest='dad_dir', help='The directory to the Dashcam Accident Dataset', type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str)
    parser.add_argument('--n_frames', dest='n_frames', help='The number of frames sampled from each video', default=100)
    parser.add_argument('--n_boxes', dest='n_boxes', help='The number of bounding boxes for each frame', default=19)
    parser.add_argument('--dim_feat', dest='dim_feat', help='The dimension of extracted ResNet101 features', default=2048)
    parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS threshold to get detection boxes', default=0.8)
    # The followings are configurations for Faster R-CNN
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model', help='model to test', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to test', default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode', action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image', help='max number of detections per image', default=100, type=int)
    parser.add_argument('--tag', dest='tag', help='tag of the model', default='', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res50, res101, res152, mobile', default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def prepare_frcnn():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    # read arguments
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    net_name, imdb_name, model_name = args.net, args.imdb_name, args.model
    if not os.path.isfile(model_name + '.meta'):
        raise IOError(('{:s} not found.\n').format(model_name + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)

    # build faster rcnn detector
    if net_name == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 4, tag='default', anchor_scales=[2,4,8,16,32], anchor_ratios=[0.33,0.5,1,2,3])
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    return args, sess, net


def collect_bboxes(boxes, scores, roi_feats, nms_thresh=0.8, max_per_image=19):
    """
    :param boxes: (300, 16)
    :param scores: (300, 4), for 3 classes
    :param roi_feats: (300, 2048)
    """
    dets_all = []
    feats_all = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        feats = roi_feats[keep, :]
        if len(keep) > 0:
            boxes_cls = np.hstack(
                [dets, cls_ind * np.ones((len(keep), 1), dtype=np.int)])  # n x 6 (x1, y1, x2, y2, score, cls)
            dets_all.append(boxes_cls)
            feats_all.append(feats)

    dets_all = np.concatenate(dets_all)
    feats_all = np.concatenate(feats_all)
    assert dets_all.shape[0] >= max_per_image
    image_scores = dets_all[:, 4]
    thresh = np.sort(image_scores)[-max_per_image]
    keep = np.where(image_scores >= thresh)[0]
    dets_all = dets_all[keep, :]
    feats_all = feats_all[keep, :]

    return dets_all, feats_all


def scda_aggregate(feat_maps):
    '''re-implementation of Selective Convolutional Descriptors Aggregation (SCDA)
       ref: https://arxiv.org/abs/1604.04994
    Args:
        feat_maps: 3D tensor (feat_h x feat_w x feat_dim)
        name:
    Returns:
    '''
    A = np.sum(feat_maps, axis=-1)  # (h, w)
    thresh = np.mean(A, keepdims=True)  # (1, 1)
    mask = (A >= thresh).astype(np.float32)  # (h, w)
    feat_sel = np.expand_dims(mask, axis=-1) * feat_maps  # (h, w, D)
    feat_avg = np.mean(feat_sel, axis=(0, 1))  # (D,)
    feat_max = np.max(feat_sel, axis=(0, 1))  # (D,)
    feat_scda = np.concatenate((feat_avg, feat_max), axis=-1)  # (2D,)
    return feat_scda


def get_video_frames(video_file, n_frames=100):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert counter == n_frames
    return video_data


def extract_features(data_path, video_path, dest_path, phase):
    files_list = []
    batch_id = 1
    all_batches = os.listdir(os.path.join(data_path, phase))
    for filename in sorted(all_batches):
        filepath = os.path.join(data_path, phase, filename)
        all_data = np.load(filepath)
        # parse the original DAD dataset 
        labels = all_data['labels']  # 10 x 2
        videos = all_data['ID']  # 10
        # features_old = all_data['data']  # 10 x 100 x 20 x 4096 (will be replaced)
        # detections = all_data['det']  # 10 x 100 x 19 x 6 (will be replaced)
        # start to process each video
        nid = 1
        for i, vid in tqdm(enumerate(videos), desc="The %d-th batch"%(batch_id), total=len(all_batches)):
            vidname = 'b' + str(batch_id).zfill(3) + '_' + vid.decode('UTF-8')
            if vidname in files_list:
                vidname = vidname + '_' + str(nid).zfill(2)
                nid += 1
            feat_file = os.path.join(dest_path, vidname + '.npz')
            if os.path.exists(feat_file):
                continue
            # continue on feature extraction
            tag = 'positive' if labels[i, 1] > 0 else 'negative'
            video_file = os.path.join(video_path, phase, tag, vid.decode('UTF-8') + '.mp4')
            video_frames = get_video_frames(video_file, n_frames=args.n_frames)
            # start to process each frame
            features_res101 = np.zeros((args.n_frames, args.n_boxes + 1, args.dim_feat), dtype=np.float32)  # (100 x 20 x 2048)
            detections = np.zeros((args.n_frames, args.n_boxes, 6))  # (100 x 19 x 6)
            for j, frame in tqdm(enumerate(video_frames), desc="The %d-th video"%(i+1), total=len(video_frames)):
                # run Faster R-CNN detection
                scores, boxes, conv_feat, roi_feats = im_detect(session, faster_rcnn, frame, with_feat=True)
                # collect bounding boxes (19 per image) and features
                dets_all, roi_feats = collect_bboxes(boxes, scores, roi_feats, nms_thresh=args.nms_thresh, max_per_image=args.n_boxes)
                frame_feats = scda_aggregate(conv_feat[0])  # (2048,)
                features_res101[j, 0, :] = frame_feats
                features_res101[j, 1:,:] = roi_feats
                detections[j, :, :] = dets_all
            np.savez_compressed(feat_file, data=features_res101, det=detections[i], labels=labels[i], ID=vidname)
            files_list.append(vidname)
        batch_id += 1


def run(data_path, video_path, dest_path):
    # prepare the result paths
    train_path = os.path.join(dest_path, 'training')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(dest_path, 'testing')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # process training set
    train_list = extract_features(data_path, video_path, train_path, 'training')
    print('Training samples: %d'%(len(train_list)))
    # process testing set
    test_list = extract_features(data_path, video_path, test_path, 'testing')
    print('Testing samples: %d' % (len(test_list)))


if __name__ == "__main__":
    # prepare faster rcnn detector
    args, session, faster_rcnn = prepare_frcnn()

    data_path = osp.join(args.dad_dir, 'features')
    video_path = osp.join(args.dad_dir, 'videos')
    run(data_path, video_path, args.out_dir)

    print("Done!")