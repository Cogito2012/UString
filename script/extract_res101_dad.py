from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import os, cv2
import argparse, sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

CLASSES = ('__background__', 'Car', 'Pedestrian', 'Cyclist')

class ResNet(nn.Module):
    def __init__(self, n_layers=101):
        super(ResNet, self).__init__()
        if n_layers == 50:
            self.net = models.resnet50(pretrained=True)
        elif n_layers == 101:
            self.net = models.resnet101(pretrained=True)
        else:
            raise NotImplementedError
        self.dim_feat = 2048
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output


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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


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


def bbox_to_imroi(bboxes, image):
    """
    bboxes: (n, 4), ndarray
    image: (H, W, 3), ndarray
    """
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data


def get_boxes(dets_all, im_size):
    bboxes = []
    for bbox in dets_all:
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        x1 = min(max(0, x1), im_size[1]-1)  # 0<=x1<=W-1
        y1 = min(max(0, y1), im_size[0]-1)  # 0<=y1<=H-1
        x2 = min(max(x1, x2), im_size[1]-1) # x1<=x2<=W-1
        y2 = min(max(y1, y2), im_size[0]-1) # y1<=y2<=H-1
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        if h > 2 and w > 2:  # the area is at least 9
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes, dtype=np.int32)
    return bboxes


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
        detections = all_data['det']  # 10 x 100 x 19 x 6
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
            for j, frame in tqdm(enumerate(video_frames), desc="The %d-th video"%(i+1), total=len(video_frames)):
                # find the non-empty boxes
                bboxes = get_boxes(detections[i, j], frame.shape)  # n x 4
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    # extract image feature
                    image = transform(Image.fromarray(frame))
                    ims_frame = torch.unsqueeze(image, dim=0).float().to(device=device)
                    feature_frame = torch.squeeze(feat_extractor(ims_frame))
                    features_res101[j, 0, :] = feature_frame.cpu().numpy() if feature_frame.is_cuda else feature_frame.detach().numpy()
                    # extract object feature
                    if len(bboxes) > 0:
                        # bboxes to roi data
                        ims_roi = bbox_to_imroi(bboxes, frame)  # (n, 3, 224, 224)
                        ims_roi = ims_roi.float().to(device=device)
                        feature_roi = torch.squeeze(torch.squeeze(feat_extractor(ims_roi), dim=-1), dim=-1)  # (2048,)
                        features_res101[j, 1:len(bboxes)+1,:] = feature_roi.cpu().numpy() if feature_roi.is_cuda else feature_roi.detach().numpy()
            # we only update the features
            np.savez_compressed(feat_file, data=features_res101, det=detections[i], labels=labels[i], ID=vidname)
            files_list.append(vidname)
        batch_id += 1
    return files_list


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

    args = parse_args()

    # prepare faster rcnn detector
    feat_extractor = ResNet(n_layers=101)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feat_extractor = feat_extractor.to(device=device)
    feat_extractor.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    data_path = osp.join(args.dad_dir, 'features')  # /data/DAD/features
    video_path = osp.join(args.dad_dir, 'videos')   # /data/DAD/videos
    run(data_path, video_path, args.out_dir)        # out: /data/DAD/res101_features

    print("Done!")