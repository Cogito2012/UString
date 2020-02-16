import os
import cv2
import numpy as np
import torch
# from torchvision.ops import RoIAlign
# import torchvision.ops.roi_align as roi_align
import sys
sys.path.insert(0, './lib/RoIAlign')
from roi_align import RoIAlign      # RoIAlign module
from pytorch_i3d import InceptionI3d
from torch.autograd import Variable

def get_video_data(video_file, ratio=0.5):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        # reduce the spatial dimension
        # ratio = 512.0 / min(frame.shape[0], frame.shape[1])
        dim_reduced = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        # dim_reduced = (224, 224)
        frame = cv2.resize(frame, dim_reduced, interpolation=cv2.INTER_AREA)
        frame = (frame/255.)*2-1
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert counter == N_FRAMES
    video_data = np.array(video_data, dtype=np.float32)  # 100 x 360 x 640 x 3
    # video_data = video_data[range(0, counter, 2)]  # 50 x 360 x 640
    video_data = np.expand_dims(np.transpose(video_data, [3, 0, 1, 2]), axis=0)    # 1 x 3 x 50 x 360 x 640
    return video_data

def extract_features(data_path, video_path, dest_path, phase):
    files_list = []
    batch_id = 1
    time_step = 20  # 1 seconds per run
    for filename in sorted(os.listdir(os.path.join(data_path, phase))):
        filepath = os.path.join(data_path, phase, filename)
        all_data = np.load(filepath)
        features_old = all_data['data']  # 10 x 100 x 20 x 4096
        labels = all_data['labels']  # 10 x 2
        detections = all_data['det']  # 10 x 100 x 19 x 6
        videos = all_data['ID']  # 10
        nid = 1
        features_i3d = np.zeros((labels.shape[0], N_FRAMES, N_BOXES + 1, FEAT_DIM))
        for i, vid in enumerate(videos):
            vidname = 'b' + str(batch_id).zfill(3) + '_' + vid.decode('UTF-8')
            tag = 'positive' if labels[i, 1] > 0 else 'negative'
            video_file = os.path.join(video_path, phase, tag, vid.decode('UTF-8') + '.mp4')
            # read video data and run inference
            # import ipdb; ipdb.set_trace()
            data_input = get_video_data(video_file, ratio=0.5)
            num_segments = int(data_input.shape[2] / time_step)  # 5
            with torch.no_grad():
                data_input = torch.from_numpy(data_input).cuda()    # 1 x 3 x 50 x 360 x 640
                for t in range(num_segments):
                    # get I3D features for each segment
                    start = t * time_step
                    end = (t+1) * time_step
                    feats = i3d.extract_features(data_input[:, :, start: end, :, :])  # 1 x 2048 x 10 x 23 x 40
                    feats = feats.squeeze(0).permute(1, 0, 2, 3)  # 10 x 2048 x 23 x 40
                    feats_global = torch.mean(feats, dim=(2, 3))  # 10 x 2048
                    for f in range(start, end):
                        # ROIAlign
                        boxes = torch.from_numpy(detections[i, f, :, :4] / STRIDE).to(torch.float).cuda()  # 19 x 5
                        # Note that the features are temporally pooled with 2
                        # We need to map the features to original temporal domain (10xCxHxW --> 20xCxHxW)
                        idx = int((f-start) / 2)
                        feat_map = feats[idx].unsqueeze(0).contiguous()
                        batch_inds = torch.tensor([0] * boxes.size(0), dtype=torch.int).cuda()
                        feats_rois = roi_align(feat_map, boxes, batch_inds)  # 19 x 2048 x 7 x 7
                        features_i3d[i, f, :N_BOXES, :] = torch.mean(feats_rois, dim=(2, 3)).cpu().numpy()
                        features_i3d[i, f, N_BOXES, :] = feats_global[idx].unsqueeze(0).contiguous().cpu().numpy()
                        # feats_rois = roi_align(feats[idx], rois, (7, 7), spatial_scale=1.0/32, sampling_ratio=-1)
            if vidname in files_list:
                vidname = vidname + '_' + str(nid).zfill(2)
                nid += 1
            feat_file = os.path.join(dest_path, vidname + '.npz')
            if os.path.exists(feat_file):
                continue
            np.savez_compressed(feat_file, data=features_i3d[i], labels=labels[i], det=detections[i], ID=vidname)
            print('batch: %03d, %s file: %s' % (batch_id, phase, vidname))
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

if __name__ == '__main__':
    DAD_PATH = '/data/DAD/features'
    VIDEO_PATH = '/data/DAD/videos'
    DEST_PATH = '/data/DAD/features_i3d'
    MODEL_FILE = 'models_i3d/i3d_r50_nl_kinetics.pth'
    N_FRAMES = 100
    N_BOXES = 19
    STRIDE = 16
    FEAT_DIM = 2048
    # original Inception-based I3D
    # i3d = InceptionI3d(400, in_channels=3)
    # #i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load(MODEL_FILE))
    # i3d.cuda()

    # ResNet-50 I3D
    from resi3d import resnet
    i3d = resnet.i3_res50_nl(400)
    # roi_align = RoIAlign((7, 7), spatial_scale=1.0/32, sampling_ratio=-1)
    roi_align = RoIAlign(7, 7)
    roi_align = roi_align.cuda()

    run(DAD_PATH, VIDEO_PATH, DEST_PATH)
