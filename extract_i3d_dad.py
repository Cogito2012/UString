import os
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, './lib/RoIAlign')
from roi_align import RoIAlign      # RoIAlign module
from torch.autograd import Variable
from src.ResNetI3D import i3_res50_nl


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
        features_i3d = np.zeros((labels.shape[0], N_FRAMES, N_BOXES + 1, FEAT_DIM))  # (10 x 100 x 20 x 2048)
        for i, vid in enumerate(videos):
            vidname = 'b' + str(batch_id).zfill(3) + '_' + vid.decode('UTF-8')
            tag = 'positive' if labels[i, 1] > 0 else 'negative'
            video_file = os.path.join(video_path, phase, tag, vid.decode('UTF-8') + '.mp4')
            # read video data and run inference
            # import ipdb; ipdb.set_trace()
            data_input = get_video_data(video_file, ratio=0.5)
            # num_segments = int(data_input.shape[2] / time_step)  # 5
            with torch.no_grad():
                data_input = torch.from_numpy(data_input).cuda()    # 1 x 3 x 100 x 360 x 640
                batch_inds = torch.tensor([0] * N_BOXES, dtype=torch.int).cuda()
                # compute the features for each frame 
                for t in range(N_FRAMES - TIME_STEP): # (1-90)
                    # get I3D features for each segment
                    feats = i3d.extract_features(data_input[:, :, t: (t+TIME_STEP), :, :])  # 1 x 2048 x 5 x 23 x 40
                    feats = feats.squeeze(0).permute(1, 0, 2, 3)  # 5 x 2048 x 23 x 40
                    feats = torch.mean(feats, dim=0)  # 2048 x 23 x 40
                    # ROIAlign
                    feat_map = feats.unsqueeze(0).contiguous()  # 1 x 2048 x 23 x 40
                    boxes = torch.from_numpy(detections[i, t, :, :4] / STRIDE).to(torch.float).cuda()  # 19 x 4
                    feats_rois = roi_align(feat_map, boxes, batch_inds)  # 19 x 2048 x 7 x 7
                    # full frame features
                    features_i3d[i, t, 0, :] = torch.mean(feats, dim=(1,2)).cpu().numpy()  # 2048
                    features_i3d[i, t, 1:, :] = torch.mean(feats_rois, dim=(2, 3)).cpu().numpy()
                # replicate the rest TIME_STEP (10) frame features
                features_i3d[i, (N_FRAMES - TIME_STEP):, :, :] = np.tile(np.expand_dims(features_i3d[i, N_FRAMES - TIME_STEP, :, :], axis=0), reps=(TIME_STEP, 1, 1))

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
    TIME_STEP = 10

    # ResNet-50 I3D
    i3d = i3_res50_nl(400)
    # roi_align = RoIAlign((7, 7), spatial_scale=1.0/32, sampling_ratio=-1)
    roi_align = RoIAlign(7, 7)
    roi_align = roi_align.cuda()

    run(DAD_PATH, VIDEO_PATH, DEST_PATH)
