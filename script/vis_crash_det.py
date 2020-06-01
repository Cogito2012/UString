import os, cv2
import numpy as np


def get_video_frames(video_file, n_frames=50):
    assert os.path.exists(video_file), video_file
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert len(video_data) >= n_frames
    video_data = video_data[:n_frames]
    return video_data


def vis_det(feat_path, video_path, out_path, tag='positive'):
    for filename in sorted(os.listdir(feat_path)):
        vid = filename.strip().split('.')[0]
        # load information
        filepath = os.path.join(feat_path, filename)
        all_data = np.load(filepath)
        features = all_data['data']  # 50 x 20 x 2048
        labels = all_data['labels']  # 1 x 2
        detections = all_data['det']  # 50 x 19 x 6
        videos = all_data['ID']  # 1
        # visualize dets
        video_file = os.path.join(video_path, vid + '.mp4')
        frames = get_video_frames(video_file, n_frames=50)
        # save_dir
        save_dir = os.path.join(out_path, vid)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        text_color = (0, 0, 255) if labels[1] > 0 else (0, 255, 255)
        for counter, frame in enumerate(frames):
            new_bboxes = detections[counter, :, :]
            for num_box in range(new_bboxes.shape[0]):
                cv2.rectangle(frame, (int(new_bboxes[num_box, 0]), int(new_bboxes[num_box, 1])),
                                    (int(new_bboxes[num_box, 2]), int(new_bboxes[num_box, 3])), (255, 0, 0), 2)
            cv2.putText(frame, tag, (int(frame.shape[1] / 2) - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        text_color, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_dir, str(counter) + '.jpg'), frame)

if __name__ == '__main__':
    FEAT_PATH = './data/crash/vgg16_features/positive'
    VIDEO_PATH = './data/crash/videos/Crash-1500'
    OUT_PATH = './data/crash/vgg16_features/vis_dets'
    vis_det(FEAT_PATH, VIDEO_PATH, OUT_PATH, tag='positive')