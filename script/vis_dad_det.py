import os
import numpy as np
import cv2

def vis_det(data_path, video_path, phase='training'):
    files_list = []
    batch_id = 1
    for filename in sorted(os.listdir(os.path.join(data_path, phase))):
        filepath = os.path.join(data_path, phase, filename)
        all_data = np.load(filepath)
        features = all_data['data']  # 10 x 100 x 20 x 4096
        labels = all_data['labels']  # 10 x 2
        detections = all_data['det']  # 10 x 100 x 19 x 6
        videos = all_data['ID']  # 10
        nid = 1
        for i, vid in enumerate(videos):
            tag = 'positive' if labels[i, 1] > 0 else 'negative'
            video_file = os.path.join(video_path, phase, tag, vid.decode('UTF-8') + '.mp4')
            if not os.path.exists(video_file):
                raise FileNotFoundError
            bboxes = detections[i]
            counter = 0
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
            text_color = (0, 0, 255) if labels[i, 1] > 0 else (0, 255, 255)
            while (ret):
                new_bboxes = bboxes[counter, :, :]
                for num_box in range(new_bboxes.shape[0]):
                    cv2.rectangle(frame, (new_bboxes[num_box, 0], new_bboxes[num_box, 1]),
                                      (new_bboxes[num_box, 2], new_bboxes[num_box, 3]), (255, 0, 0), 2)
                cv2.putText(frame, tag, (int(frame.shape[1] / 2) - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            text_color, 2, cv2.LINE_AA)
                cv2.imshow('result', frame)
                c = cv2.waitKey(50)
                ret, frame = cap.read()
                if c == ord('q') and c == 27 and ret:
                    break;
                counter += 1
            cv2.destroyAllWindows()

if __name__ == '__main__':
    FEAT_PATH = '/data/DAD/features'
    VIDEO_PATH = '/data/DAD/videos'
    vis_det(FEAT_PATH, VIDEO_PATH)
