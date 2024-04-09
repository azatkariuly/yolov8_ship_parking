import cv2
import numpy as np

from ultralytics import YOLO

det_weight = 'best.pt'

class YOLOv8_Detection:
    def __init__(self, model_path, conf=0.5):
        self.model = YOLO(model_path, task='predict')
        self.conf = conf

    def detect(self, img):

        result = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)[0]

        bboxes, class_ids, scores = [], [], []

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
        class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
        scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        has_score = False

        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            (x, y, x2, y2) = bbox
            cv2.rectangle(img, (x,y), (x2,y2), (0, 0, 255), 2)
            # cv2.putText(img, self.cl_pr[class_id][0], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, self.cl_pr[class_id][1], 2)

            # if class_id == 2:
            #     ball_position = bbox

            # if class_id == 3:
            #     has_score = True

        return img
    

model_det = YOLOv8_Detection(det_weight)

# Open the video file
video_path = '2024-03-13/하이트비전영상/192.168.1.7_01_20240313105731960_5.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# # Get the frame rate of the video
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Calculate the frame number to skip to 1 minute
# frame_to_skip = int(332 * fps )

# # Set the frame position to skip to 1 minute
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_skip)

# Read and display frames until the video ends
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True
    if ret:
        # Display the frame
        frame = model_det.detect(frame)
        cv2.imshow('Frame', frame)
        # frame_filename = f"frame_5_test.jpg"

        # cv2.imwrite(frame_filename, frame)

        # Wait for 25 milliseconds. If 'q' is pressed, exit the loop.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if no frame is read
        break

    # break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
