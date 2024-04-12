import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ultralytics import YOLO

det_weight = 'last.pt'

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

def make_stats(left_distance='undefined', right_distance='undefined'):
    bg_width, bg_height = 1500, 500
    background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)

    # Define text content and parameters
    line1 = "Left:  " + str(left_distance)
    line2 = "Right: " + str(right_distance)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 20
    font_color = (255, 255, 255)  # White color

    # Add the text to the black background in two lines
    cv2.putText(background, line1, (200, 180), font, font_scale, font_color, font_thickness)
    cv2.putText(background, line2, (200, 380), font, font_scale, font_color, font_thickness)

    return background

model_det = YOLOv8_Detection(det_weight)

# Open the video file
video_path = '../2024-03-13/하이트비전영상/192.168.1.7_01_20240313105731960_3.mp4'
data_path = '../2024-03-13/Laser.xlsx'

cap = cv2.VideoCapture(video_path)
dataframe = pd.read_excel(data_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# Get the frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Calculate the frame number to skip to 1 minute
# frame_to_skip = int(39 * fps)

# # Set the frame position to skip to 1 minute
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_skip)

frame_counter = 1500
seconds_counter = 0

# Read and display frames until the video ends
while cap.isOpened():

    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True
    if ret:
        # Display the frame
        frame = model_det.detect(frame)
        frame_height, frame_width, _ = frame.shape

        frame = cv2.line(frame, (1000, 2100), (2800, 230), color=(255, 0, 0), thickness=9)
        frame = cv2.line(frame, (6855, 2800), (5280, 230), color=(255, 0, 0), thickness=9)

        bg_x = frame_width - 1500 # bg_width
        bg_y = 0

        left_text = None
        right_text = None

        if frame_counter > 39 * fps:
            d = dataframe[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].values[176:][seconds_counter // fps]
            
            left_text = int(str(d[1]).split('.')[0])
            right_text = int(str(d[2]).split('.')[0])

            seconds_counter += 1

        frame[bg_y:bg_y + 500, bg_x:bg_x + 1500] = make_stats(left_text if left_text else None, right_text if right_text else None)
        cv2.imshow('Frame', frame)
        # frame_filename = f"frame_5_test.jpg"

        # cv2.imwrite(frame_filename, frame)

        # Wait for 25 milliseconds. If 'q' is pressed, exit the loop.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if no frame is read
        break

    frame_counter += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
