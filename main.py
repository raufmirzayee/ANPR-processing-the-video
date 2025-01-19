from util import get_car, read_license_plate, write_txt
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort

# Initialize the SORT tracker
mot_tracker = Sort()

# Load models
coco_model = YOLO('./models/yolov8n.pt')
license_plate_detector = YOLO('./models/best1.pt')
categorization_model = YOLO('./models/best2.pt')  

# Define the list of vehicle class IDs
vehicles = [2, 3, 5, 7]


# Open the video capture
video_path = '4_5832256706804978506.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Dictionary to store results
results = {}

# Loop through frames in the video stream
frame_nmr = -1
ret = True
while ret:
    # Increment frame counter
    frame_nmr += 1

    # Read a frame from the video
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        try:
            track_ids = mot_tracker.update(np.asarray(detections_))
        except:
            pass
        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Use the categorization model to categorize the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_category = categorization_model(license_plate_crop)[0]

            # Initialize dictionaries to store cropped classes and their corresponding grayscale versions
            class_crops = {"Dari": None, "Interval": None, "English": None}
            class_crops_gray = {"Dari": None, "Interval": None, "English": None}

            # Modify the categorization logic based on recognized text
            class_name = None

            # Get the recognized text from OCR results
            for categories_detection in license_plate_category.boxes.data.tolist():
                x1_class, y1_class, x2_class, y2_class, score_class, class_id_class = categories_detection

                xcar1, ycar1, xcar2, ycar2, car_id, license_plate_category = get_car(
                    license_plate, track_ids, categories_detection)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Crop the detected class from the license plate image
                    class_name = None
                    if class_id_class == 0:
                        class_name = "Dari"
                    elif class_id_class == 1:
                        class_name = "English"
                    elif class_id_class == 2:
                        class_name = "Interval"
                    if class_name:
                        class_crop = license_plate_crop[int(y1_class):int(y2_class), int(x1_class):int(x2_class), :]

                        # Perform OCR on the segmented license plate using the recognized language
                        class_text, class_text_score = read_license_plate(class_crop, class_name)
                        print(class_text, class_text_score)
                        if class_text is not None:
                            # Check if the car_id is already in the results dictionary
                            if car_id not in results[frame_nmr]:
                                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'bbox_score': score,
                                                                                'Dari_Text': '',
                                                                                'Dari_Text_Score': '',
                                                                                'Interval_Text': '',
                                                                                'Interval_Text_Score': '',
                                                                                'English_Text': '',
                                                                                'English_Text_Score': ''}}
                            # Update the text values in the results dictionary
                            results[frame_nmr][car_id]['license_plate'][f'{class_name}_Text'] = class_text
                            results[frame_nmr][car_id]['license_plate'][f'{class_name}_Text_Score'] = class_text_score


        write_txt(results ,'4_5832256706804978506.txt')
