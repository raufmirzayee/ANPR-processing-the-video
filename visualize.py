import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=20, line_length_x=50, line_length_y=50):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def draw_border_with_text(img, top_left, bottom_right, text, color=(0, 255, 0), text_color=(0, 255, 0), thickness=10):
    x1, y1 = top_left
    x2, y2 = bottom_right

    img = draw_border(img, top_left, bottom_right, color, thickness)  # Draw the border

    # Display text above the license plate
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    font_path = 'Yekan.ttf'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = int((x1 + x2 - text_width) / 2)
    text_y = int(y1 - 10)  # Adjust this value to control the vertical position of the text
    cv2.putText(img, text, (text_x, text_y), font, FONT_SCALE, text_color, FONT_THICKNESS)

    return img



# Constants and Configurations
RESULTS_txt = 'Final.txt'
VIDEO_PATH = '4_5832256706804978506.mp4'
OUTPUT_VIDEO_PATH = 'Final.mp4'
FONT_SCALE = 1.0
FONT_THICKNESS = 5
BORDER_THICKNESS = 10
LICENSE_PLATE_COLOR = (0, 0, 255)
BORDER_COLOR = (0, 255, 0)

def load_license_plate_info(results):
    license_plate_info = {}
    unique_car_ids = np.unique(results['Car_ID'])
    for car_id in unique_car_ids:
        max_D = np.amax(results[results['Car_ID'] == car_id]['Dari_Text_Score'])
        max_I = np.amax(results[results['Car_ID'] == car_id]['Interval_Text_Score'])
        max_E = np.amax(results[results['Car_ID'] == car_id]['English_Text_Score'])
        
        license_info = {}
        license_info['license_crop'] = None
        
        if max_D > 0:
            license_info['license_plate_Dari'] = results[(results['Car_ID'] == car_id) & (results['Dari_Text_Score'] == max_D)]['Dari_Text'].iloc[0]
        if max_I > 0:
            license_info['license_plate_Interval'] = results[(results['Car_ID'] == car_id) & (results['Interval_Text_Score'] == max_I)]['Interval_Text'].iloc[0]
        if max_E > 0:
            license_info['license_plate_English'] = results[(results['Car_ID'] == car_id) & (results['English_Text_Score'] == max_E)]['English_Text'].iloc[0]
        
        license_plate_info[car_id] = license_info
    
    return license_plate_info


def process_frame(frame, results, license_plate_info):
    frame_nmr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    df_ = results[results['Frame'] == frame_nmr]

    for row_indx in range(len(df_)):
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['Car_Bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        draw_border_with_text(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), "", BORDER_COLOR, text_color=(0, 0, 0), thickness=BORDER_THICKNESS)
        x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['License_Plate_Bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        car_id = df_.iloc[row_indx]['Car_ID']
        license_info = license_plate_info.get(car_id, {})
        text = license_info.get('license_plate_Dari', '\n')
        draw_border_with_text(frame, (int(x1), int(y1)), (int(x2), int(y2)), text, LICENSE_PLATE_COLOR, thickness=BORDER_THICKNESS)

# Load Results and License Plate Info
results = pd.read_csv(RESULTS_txt, delimiter='\t', encoding='utf-8')
license_plate_info = load_license_plate_info(results)

# Load Video
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Process Frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    process_frame(frame, results, license_plate_info)
    out.write(frame)

out.release()
cap.release()
