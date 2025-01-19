import numpy as np
from ultralytics import YOLO
import re




# Utility functions
def get_car(license_plate, vehicle_track_ids, license_plate_category):
    x1, y1, x2, y2, score, class_id = license_plate
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id, license_plate_category
    return -1, -1, -1, -1, -1, license_plate_category

def write_txt(results, output_path): 
    with open(output_path, mode='w', encoding='utf-8') as f:
        f.write("Frame\tCar_ID\tCar_Bbox\tLicense_Plate_Bbox\tLicense_Plate_Bbox_Score\tDari_Text\tDari_Text_Score\tInterval_Text\tInterval_Text_Score\tEnglish_Text\tEnglish_Text_Score\n")

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'Dari_Text' in results[frame_nmr][car_id]['license_plate'].keys() and \
                   'Interval_Text' in results[frame_nmr][car_id]['license_plate'].keys() and \
                   'English_Text' in results[frame_nmr][car_id]['license_plate'].keys():

                    car_bbox = results[frame_nmr][car_id]['car']['bbox']
                    license_plate_bbox = results[frame_nmr][car_id]['license_plate']['bbox']
                    bbox_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                    dari_text = results[frame_nmr][car_id]['license_plate']['Dari_Text']
                    dari_text_score = results[frame_nmr][car_id]['license_plate']['Dari_Text_Score']
                    interval_text = results[frame_nmr][car_id]['license_plate']['Interval_Text']
                    interval_text_score = results[frame_nmr][car_id]['license_plate']['Interval_Text_Score']
                    english_text = results[frame_nmr][car_id]['license_plate']['English_Text']
                    english_text_score = results[frame_nmr][car_id]['license_plate']['English_Text_Score']

                    # Ensure empty strings are saved as placeholders
                    if not dari_text:
                        dari_text = '-'
                        dari_text_score = 0.0
                    if not interval_text:
                        interval_text = '-'
                        interval_text_score = 0.0
                    if not english_text:
                        english_text = '-'
                        english_text_score = 0.0


                    f.write(f"{frame_nmr}\t{int(car_id)}\t{car_bbox}\t{license_plate_bbox}\t{bbox_score}\t{dari_text}\t{dari_text_score}\t{interval_text}\t{interval_text_score}\t{english_text}\t{english_text_score}\n")

    


# Load a pretrained YOLOv8n-seg Segment model
license_plate_reader = YOLO('models/CNN.pt')
def read_license_plate(license_crop, class_names):
    char_detections = license_plate_reader(license_crop)[0]
    total_score = 0.0
    for char_detection in char_detections.boxes.data.tolist():
        x1, y1, x2, y2, score_class, class_id_class = char_detection
        
        if class_names == 'Dari':
            detected_classes =  Sorting(char_detections)
            translated_final = [translate_class_label(label) for label in detected_classes]
            remove_ = remove_specific_classes(translated_final) #removing English Classes
            clean = ''
            for i in remove_:
                clean += i
            total_score += score_class
            return clean, total_score
        
        elif class_names == 'Interval':
            detected_classes =  Sorting(char_detections)
            translated_final = [translate_class_label(label) for label in detected_classes]
            remove_ = remove_specific_classes(translated_final) #removing English Classes
            clean = ''
            for i in remove_:
                clean += i
            total_score += score_class
            return clean, total_score

        elif class_names == 'English':
            detected_classes =  Sorting(char_detections)
            translated_final = [translate_class_label(label) for label in detected_classes]
            clean = ''
            for i in translated_final:
                clean += i
            # Define a regular expression pattern to match Persian (Farsi) characters
            pattern = r'[آ-ی-۱-۹]'

            # Use re.sub to replace the matched pattern with an empty string
            cleaned_string = re.sub(pattern, '', clean)
            total_score += score_class

            return cleaned_string, total_score
            
    return None, None

def Sorting(char_detection):
    detected_classes = []

    # Sort the boxes by their leftmost x-coordinate (from left to right)
    sorted_boxes = sorted(char_detection.boxes, key=lambda box: box.xyxy[0, 0].item())

    # Iterate through the sorted boxes
    for box in sorted_boxes:
        class_id = char_detection.names[box.cls[0].item()]  # Get the class label
        cords = box.xyxy[0].tolist()  # Get bounding box coordinates
        cords = [round(x) for x in cords]  # Round coordinates to integers
        detected_classes.append(class_id)  # Append the class label to the final list
    return detected_classes



def remove_specific_classes(final_list):
    classes_to_remove = {
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'BLH', 'DUPL', 'GZN', 'HRT', 'KBL', 'L', 'NGR', 'PRV', 'T'
    }

    # Use a list comprehension to filter out classes not in the 'classes_to_remove' set
    filtered_final = [item for item in final_list if item not in classes_to_remove]

    # Check if the filtered list has more than 7 items
    if len(filtered_final) > 7:
        # Remove the second item (index 1) from the filtered list
        del filtered_final[1]

    return filtered_final

def translate_class_label(class_label):
    translation_dict = {
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        'B': 'B',
        'BLH': 'BLH',
        'DUPL': 'DUPL',
        'GZN': 'GZN',
        'HRT': 'HRT',
        'KBL': 'KBL',
        'L': 'L',
        'NGR': 'NGR',
        'PRV': 'PRV',
        'T': 'T',
        'ba': 'ب',
        'balkh': 'بلخ',
        'eight': '۸',
        'five': '۵',
        'four': '۴',
        'ghazni': 'غزنی',
        'herat': 'هرات',
        'kabul': 'کابل',
        'lam': 'ل',
        'mowaqat': 'موقت',
        'nengarhar': 'ننگرهار',
        'nine': '۹',
        'one': '۱',
        'seven': '۷',
        'sh': 'PRV',
        'shin': 'ش',
        'six': '۶',
        'sticker': 'استکر',
        'ta': 'ت',
        'three': '۳',
        'two': '۲'
    }

    return translation_dict.get(class_label, "")

