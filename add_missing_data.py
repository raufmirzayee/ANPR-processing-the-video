import numpy as np
from scipy.interpolate import interp1d
def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['Frame']) for row in data])
    car_ids = np.array([int(float(row['Car_ID'])) for row in data])
    car_bboxes = np.array([list(map(float, row['Car_Bbox'][1:-1].replace(',','').split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['License_Plate_Bbox'][1:-1].replace(',','').split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        frame_numbers_ = [int(p['Frame']) for p in data if int(float(p['Car_ID'])) == car_id]

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['Frame'] = str(frame_number)
            row['Car_ID'] = str(car_id)
            row['Car_Bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['License_Plate_Bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if frame_number not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['License_Plate_Bbox_Score'] = 0.0
                row['Dari_Text'] = '-'
                row['Dari_Text_Score'] = 0.0
                row['Interval_Text'] = '-'
                row['Interval_Text_Score'] = 0.0
                row['English_Text'] = '-'
                row['English_Text_Score'] = 0.0
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['Frame']) == frame_number and int(float(p['Car_ID'])) == car_id][0]
                row['License Plate Bbox_Score'] = original_row['License_Plate_Bbox_Score'] if 'License_Plate_Bbox_Score' in original_row else 0.0
                row['Dari_Text'] = original_row['Dari_Text'] if 'Dari_Text' in original_row else '-'
                row['Dari_Text_Score'] = original_row['Dari_Text_Score'] if 'Dari_Text_Score' in original_row else 0.0
                row['Interval_Text'] = original_row['Interval_Text'] if 'Interval_Text' in original_row else '-'
                row['Interval_Text_Score'] = original_row['Interval_Text_Score'] if 'Interval_Text_Score' in original_row else 0.0
                row['English_Text'] = original_row['English_Text'] if 'English_Text' in original_row else '-'
                row['English_Text_Score'] = original_row['English_Text_Score'] if 'English_Text_Score' in original_row else 0.0



            interpolated_data.append(row)

    return interpolated_data



with open('new.txt', 'r', encoding='utf-8') as file:
    data_lines = file.readlines()

# Parse the lines into dictionaries
data = []
header = data_lines[0].strip().split('\t')  # Split the header line
for line in data_lines[1:]:  # Skip the header line
    line = line.strip().split('\t')
    if len(line) == len(header):  # Check if the line has the same number of values as the header
        row = {}
        for i in range(len(header)):
            row[header[i]] = line[i]
        data.append(row)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new TXT file
with open('new1.txt', 'w', encoding='utf-8') as file:
    # Write the header line
    file.write('\t'.join(header) + '\n')
    for row in interpolated_data:
        row_str = '\t'.join(str(row.get(field, 0)) for field in header)
        file.write(row_str + '\n')
