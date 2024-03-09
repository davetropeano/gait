# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:25:55 2023

@author: Jacob
"""
import cv2
import mediapipe as mp
import csv
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# Function to calculate angle between three points in degrees
def calculate_angle(point1, point2, point3):
    vector1 = [point2.x - point1.x, point2.y - point1.y]
    vector2 = [point3.x - point2.x, point3.y - point2.y]

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5

    if magnitude1 * magnitude2 == 0:
        return 0  # Avoid division by zero
    else:
        cosine_theta = dot_product / (magnitude1 * magnitude2)
        angle_in_degrees = abs(180 - abs(np.degrees(np.arccos(cosine_theta))))
        return angle_in_degrees

# Open a video file
video_path = r'""'
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, frames per second)
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

# Create a video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video_p22.mp4', fourcc, fps, (width, height))

# Create and open a CSV file for writing
csv_file = open('pose_measurements_p22.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Femur/Thigh Length', 'Hip Hinge Range', 'Ankle Angle Range', 'Shank Length', 'Torso Length', 'Left Knee X - Left Toe X', 'Knee Over Toe Translation', 'Hip Hinge Angle Diff', 'Ankle Angle Diff', 'Torso to Femur Ratio', 'Knee Flexion Angle', 'Femur Parallel'])

frame_count = 0
hip_hinge_angles = []  # List to store hip hinge angles
ankle_angles = []  # List to store ankle angles
torso_femur_ratios = []  # List to store torso to femur ratios

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        hip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee_landmark = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        toe_landmark = landmarks[31]  # Index for LEFT_BIG_TOE
        shoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        femur_length = calculate_distance(hip_landmark, knee_landmark)
        hip_hinge_range = calculate_angle(knee_landmark, hip_landmark, shoulder_landmark)
        ankle_angle_range = calculate_angle(knee_landmark, ankle_landmark, toe_landmark)
        shank_length = calculate_distance(knee_landmark, ankle_landmark)
        torso_length = calculate_distance(hip_landmark, shoulder_landmark)

        left_knee_x = knee_landmark.x
        left_toe_x = toe_landmark.x
        left_knee_toe_diff = left_knee_x - left_toe_x

        # Add Knee Over Toe Translation column
        knee_over_toe_translation = True if left_knee_toe_diff < 0 else False

        # Append hip hinge and ankle angles to the lists
        hip_hinge_angles.append(hip_hinge_range)
        ankle_angles.append(ankle_angle_range)

        # Calculate torso to femur ratio
        torso_femur_ratio = torso_length / femur_length
        torso_femur_ratios.append(torso_femur_ratio)

        # Calculate knee flexion angle
        knee_flexion_angle = calculate_angle(hip_landmark, knee_landmark, ankle_landmark)

        # Check if femur is parallel to the floor
        femur_parallel = True if knee_flexion_angle < 65 else False

        csv_writer.writerow([frame_count, femur_length, hip_hinge_range, ankle_angle_range, shank_length, torso_length,
                             left_knee_toe_diff, knee_over_toe_translation, None, None, torso_femur_ratio, knee_flexion_angle, femur_parallel])

        # Draw landmarks and connections on the frame with color-coded annotations
        mp_drawing = mp.solutions.drawing_utils
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=landmark_drawing_spec,
                                  connection_drawing_spec=connection_drawing_spec)

        # Display live measurements on the frame with color-coded font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Define colors corresponding to each measurement
        color_femur = (0, 255, 0)  # Green
        color_hip_hinge = (255, 0, 0)  # Red
        color_ankle_angle = (0, 0, 255)  # Blue
        color_shank = (255, 255, 0)  # Yellow
        color_torso = (255, 165, 0)  # Orange
        color_knee_toe_diff = (255, 0, 255)  # Magenta for knee-toe difference

        # Define colors corresponding to each measurement (continued)
        color_torso_femur_ratio = (0, 255, 255)  # Cyan
        color_femur_parallel = (0, 255, 0)  # Green for parallel, Red for not parallel

        # Display live measurements on the frame
        cv2.putText(frame, f'Femur Length: {femur_length:.2f}', (10, 30), font, font_scale, color_femur, font_thickness)
        cv2.putText(frame, f'Hip Hinge Range: {hip_hinge_range:.2f} degrees', (10, 60), font, font_scale, color_hip_hinge, font_thickness)
        cv2.putText(frame, f'Ankle Angle Range: {ankle_angle_range:.2f} degrees', (10, 90), font, font_scale, color_ankle_angle, font_thickness)
        cv2.putText(frame, f'Shank Length: {shank_length:.2f}', (10, 120), font, font_scale, color_shank, font_thickness)
        cv2.putText(frame, f'Torso Length: {torso_length:.2f}', (10, 150), font, font_scale, color_torso, font_thickness)
        cv2.putText(frame, f'Left Knee X - Left Toe X: {left_knee_toe_diff:.2f}', (10, 180), font, font_scale, color_knee_toe_diff, font_thickness)
        cv2.putText(frame, f'Knee Over Toe Translation: {knee_over_toe_translation}', (10, 210), font, font_scale, color_knee_toe_diff, font_thickness)
        cv2.putText(frame, f'Torso to Femur Ratio: {torso_femur_ratio:.2f}', (10, 240), font, font_scale, color_torso_femur_ratio, font_thickness)
        cv2.putText(frame, f'Knee Flexion Angle: {knee_flexion_angle:.2f} degrees', (10, 270), font, font_scale, color_torso_femur_ratio, font_thickness)
        cv2.putText(frame, f'Femur Parallel: {femur_parallel}', (10, 300), font, font_scale, color_femur_parallel, font_thickness)

    # Write frame to the output video
    output_video.write(frame)

    cv2.imshow('Pose Measurements', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Calculate the difference between the max and min values of hip hinge and ankle angles
hip_hinge_diff = max(hip_hinge_angles) - min(hip_hinge_angles)
ankle_angle_diff = max(ankle_angles) - min(ankle_angles)

# Calculate the average torso to femur ratio
average_torso_femur_ratio = sum(torso_femur_ratios) / len(torso_femur_ratios)

# Update the CSV file with the calculated differences and average torso to femur ratio
csv_writer.writerow([None, None, None, None, None, None, None, None, hip_hinge_diff, ankle_angle_diff, average_torso_femur_ratio, None, None])

# Release resources
cap.release()
output_video.release()
csv_file.close()
cv2.destroyAllWindows()


