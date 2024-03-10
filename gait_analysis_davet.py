# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:29:52 2024

@author: lexie

NOTES
- using numpy for as much math as possible. Or do without numpy
- you had multiple versions of calculating a discrete derivative. The code was identical except for the variable names. Simplified to one function d_dt
- AVOID using hard coded numbers when there are enumerated values to use instead
"""

import csv
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# calculate Euclidean distance between two points
def distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# calculate angle between three points in degrees
def angle(point1, point2, point3):
    vector1 = [point2.x - point1.x, point2.y - point1.y]
    vector2 = [point3.x - point2.x, point3.y - point2.y]

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5

    if (magnitude1 * magnitude2) == 0:
        return 0  # Avoid division by zero
    else:
        cosine_theta = dot_product / (magnitude1 * magnitude2)
        angle_in_degrees = abs(180 - abs(np.degrees(np.arccos(cosine_theta))))
        return angle_in_degrees

def d_dt(initial, final, dt):
    if dt == 0:
        return 0
    
    return (final - initial) / dt

# Open a video file
video_path = "videos/TS1fl.mov"
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps if fps else 1

# Create a video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video_p22.mp4', fourcc, fps, (width, height))

# Create and open a CSV file for writing
csv_file = open('pose_measurements_p22.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Foot Magnitude', 'Shank Magnitude', 'Thigh Magnitude', 'Trunk Magnitude', 'Ankle Plantar/Dorsi-Flexion Angle', 'Knee Flexion/Extension Angle', 'Hip Flexion/Extension'])

frame_count = 0
ankle_angles = []           
knee_angles = []            
hip_angles = []            

hip_lin_velocities = []     
knee_lin_velocities = []   
ankle_lin_velocities = []   
toe_lin_velocities = []   

hip_ang_velocities = []   
knee_ang_velocities = []  
ankle_ang_velocities = []   

hip_ang_accelerations = []  
knee_ang_accelerations = []  
ankle_ang_accelerations = []  

prev_toe_landmark = None      
prev_ankle_landmark = None    
prev_knee_landmark = None     
prev_hip_landmark = None      

linear_velocity_hip = None    
linear_velocity_knee = None   
linear_velocity_ankle = None  
linear_velocity_toe = None    

angular_velocity_hip = None   
angular_velocity_knee = None  
angular_velocity_ankle = None 

angular_accel_hip = None      
angular_accel_knee = None     
angular_accel_ankle = None    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        toe_landmark = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]  
        ankle_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        knee_landmark = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        hip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        shoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate Segment Lengths 
        foot_magnitude = distance(toe_landmark, ankle_landmark)
        shank_magnitude = distance(knee_landmark, ankle_landmark)
        thigh_magnitude = distance(hip_landmark, knee_landmark)
        trunk_magnitude = distance(shoulder_landmark, hip_landmark)
        
        # Calculate Segment Angles 
        ankle_angle = angle(knee_landmark, ankle_landmark, toe_landmark)
        knee_angle = angle(hip_landmark, knee_landmark, ankle_landmark) 
        hip_angle = angle(knee_landmark, hip_landmark, shoulder_landmark)

    
        hip_angles.append(hip_angle)
        knee_angles.append(knee_angle)
        ankle_angles.append(ankle_angle)

        # Hip Data Collection
        if prev_hip_landmark is not None:                
            linear_velocity_hip = d_dt(prev_hip_landmark.x, hip_landmark.x, dt) 
            hip_lin_velocities.append(linear_velocity_hip)
                
            angular_velocity_hip = d_dt(prev_hip_angle, hip_angle, dt) 
            hip_ang_velocities.append(angular_velocity_hip)
        
            if len(hip_ang_velocities) >= 2:
                # davet - this seems wrong. Why index by [-2]? Don't you just want the previous value?
                angular_accel_hip = d_dt(hip_ang_velocities[-2], angular_velocity_hip, dt) 
                hip_ang_accelerations.append(angular_accel_hip)
                
        prev_hip_landmark = hip_landmark
        prev_hip_angle = hip_angle
        
        # Knee Data Collection
        if prev_knee_landmark is not None:
            linear_velocity_knee = d_dt(prev_knee_landmark.x, knee_landmark.x, dt)
            knee_lin_velocities.append(linear_velocity_knee)
                
            angular_velocity_knee = d_dt(prev_knee_angle, knee_angle, dt)
            knee_ang_velocities.append(angular_velocity_knee)
                
            if len(knee_ang_velocities) >= 2:
                # davet - this seems wrong. Why index by [-2]? Don't you just want the previous value?
                angular_accel_knee = d_dt(knee_ang_velocities[-2], angular_velocity_knee, dt)
                knee_ang_accelerations.append(angular_accel_knee)
                
        prev_knee_landmark = knee_landmark
        prev_knee_angle = knee_angle
                
        # Ankle Data Collection                
        if prev_ankle_landmark is not None:
            linear_velocity_ankle = d_dt(prev_ankle_landmark.x, ankle_landmark.x, dt)
            ankle_lin_velocities.append(linear_velocity_ankle)
                
            angular_velocity_ankle = d_dt(prev_ankle_angle, ankle_angle, dt)
            ankle_ang_velocities.append(angular_velocity_ankle)
                
            if len(ankle_ang_velocities) >= 2:
                # davet - this seems wrong. Why index by [-2]? Don't you just want the previous value?
                angular_accel_ankle = d_dt(ankle_ang_velocities[-2], angular_velocity_ankle, dt)
                ankle_ang_accelerations.append(angular_accel_ankle)

        prev_ankle_landmark = ankle_landmark
        prev_ankle_angle = ankle_angle
          
        # Toe Data Collection                
        if prev_toe_landmark is not None:
            linear_velocity_toe = d_dt(prev_toe_landmark.x, toe_landmark.x, dt)
            toe_lin_velocities.append(linear_velocity_toe)
        
        prev_toe_landmark = toe_landmark

        
        csv_writer.writerow([frame_count, foot_magnitude, shank_magnitude, thigh_magnitude, trunk_magnitude, None, None, ankle_angle, knee_angle, hip_angle, 
                             None, None, linear_velocity_hip, linear_velocity_knee, linear_velocity_toe, angular_velocity_hip, angular_velocity_knee, 
                             angular_velocity_ankle, None, None, angular_accel_hip, angular_accel_knee, angular_accel_ankle])

        
        
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
        color_hip_angle = (255, 0, 0)  # Red
        color_knee_angle = (255, 0, 0)  # Red
        color_ankle_angle = (255, 0, 0)  # Red
        color_thigh_magnitude = (0, 255, 0)  # Green
        color_shank_magnitude = (0, 255, 0)  # Green
        color_foot_magnitude = (0, 255, 0)  # Green


        # Display live measurements on the frame
        cv2.putText(frame, f'Foot Length: {foot_magnitude:.2f}', (10, 30), font, font_scale, color_foot_magnitude, font_thickness)
        cv2.putText(frame, f'Shank Length: {shank_magnitude:.2f}', (10, 120), font, font_scale, color_shank_magnitude, font_thickness)
        cv2.putText(frame, f'Thigh Length: {thigh_magnitude:.2f}', (10, 120), font, font_scale, color_thigh_magnitude, font_thickness)
        cv2.putText(frame, f'Knee Joint Angle: {knee_angle:.2f} degrees', (10, 270), font, font_scale, color_knee_angle, font_thickness)
        cv2.putText(frame, f'Hip Joint Angle: {hip_angle:.2f} degrees', (10, 60), font, font_scale, color_hip_angle, font_thickness)
        cv2.putText(frame, f'Ankle Joint Angle: {ankle_angle:.2f} degrees', (10, 90), font, font_scale, color_ankle_angle, font_thickness)


    # Write frame to the output video
    output_video.write(frame)

    cv2.imshow('Gait Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Calculate the difference between the max and min values of hip hinge and ankle angles
max_hip_angle = max(hip_angles) 
max_knee_angle = max(knee_angles)
max_ankle_angle = max(ankle_angles) 


# Release resources
cap.release()
output_video.release()
csv_file.close()
cv2.destroyAllWindows()