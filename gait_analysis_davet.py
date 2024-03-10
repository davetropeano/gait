# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:29:52 2024

@author: lexie

NOTES
- you had multiple versions of calculating a discrete derivative. The code was identical except for the variable names. Simplified to one function d_dt
- AVOID using hard coded numbers when there are enumerated values to use instead
- removed code that looked like it wasn't being used
- used a dataclass to reduce amount of repetitive code
- csv output everything in a regular fashion (even data you don't need)
- increased font size and spacing on video output
"""

from dataclasses import dataclass

import csv
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ------------------

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

# -----------------------

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
csv_writer.writerow([
     'Frame', 
     'Foot Magnitude', 'Shank Magnitude', 'Thigh Magnitude', 'Trunk Magnitude', 
     'Toe Angle', 'Ankle Angle', 'Knee Angle', 'Hip Angle',
     'Toe Linear Velocity', 'Ankle Linear Velocity', 'Knee Linear Velocity', 'Hip Linear Velocity',
     'Toe Angular Velocity', 'Ankle Angular Velocity', 'Knee Angular Velocity', 'Hip Angular Velocity',
     'Toe Angular Acceleration', 'Ankle Angular Acceleration', 'Knee Angular Acceleration', 'Hip Angular Acceleration'
])

# -----------------------

@dataclass
class LandmarkData():
    curr: any = None
    prev: any = None
    angle: float = None
    prev_angle: float = None
    linear_velocity: float = 0.0
    angular_velocity: float = None
    prev_angular_velocity: float = None
    angular_acceleration: float = 0.0

    def update(self, landmark, angle):
        self.prev = self.curr
        self.prev_angle = self.angle
        self.prev_angular_velocity = self.angular_velocity

        self.curr = landmark
        self.angle = angle
        if self.prev is not None:                
            self.linear_velocity = d_dt(self.prev.x, self.curr.x, dt)               
            self.angular_velocity = d_dt(self.prev_angle, self.angle, dt)
            if self.prev_angular_velocity is not None:
                self.angular_acceleration = d_dt(self.prev_angular_velocity, self.angular_velocity, dt)
                
toe = LandmarkData()
ankle = LandmarkData()
knee = LandmarkData()
hip = LandmarkData()

frame_count = 0
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

        toe.update(toe_landmark, 0.0)
        hip.update(hip_landmark, angle(knee_landmark, hip_landmark, shoulder_landmark))
        knee.update(knee_landmark, angle(hip_landmark, knee_landmark, ankle_landmark))
        ankle.update(ankle_landmark, angle(knee_landmark, ankle_landmark, toe_landmark))        
          
        csv_writer.writerow([
            frame_count, 
            foot_magnitude, shank_magnitude, thigh_magnitude, trunk_magnitude, 
            toe.angle, ankle.angle, knee.angle, hip.angle,
            toe.linear_velocity, ankle.linear_velocity, knee.linear_velocity, hip.linear_velocity,
            toe.angular_velocity, ankle.angular_velocity, knee.angular_velocity, hip.angular_velocity,
            toe.angular_acceleration, ankle.angular_acceleration, knee.angular_acceleration, hip.angular_acceleration,
        ])

        
        
        # Draw landmarks and connections on the frame with color-coded annotations
        mp_drawing = mp.solutions.drawing_utils
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=landmark_drawing_spec,
                                  connection_drawing_spec=connection_drawing_spec)

        # Display live measurements on the frame with color-coded font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3

        # Define colors corresponding to each measurement
        color_hip_angle = (255, 0, 0)  # Red
        color_knee_angle = (255, 0, 0)  # Red
        color_ankle_angle = (255, 0, 0)  # Red
        color_thigh_magnitude = (0, 255, 0)  # Green
        color_shank_magnitude = (0, 255, 0)  # Green
        color_foot_magnitude = (0, 255, 0)  # Green

        # davet - making all the text black seems more readable to me
        color_hip_angle = color_knee_angle = color_ankle_angle = color_thigh_magnitude = color_shank_magnitude = color_foot_magnitude = (0,0,0) # black


        # Display live measurements on the frame
        cv2.putText(frame, f'Foot Length: {foot_magnitude:.2f}', (10, 30), font, font_scale, color_foot_magnitude, font_thickness)
        cv2.putText(frame, f'Shank Length: {shank_magnitude:.2f}', (10, 60), font, font_scale, color_shank_magnitude, font_thickness)
        cv2.putText(frame, f'Thigh Length: {thigh_magnitude:.2f}', (10, 90), font, font_scale, color_thigh_magnitude, font_thickness)
        cv2.putText(frame, f'Knee Joint Angle: {knee.angle:.2f} degrees', (10, 120), font, font_scale, color_knee_angle, font_thickness)
        cv2.putText(frame, f'Hip Joint Angle: {hip.angle:.2f} degrees', (10, 150), font, font_scale, color_hip_angle, font_thickness)
        cv2.putText(frame, f'Ankle Joint Angle: {ankle.angle:.2f} degrees', (10, 180), font, font_scale, color_ankle_angle, font_thickness)


    # Write frame to the output video
    output_video.write(frame)

    cv2.imshow('Gait Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
output_video.release()
csv_file.close()
cv2.destroyAllWindows()