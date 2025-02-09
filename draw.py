import cv2
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

WIDTH = 1800
HEIGHT= 720
MARGIN = 5
COLORS = (
    ((252, 246, 238), (237, 202, 151), (219, 152, 52), (139, 93, 24)), # close
    ((0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)), # high
    ((0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)), # low
    ((252, 246, 238), (237, 202, 151), (219, 152, 52), (139, 93, 24)) # open
)

def draw_help(frame, help_msg):
    y_diff = 20
    y_top = 20
    for line in help_msg:
        cv2.putText(frame, f"{line[0]} + {line[1]}: {line[2]}", (100, y_top), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
        y_top += y_diff
    return frame

def draw_whiteboard(frame, points:list):
    # draw points from data to frame
    for p in points:
        cv2.circle(frame, p, 1, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


# write cv2 to frame
def draw_data(frame, data:np.ndarray, scale:float, series:list, symbols:list):
    sum_series = sum(series)
    if data.size > 0:
        np_min, np_max = data.min(), data.max()
    else:
        np_min = np_max = 0
    np_len = data.shape[0]
    
    # resize data to fit frame with margin
    data = HEIGHT - 4*MARGIN - (data - np_min)*650 / (np_max - np_min) # leaves some vertical margin

    for n in range(len(symbols)):
        colors = [[j/255 for j in i[n]] for i in COLORS]
        #print(colors)
        subset = data[:, [len(symbols)*m+n for m in range(4)]]
        if subset.size < 400:
            LINE_THICKNESS = 3
        elif subset.size < 800:
            LINE_THICKNESS = 2
        else:
            LINE_THICKNESS = 1
        for i in range(np_len):
            for j in range(4):
                if series[j]:
                    cv2.circle(frame, (MARGIN+int(i*scale), int(subset[i, j])), LINE_THICKNESS, colors[j], -1, cv2.LINE_AA)
        if sum_series > 1:
            for i in range(np_len):
                cv2.line(frame, (MARGIN+int(i*scale), int(np.max(subset[i, series]))), (MARGIN+int(i*scale), int(np.min(subset[i, series]))), colors[0], 1, cv2.LINE_AA)
        elif sum_series == 1: # implement arima?
            j = series.index(True)
            for i in range(np_len):
                cv2.line(frame, (MARGIN+int(i*scale), int(subset[i, j])), (MARGIN+int((i-1)*scale), int(subset[i-1, j])), colors[j], 1, cv2.LINE_AA)
        cv2.putText(frame, symbols[n], (MARGIN, int(subset[0, 0])-20), cv2.FONT_HERSHEY_DUPLEX, 0.75, colors[0], 1, cv2.LINE_AA)

    # put description in corner
    #cv2.putText(frame, f"y_min: {np_min}", (MARGIN, 60), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    #cv2.putText(frame, f"y_max: {np_max}", (MARGIN, 80), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    #cv2.putText(frame, f"x_min: {x_min}", (MARGIN, 100), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    #cv2.putText(frame, f"x_max: {x_min+np_len}", (MARGIN, 120), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    
    return frame




GESTURE_FONT_SIZE = 0.5
GESTURE_FONT_THICKNESS = 1
GESTURE_TEXT_COLOR = (230, 230, 230)
def draw_gesture(frame, gesture):
    cv2.putText(frame, 
                #f"LEFT: {' '.join(gesture[1][0].split('_'))}",
                f"LEFT:  {gesture[1][0]}",
                (MARGIN, 20), cv2.FONT_HERSHEY_DUPLEX,
                GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR,
                GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    cv2.putText(frame, 
                #f"RIGHT: {gesture[0][0]}",
                f"RIGHT: {gesture[0][0]}",
                (MARGIN, 40), cv2.FONT_HERSHEY_DUPLEX,
                GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR,
                GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    return frame

def draw_state(frame, state):
    y_diff = 20
    y_top = 720 - MARGIN # 1080 = main.SCREEN_HEIGHT
    cv2.putText(frame, str(state),
        (MARGIN, y_top), cv2.FONT_HERSHEY_DUPLEX,
        GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR,
        GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    return frame

def draw_result(frame:np.ndarray, result):
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]


        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
    
    return frame