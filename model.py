from mediapipe import tasks
from classes import result
import numpy as np

RESULT = result()

def update_result(result, *args):

    for i in range(len(result.handedness)):
        idx = int(result.handedness[i][0].category_name == 'Left') # 0 == 'Right' but frame is flipped so actually it's left

        if len(result.hand_landmarks[i]) == 0: # edge case
            RESULT.fresh[idx] += 1
            continue

        RESULT.fresh[idx] = 0

        for j in range(5):
            RESULT.landmarks[idx][j] = (result.hand_landmarks[i][4*j+4].x, result.hand_landmarks[i][4*j+4].y) # [[x, y], ...]

        thumb = RESULT.landmarks[idx][0]
        tips = RESULT.landmarks[idx][1:]
        RESULT.distances[idx] = ((tips - thumb)**2).sum(axis=1) < 0.003

options = tasks.vision.GestureRecognizerOptions(
    base_options=tasks.BaseOptions(model_asset_path='gesture_recognizer.task'), # switch out for hand_landmarker
    running_mode=tasks.vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5
)


recognizer = tasks.vision.GestureRecognizer.create_from_options(options)


'''
GestureRecognizerResult(
    gestures=[[Category(index=-1, score=0.9859821796417236, display_name='', category_name='None')]], 
    handedness=[[Category(index=0, score=0.6829532384872437, display_name='Right', category_name='Right')]], 
    hand_landmarks=[[
        NormalizedLandmark(x=0.04376596957445145, y=0.9778777360916138, z=1.7586305034456018e-07, visibility=0.0, presence=0.0), 
        ...
    ]], 
    hand_world_landmarks=[[
        Landmark(x=-0.00787736102938652, y=0.073121078312397, z=0.02462412603199482, visibility=0.0, presence=0.0),
        ...
    ]])
'''