from mediapipe import tasks
from classes import result
import numpy as np

RESULT = result()

def update_result(result, *args):
    global RESULT
    RESULT.landmarks[0].clear() # deprecating 
    RESULT.landmarks[1].clear()
    
    # print(len(result.handedness))
    # print(result.hand_landmarks)
    # print(result)

    for i in range(len(result.handedness)):
        idx = result.handedness[i][0].category_name == 'Left' # 0 == 'Right' but frame is flipped so actually it's left
        
        if len(RESULT.landmarks[idx]) == 0: # edge case
            RESULT.stale[idx] += 1
            continue

        RESULT.landmarks[idx] = result.hand_landmarks[i] # change this

        # print(idx)
        thumb = RESULT.landmarks[idx][4]
        for i in range(2, 6): # [2, ..., 5]
            p = RESULT.landmarks[idx][i*4]
            RESULT.distances[idx][i-2] = ((thumb.x - p.x)**2 + (thumb.y - p.y)**2)**0.5 < 0.05 # distance threshold = 0.05
        RESULT.stale[idx] = 0


    # print(RESULT.landmarks)


options = tasks.vision.GestureRecognizerOptions(
    base_options=tasks.BaseOptions(model_asset_path='gesture_recognizer.task'),
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