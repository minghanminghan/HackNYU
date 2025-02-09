import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


GESTURES = [
    ('None', 1),
    ('None', 1)
]
RESULT = None


def update_result(result, *args):
    global RESULT, GESTURES
    RESULT = result
    #print([(result.handedness[i], result.gestures[i]) for i in range(len(result.handedness))])

    for i in range(len(result.handedness)):
        if result.handedness[i][0].category_name == 'Left':
            GESTURES[0] = (result.gestures[i][0].category_name, result.gestures[i][0].score)
        elif result.handedness[i][0].category_name == 'Right':
            if result.gestures[i][0].category_name != 'None': # fix
                GESTURES[1] = (result.gestures[i][0].category_name, result.gestures[i][0].score)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result
)

recognizer = GestureRecognizer.create_from_options(options)