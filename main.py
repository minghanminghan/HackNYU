import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

import model
import draw

SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1440
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)


# data states
timestamp = 0
whiteboard = []
class state:

    def finger_distance(self):
        return ((self.left_index[0] - self.right_index[0])**2 + (self.left_index[1] - self.right_index[1])**2) ** 0.5

    def __init__(self):
        self.datetimes = [0]
        self.last_timestamp = 0
        self.symbols = []
        self.series = [True, True, True, True]
        self.scale_mode = None
        self.between = 0
        self.scale = 1
        self.len = 0
        self.left = 0
        self.left_index = (0, 0)
        self.right_index = (0, 0)
        self.capture_index = False # record position of index finger
        self.whiteboard = 0
        self.show_video = False
        self.show_hands = True

    def __str__(self):
        return f"start: {self.datetimes[max(0, self.left)]}, end: {self.datetimes[min(self.len-1, self.left+int(1280/self.scale))]}"

    def process_commands(self, GESTURES):
        # LEFT: control panel
        if GESTURES[1][0] == 'Closed_Fist': # toggle views (high, low, open, close)
            if timestamp - self.last_timestamp > 36:
                if GESTURES[0][0] == 'Pointing_Up': # close
                    self.series[0] = not self.series[0]
                    self.last_timestamp = timestamp
                elif GESTURES[0][0] == 'Thumb_Up': # high
                    self.series[1] = not self.series[1]
                    self.last_timestamp = timestamp
                elif GESTURES[0][0] == 'Thumb_Down': # low
                    self.series[2] = not self.series[2]
                    self.last_timestamp = timestamp
                elif GESTURES[0][0] == 'Victory': # open
                    self.series[3] = not self.series[3]
                    self.last_timestamp = timestamp
                elif GESTURES[0][0] == 'Open_Palm': # all
                    self.series = [True, True, True, True]
                    self.last_timestamp = timestamp
            
        elif GESTURES[1][0] == 'Pointing_Up': # data mode commands: zoom, shift
            if GESTURES[0][0] == 'Pointing_Up': # pinch in/out
                self.scale_mode = 'zoom'
                if self.between == 0:
                    self.between = self.finger_distance() / self.scale
            elif GESTURES[0][0] == 'Thumb_Up': # swipe
                self.scale_mode = 'scroll'
            elif GESTURES[0][0] == 'Open_Palm': # reset
                self.scale_mode = None
                self.left = 0
                self.scale = int(1280 / self.len)
                
        elif GESTURES[1][0] == 'Victory': # whiteboard
            if GESTURES[0][0] == 'Pointing_Up':
                self.capture_index = True
            elif GESTURES[0][0] == 'Victory':
                self.capture_index = False
            elif GESTURES[0][0] == 'Open_Palm':
                self.capture_index = False
                global whiteboard
                whiteboard.clear()
                self.whiteboard = 0

        elif GESTURES[1][0] == 'Open_Palm': # video mode
            if GESTURES[0][0] == 'Closed_Fist':
                self.video_mode = 0
            elif GESTURES[0][0] == 'Pointing_Up':
                self.video_mode = 1
            elif GESTURES[0][0] == 'Victory':
                self.video_mode = 2

STATE = state()
print(STATE)

help_msg = (
    # (L, R, input)
    ('Thumbs Up', 'None', 'Hide help'),
    ('Thumbs Down', 'None', 'Hide help')
)


def set_index(result): # set state of index fingers
    for i in range(len(result.handedness)):
        if result.handedness[i][0].category_name == 'Left': # Left=right, Right=left
            STATE.right_index = (int(result.hand_landmarks[i][8].x*1280), int(result.hand_landmarks[i][8].y*720))
        else:
            STATE.left_index = (int(result.hand_landmarks[i][8].x*1280), int(result.hand_landmarks[i][8].y*720))
    return None


# Display block
def event_loop(data, symbols, datetimes):
    # new data view: (GOOG Close, META Close, GOOG High, META High, GOOG Low, META Low, GOOG Open, META Open)

    global STATE, whiteboard, timestamp
    STATE.symbols = symbols
    STATE.len = data.shape[0]
    STATE.scale = 1280 / data.shape[0]
    STATE.datetimes = datetimes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # better for gesture recog
        model.recognizer.recognize_async(mp_image, timestamp)

        STATE.process_commands(model.GESTURES)


        # state subscribers
        if STATE.show_video:
            display = frame
        else:
            display = np.zeros(frame.shape)
        if model.RESULT != None:
            set_index(model.RESULT)
            if STATE.capture_index:
                whiteboard.append(STATE.right_index)
                #STATE.whiteboard += 1
            if STATE.show_hands:
                display = draw.draw_result(display, model.RESULT)
            if STATE.scale_mode == 'zoom':
                STATE.scale = STATE.finger_distance() / STATE.between
            elif STATE.scale_mode == 'scroll':
                STATE.left = int(STATE.right_index[0]/1280*STATE.len)


        # stuff that always draws
        display = draw.draw_state(display, STATE)
        display = draw.draw_gesture(display, model.GESTURES)
        display = draw.draw_data(display, data[STATE.left:STATE.left+int(1280/STATE.scale)], STATE.scale, STATE.series, STATE.symbols)
        display = draw.draw_whiteboard(display, whiteboard) # buggy but basically works


        cv2.imshow('app name', display) # change this
        timestamp += 1
        if cv2.waitKey(1) == ord('q'):
            break

    print('active frames:', timestamp)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    a1 = np.random.rand(500, 4).round(3) * 710
    print(a1.shape)
    event_loop(a1)