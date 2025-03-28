import cv2
import mediapipe as mp
import numpy as np
import time

from classes import DATA, STATE
from model import recognizer, RESULT
from draw import cap, window_name, window_w, window_h, draw_const, draw_data, draw_wheel, draw_whiteboard

'''
DATA.values:
[ date_1:
    [ close:  [ticker_1, ..., ticker_n],
      high:   [ticker_1, ..., ticker_n],
      low:    [ticker_1, ..., ticker_n],
      open:   [ticker_1, ..., ticker_n],
      volume: [ticker_1, ..., ticker_n] ],
  date_2:
    [ close: ..., high: ..., low: ..., open: ..., volume: ...],
  ...
  date_n: ...
]
'''


def process_touch():
    if STATE.index != 4: # toggle on
        STATE.index = 4
        STATE.subindex = 0
    else: # toggle off: get index distance
        if RESULT.fresh[0] < 6 and RESULT.fresh[1] < 6: # left = 0
            STATE.index = min(3, max(0, STATE.subindex))
            STATE.subindex = 0
    STATE.cooldown = 24


def process_scroll(): # [none, resize, scroll]
    # next
    if RESULT.distances[0][0]: # left thumb-index
        STATE.subindex = (STATE.subindex + 1) % 3
        STATE.cooldown = 24
    # prev
    elif RESULT.distances[0][3]: # left thumb-pinky
        STATE.subindex = (STATE.subindex - 1) % 3
        STATE.cooldown = 24


def process_whiteboard(): # add index finger to whiteboard
    # toggle
    if RESULT.distances[0][0]: # left thumb-index
        STATE.subindex = (STATE.subindex + 1) % 2
        STATE.cooldown = 24
    # cls
    elif RESULT.distances[0][3]: # left thumb-pinky
        STATE.subindex = 0
        STATE.whiteboard.clear()
        STATE.cooldown = 24


def process_toggle():
    if RESULT.distances[0][0]: # prev (left index)
        STATE.subindex = (STATE.subindex - 1) % DATA.shape[2]
    
    elif RESULT.distances[0][3]: # next (left pinky)
        STATE.subindex = (STATE.subindex + 1) % DATA.shape[2]
    
    elif RESULT.distances[1][0]: # toggle cur (right index)
        STATE.display_symbols[STATE.subindex] = not STATE.display_symbols[STATE.subindex]
        STATE.display_names = [DATA.series_names[1][i] for i in range(len(DATA.series_names[1])) if STATE.display_symbols[i]]

    elif RESULT.distances[1][1]: # toggle all (right middle)
        val = not STATE.display_symbols[STATE.subindex]
        STATE.display_symbols = [val for _ in STATE.display_symbols]
        STATE.display_names = [DATA.series_names[1][i] for i in range(len(DATA.series_names[1])) if STATE.display_symbols[i]]
    
    else:
        return
    
    STATE.cooldown = 24


def event_loop():
    start = int(time.time()*1000)
    timestamp = int(time.time()*1000)
    while True:
        # 0. capture video
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (window_w, window_h))

        # 1. mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # better for gesture recog
        timestamp = int(time.time()*1000)
        recognizer.recognize_async(mp_image, timestamp)

        # 1. logic
        if STATE.cooldown == 0:
            if RESULT.distances[1][3] and RESULT.fresh[1] < 6: # toggle
                process_touch()
            elif STATE.index == 0: # scroll
                process_scroll()
            elif STATE.index == 1: # whiteboard
                process_whiteboard()
            elif STATE.index == 2: # toggle series
                process_toggle()

        # whiteboard
        if  STATE.index == 1 and STATE.subindex == 1 and RESULT.fresh[1] < 6:
            STATE.whiteboard.append((window_w * RESULT.landmarks[1][1][0], window_h * RESULT.landmarks[1][1][1]))

        # 2. render
        display = np.zeros(frame.shape)

        #always rendered
        draw_const(display)
        draw_whiteboard(display) # whiteboard
        
        if STATE.index == 4:
            if RESULT.fresh[0] < 6 and RESULT.fresh[1] < 6:
                distance = ((RESULT.landmarks[0][1][0] - RESULT.landmarks[1][1][0])**2 + (RESULT.landmarks[0][1][1] - RESULT.landmarks[1][1][1])**2 )**0.5
                STATE.subindex = min(3, max(0, int(7 * distance)))
                draw_wheel(display)
        else:
            draw_data(display)
            if STATE.index == 0 and RESULT.fresh[1] < 6:
                if STATE.subindex == 1: # resize
                    idx = (min(0.75, max(0.25, RESULT.landmarks[1][1][0])) - 0.25) * 2 # right index finger: [0.25, 0.75] -> [0, len(series)]
                    idx = min(DATA.shape[0] - STATE.left_index, max(STATE.left_index, int(idx * DATA.shape[0])))
                    STATE.display_length = idx
                elif STATE.subindex == 2: # scroll
                    idx = (min(0.75, max(0.25, RESULT.landmarks[1][1][0])) - 0.25) * 2 # right index finger: [0.25, 0.75] -> [0, len(series)]
                    idx = min(DATA.shape[0] - STATE.display_length, max(0, int(idx * DATA.shape[0])))
                    STATE.left_index = idx

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        cv2.imshow(window_name, display)

        # 3. cleanup
        if STATE.cooldown > 0:
            STATE.cooldown -= 1


    print(f'program duration: {timestamp - start} ms')
    cap.release()
    cv2.destroyAllWindows()