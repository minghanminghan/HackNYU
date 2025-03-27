import cv2
import mediapipe as mp
import numpy as np
import time

from classes import DATA, STATE
from model import recognizer, RESULT
from draw import cap, window_name, WIN_WIDTH, WIN_HEIGHT, CAP_WIDTH, CAP_HEIGHT, draw_gestures, draw_landmarks, draw_data, draw_wheel, draw_toggle

'''
altered data view:
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
    if STATE.hand_mode != 4: # toggle on
        STATE.hand_mode = 4
    else: # toggle off: get index distance
        # not fixed scaling (is this bad)
        distance = int(6 * ((RESULT.landmarks[0][8].x - RESULT.landmarks[1][8].x)**2 + (RESULT.landmarks[0][8].y - RESULT.landmarks[1][8].y)**2 )**0.5)
        STATE.hand_mode = min(3, max(0, distance))


def process_toggle():
    if RESULT.distances[0][0]: # prev
        STATE.toggle_index = (STATE.toggle_index - 1) % DATA.shape[2]
    
    elif RESULT.distances[0][1]: # next
        STATE.toggle_index = (STATE.toggle_index + 1) % DATA.shape[2]
    
    elif RESULT.distances[1][0]: # toggle cur
        STATE.display_symbols[STATE.toggle_index] = not STATE.display_symbols[STATE.toggle_index]
        STATE.display_names = [DATA.series_names[1][i] for i in range(len(DATA.series_names[1])) if STATE.display_symbols[i]]

    elif RESULT.distances[1][1]: # toggle arima for cur
        print('show arima')

    elif RESULT.distances[1][2]: # toggle all
        val = not STATE.display_symbols[STATE.toggle_index]
        STATE.display_symbols = [val for _ in STATE.display_symbols]
        STATE.display_names = [DATA.series_names[1][i] for i in range(len(DATA.series_names[1])) if STATE.display_symbols[i]]
    
    else:
        return
    
    STATE.cooldown = 24


# Display block
def event_loop():
    start = int(time.time()*1000)
    while cap.isOpened():
        timestamp = int(time.time()*1000)
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIN_WIDTH, WIN_HEIGHT))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # better for gesture recog
        recognizer.recognize_async(mp_image, timestamp)

        if STATE.cooldown == 0:
            # thumb-pinky
            if RESULT.distances[1][3] and (len(RESULT.landmarks[1]) == len(RESULT.landmarks[1]) > 0): # toggle
                process_touch()
                STATE.cooldown = 24
            elif STATE.hand_mode == 3: # toggling series
                process_toggle()

        display = np.zeros(frame.shape)
        #always rendered
        draw_gestures(display)
        draw_landmarks(display)

        # not menu select
        if STATE.hand_mode != 4:
            draw_data(display)
            # toggle
            if STATE.hand_mode == 3:
                draw_toggle(display)
        # both hands present
        elif len(RESULT.landmarks[0]) == len(RESULT.landmarks[1]) > 0: # selecting: requires both hands in frame
            draw_wheel(display)
        

        # if VIDEO_MODE == 1: # only hands
        #     display = draw.draw_result(display)
        # elif VIDEO_MODE == 2: # only video
        #     display = frame
        # else: # all
        #     display = frame
        #     display = draw.draw_result(display)
        
        # cleanup
        if STATE.cooldown > 0:
            STATE.cooldown -= 1


        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        cv2.imshow(window_name, display)

    print(f'program duration: {timestamp - start} ms')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    a1 = np.random.rand(500, 4).round(3) * 710
    print(a1.shape)
    event_loop(a1)