import cv2
import numpy as np

from model import RESULT
from classes import DATA, STATE

cap = cv2.VideoCapture(0)
window_name = 'app name'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
_, _, WIN_WIDTH, WIN_HEIGHT = cv2.getWindowImageRect(window_name)
CAP_WIDTH, CAP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

MARGIN = 5
COLORS = ((252, 246, 238), (0, 255, 0), (0, 0, 255), (252, 246, 238)) # close, high, low, open

# TODO: turn help into a cli message instead of gui display
# help_msg = (
#     ('"Left Hand"', '"Right Hand"', '"Action"'),
#     ('Closed Fist', 'Pointing Up', 'Toggle Close'),
#     ('Closed Fist', 'Victory', 'Toggle Open'),
#     ('Closed Fist', 'Thumb Up', 'Toggle High'),
#     ('Closed Fist', 'Thumb Down', 'Toggle Low'),
#     ('Closed Fist', 'Open Palm', 'Toggle All'),
#     ('Pointing Up', 'Pointing Up', 'Zoom In/Out'),
#     ('Pointing Up', 'Pointing Up', 'Reset'),
#     ('Victory', 'Pointing Up', 'Start Drawing'),
#     ('Victory', 'Pointing Up', 'Stop Drawing'),
#     ('Victory', 'Pointing Up', 'Clear Drawing'),
#     ('Open Palm', 'Closed Fist', 'Default View'),
#     ('Open Palm', 'Closed Fist', 'Data View'),
#     ('Open Palm', 'Closed Fist', 'Camera View'),
# )


# def draw_help(frame):
#     frame = np.zeros(frame.shape)
#     y_diff = 25
#     y_top = 250
#     for line in help_msg:
#         cv2.putText(frame, f"{line[0]} + {line[1]}: {line[2]}", (400, y_top), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
#         y_top += y_diff
#     cv2.putText(frame, "Instructions", (485, 200), cv2.FONT_HERSHEY_DUPLEX, 0.75, GESTURE_TEXT_COLOR, 1, cv2.LINE_AA)
#     return frame

# TODO: turn points into lines
def draw_whiteboard(frame, points:list):
    # draw points from data to frame
    for p in points:
        cv2.circle(frame, p, 2, (255, 255, 255), -1, cv2.LINE_AA)


# 1. scrolling
# 2. whiteboard
# 3. toggle series
def draw_toggle(frame): # draw relevant context
    cv2.putText(frame, f'Toggle index: {STATE.toggle_index}', (0, 6*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Series: {STATE.display_symbols}', (0, 7*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)


# 4. menu select
def draw_wheel(frame):
    cv2.line(frame, 
        (int(WIN_WIDTH * RESULT.landmarks[0][8].x), int(WIN_HEIGHT * RESULT.landmarks[0][8].y)),
        (int(WIN_WIDTH * RESULT.landmarks[1][8].x), int(WIN_HEIGHT * RESULT.landmarks[1][8].y)),
        (255, 255, 255), 1
    )
    midpoint = (int(WIN_WIDTH * (RESULT.landmarks[0][8].x + RESULT.landmarks[1][8].x) / 2), int(WIN_HEIGHT * (RESULT.landmarks[0][8].y + RESULT.landmarks[1][8].y) / 2))
    
    # distance & min/max logic is being reused here
    distance = int(6 * ((RESULT.landmarks[0][8].x - RESULT.landmarks[1][8].x)**2 + (RESULT.landmarks[0][8].y - RESULT.landmarks[1][8].y)**2 )**0.5)
    cv2.circle(frame, (midpoint[0], midpoint[1]), 5, (255, 255, 255), -1)
    cv2.putText(frame, f'{HAND_MODE[min(3, max(0, distance))]}', (midpoint[0] - 25, midpoint[1] - 10), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Index distance: {distance}', (0, 6*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)



MARGIN = 10
FRAME_HEIGHT = WIN_HEIGHT - 2*MARGIN
FRAME_WIDTH = WIN_WIDTH #- 2*MARGIN
def draw_data(frame):
    # 1. slice subset from DATA, STATE
    # print(STATE.display_length)
    # print(STATE.left_index)
    subset: np.ndarray = DATA.values[STATE.left_index: STATE.display_length + STATE.left_index]
    #subset = subset[:, STATE.display_symbols[1]] # unimportant for now
    subset: np.ndarray = subset[:, :, STATE.display_symbols] # outer index
    
    if subset.size == 0:
        return
    
    # 2. draw subset
    # 2.1 scale to frame
        # horizontal: width / STATE.display_length
        # vertical: height / max-min
    subset_max = subset.max()
    subset: np.ndarray = (FRAME_HEIGHT - subset*FRAME_HEIGHT/subset_max + MARGIN).astype(int) # not sure if this is right

    # 2.2 draw points and lines (parallelize?)
    x_step = FRAME_WIDTH / STATE.display_length
    for datetime in range(subset.shape[0]):
        x = int(datetime*x_step)
        for series_outer in range(subset.shape[2]):
            cv2.line(frame, (x, subset[datetime, 2, series_outer]), (x, subset[datetime, 1, series_outer]), (255, 255, 255), 1)
            for series_inner in range(subset.shape[1]):
                # x: datetime * scaling, y: subset[datetime, series_inner, series_outer]
                cv2.circle(frame, (x, subset[datetime, series_inner, series_outer]), 1, COLORS[series_inner], -1)

    # 3. label series
    for i in range(len(STATE.display_names)):
            # maybe draw a square under the text
            cv2.putText(frame, f'{STATE.display_names[i]}', (0, subset[0, 1, i]), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)

    # 4. put dates
    cv2.putText(frame, f'start: {DATA.datetimes[STATE.left_index]}, end: {DATA.datetimes[min(STATE.left_index + STATE.display_length, DATA.shape[0]-1)]}', (WIN_WIDTH-290, MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)


# TODO: deprecate
# def draw_data(frame, *args):
#     sum_series = sum(series)
#     if data.size > 0:
#         np_min, np_max = data.min(), data.max()
#     else:
#         np_min = np_max = 0
#     np_len = data.shape[0]
    
#     # resize data to fit frame with margin
#     data = HEIGHT - 4*MARGIN - (data - np_min)*650 / (np_max - np_min) # leaves some vertical margin

#     for n in range(len(symbols)):
#         colors = [[j/255 for j in i[n]] for i in COLORS]
#         #print(colors)
#         subset = data[:, [len(symbols)*m+n for m in range(4)]]
#         if subset.size < 400:
#             LINE_THICKNESS = 3
#         elif subset.size < 800:
#             LINE_THICKNESS = 2
#         else:
#             LINE_THICKNESS = 1
#         for i in range(np_len):
#             for j in range(4):
#                 if series[j]:
#                     cv2.circle(frame, (MARGIN+int(i*scale), int(subset[i, j])), LINE_THICKNESS, colors[j], -1, cv2.LINE_AA)
#         if sum_series > 1:
#             for i in range(np_len):
#                 cv2.line(frame, (MARGIN+int(i*scale), int(np.max(subset[i, series]))), (MARGIN+int(i*scale), int(np.min(subset[i, series]))), colors[0], 1, cv2.LINE_AA)
#         elif sum_series == 1: # implement arima?
#             j = series.index(True)
#             for i in range(np_len):
#                 cv2.line(frame, (MARGIN+int(i*scale), int(subset[i, j])), (MARGIN+int((i-1)*scale), int(subset[i-1, j])), colors[j], 1, cv2.LINE_AA)
#         cv2.putText(frame, symbols[n], (MARGIN, int(subset[0, 0])-20), cv2.FONT_HERSHEY_DUPLEX, 0.75, colors[0], 1, cv2.LINE_AA)

#     # put description in corner
#     cv2.putText(frame, f"High: {series[1]}", (700, 715), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
#     cv2.putText(frame, f"Close: {series[0]}", (850, 715), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
#     cv2.putText(frame, f"Open: {series[3]}", (1000, 715), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
#     cv2.putText(frame, f"Low: {series[2]}", (1150, 715), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
    
#     return frame


# TODO: add cli output on top of visual?
# def draw_state(frame, state):
#     y_top = 720 - MARGIN # 1080 = main.SCREEN_HEIGHT
#     cv2.putText(frame, str(state),
#         (MARGIN, y_top), cv2.FONT_HERSHEY_DUPLEX,
#         GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR,
#         GESTURE_FONT_THICKNESS, cv2.LINE_AA, False)
#     return frame

HAND_MODE = ('0: neutral', '1: scrolling', '2: whiteboard', '3: toggling series', '4: menu select')
GESTURE_FONT_SIZE = 0.3
GESTURE_FONT_THICKNESS = 1
GESTURE_TEXT_COLOR = (230, 230, 230)
def draw_gestures(frame):
    cv2.putText(frame, f'Gesture: {RESULT.gestures[0]}, {RESULT.gestures[1]}', (0, MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Angle: {round(RESULT.angles[0], 2)}, {round(RESULT.angles[1], 2)}', (0, 2*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Distances: {[[i for i in hand] for hand in RESULT.distances]}', (0, 3*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Hand mode: {HAND_MODE[STATE.hand_mode]}', (0, 4*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)
    cv2.putText(frame, f'Cooldown: {STATE.cooldown}', (0, 5*MARGIN), cv2.FONT_HERSHEY_DUPLEX, GESTURE_FONT_SIZE, GESTURE_TEXT_COLOR, GESTURE_FONT_THICKNESS)

def draw_landmarks(frame):
    # print(len(RESULT.landmarks))
    for hand in RESULT.landmarks:
        for landmark in hand:
            cv2.circle(frame, (int(WIN_WIDTH*landmark.x), int(WIN_HEIGHT*landmark.y)), 2, (230, 230, 230), -1)