import cv2
import numpy as np

from model import RESULT
from classes import DATA, STATE

cap = cv2.VideoCapture(0)
window_name = 'app name'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
_, _, window_w, window_h = cv2.getWindowImageRect(window_name)

COLORS = ((252, 246, 238), (0, 255, 0), (0, 0, 255), (252, 246, 238)) # close, high, low, open
INSTRUCTIONS = (
    # 0: scrolling
    ('Left Thumb + Left Index = Next Mode', 'Left Thumb + Left Pinky = Previous Mode'),
    # 1: whiteboard
    ('Left Thumb + Left Index = Toggle Pen', 'Left Thumb + Left Pinky = Clear Whiteboard'),
    # 2: toggle
    ('Left Thumb + Left Index = Next Series', 'Left Thumb + Left Pinky = Previous Series', 'Right Thumb + Right Index = Toggle Series', 'Right Thumb + Right Middle = Toggle All'),
    # 3: none
    ('Right Thumb + Right Pinky = Open Select Menu', ''),
    # 4: select
    ('Right Thumb + Right Pinky = Select Option', '')
)
states = ('Adjust Time Series', 'Whiteboard', 'Toggle Series', 'None', 'Menu')
substates = (
    ('None', 'Resize', 'Scroll', ''),
    ('Pen Off', 'Pen On'),
    tuple(DATA.series_names[1]), # loads?
    ('None', ''),
    ('Adjust Time Series', 'Whiteboard', 'Toggle Series', 'None')
)
# SCROLL_MODE = ('None', 'Resize', 'Scroll')

font_size = 0.5
font_thickness = 1
text_color = (230, 230, 230)
text_offset_h = 5 + cv2.getTextSize('', cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0][1] # → retval: (x, y), baseLine

margin_w = 100
margin_h = 100
frame_w = window_w - 2*margin_w
frame_h = window_h - 2*margin_h

box_bottom = frame_h+margin_h+text_offset_h


# 2. whiteboard
def draw_whiteboard(frame):
    # whiteboard
    pts = np.array(STATE.whiteboard, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, (255, 255, 255), 1)


# 4. select
def draw_wheel(frame):
    cv2.line(frame, 
        (int(window_w * RESULT.landmarks[0][8].x), int(window_h * RESULT.landmarks[0][8].y)),
        (int(window_w * RESULT.landmarks[1][8].x), int(window_h * RESULT.landmarks[1][8].y)),
        (255, 255, 255), 1)
    midpoint = (int(window_w * (RESULT.landmarks[0][8].x + RESULT.landmarks[1][8].x) / 2), int(window_h * (RESULT.landmarks[0][8].y + RESULT.landmarks[1][8].y) / 2))
    
    # distance & min/max logic is being reused here
    text_offset = cv2.getTextSize(states[STATE.subindex], cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0] # → retval: (x, y), baseLine

    cv2.circle(frame, (midpoint[0], midpoint[1]), 5, (255, 255, 255), -1)
    cv2.putText(frame, states[STATE.subindex], (midpoint[0] - text_offset[0]//2, midpoint[1] - text_offset[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)
    

def draw_data(frame):
    # 1. slice subset from DATA, STATE
    # print(STATE.display_length)
    # print(STATE.left_index)
    subset: np.ndarray = DATA.values[STATE.left_index: STATE.display_length + STATE.left_index]
    subset: np.ndarray = subset[:, :, STATE.display_symbols] # outer index
    
    if subset.size == 0:
        return
    
    # 2. draw subset
    # 2.1 scale to frame
    subset_max = subset.max()
    subset_min = subset.min()
    subset: np.ndarray = (frame_h * (1 - subset/subset_max) + margin_h).astype(int) # not sure if this is right

    # 2.2 draw points and lines (parallelize?)
    x_step = frame_w / STATE.display_length
    for datetime in range(subset.shape[0]):
        x = int(datetime*x_step) + margin_w
        for series_outer in range(subset.shape[2]):
            cv2.line(frame, (x, subset[datetime, 2, series_outer]), (x, subset[datetime, 1, series_outer]), (255, 255, 255), 1)
            for series_inner in range(subset.shape[1]):
                # x: datetime * scaling, y: subset[datetime, series_inner, series_outer]
                cv2.circle(frame, (x, subset[datetime, series_inner, series_outer]), 1, COLORS[series_inner], -1)

    # 3. label series
    for i in range(len(STATE.display_names)):
        offset_w = cv2.getTextSize(STATE.display_names[i], cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0][0]
        cv2.putText(frame, f'{STATE.display_names[i]}', (margin_w-offset_w-5, subset[0, 1, i]), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    #3.1 label max & min
    cv2.putText(frame, f'{round(subset_max, 2)}', (frame_w+margin_w+5, margin_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    cv2.putText(frame, f'{round(subset_min, 2)}', (frame_w+margin_w+5, int(frame_h * (1 - subset_min/subset_max) + margin_h)), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)


def draw_const(frame):
    # data-related
    cv2.rectangle(frame, (margin_w, margin_h-10), (frame_w+margin_w, frame_h+margin_h), (255, 255, 255), 1)
    
    # time series
    left = margin_w + int(STATE.left_index / DATA.shape[0] * frame_w)
    right = margin_w + int((STATE.left_index + STATE.display_length) / DATA.shape[0] * frame_w)

    cv2.line(frame, (margin_w, margin_h-30), (window_w-margin_w, margin_h-30), (255, 255, 255), 1)
    cv2.rectangle(frame, (left, margin_h-21), (right, margin_h-19), (255, 255, 255), -1)
    cv2.putText(frame, f'start: {DATA.datetimes[STATE.left_index]}', (margin_w, margin_h-2*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    text = f'end: {DATA.datetimes[min(STATE.left_index + STATE.display_length, DATA.shape[0]-1)]}'
    offset_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0][0]
    cv2.putText(frame, text, (window_w-margin_w-offset_w, margin_h-2*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)

    cv2.putText(frame, f'Mode: {states[STATE.index]}', (margin_w, box_bottom), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    instructions = INSTRUCTIONS[STATE.index]
    for i in range(len(instructions)):
        offset_w = cv2.getTextSize(instructions[i], cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0][0]
        cv2.putText(frame, instructions[i], (window_w-margin_w-offset_w, box_bottom+i*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)

    text = f'{substates[STATE.index][STATE.subindex]}'
    offset_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_size, 1)[0][0]
    cv2.rectangle(frame, (3*margin_w-5, box_bottom+STATE.subindex*(text_offset_h+5)-15), (3*margin_w+offset_w+5, box_bottom+STATE.subindex*(text_offset_h+5)+5), (255, 255, 255), 1) # highlight active
    for i in range(len(substates[STATE.index])):
        cv2.putText(frame, substates[STATE.index][i], (3*margin_w, box_bottom+i*(text_offset_h+5)), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)

    # state-related context: GET RID OF CONDITIONALS
    # if STATE.index == 0: # scrolling
    #     cv2.putText(frame, f'Scroll Mode: {SCROLL_MODE[STATE.scroll_mode]}', (margin_w, box_bottom+text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    # elif STATE.index == 1: # whiteboard
    #     cv2.putText(frame, f'Drawing: {STATE.can_draw}', (margin_w, box_bottom+text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    #     cv2.putText(frame, f'Whiteboard size: {len(STATE.whiteboard)}', (margin_w, box_bottom+2*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    # elif STATE.index == 2: # toggle series
    #     cv2.putText(frame, f'Current Series: {DATA.series_names[1][STATE.subindex]}', (margin_w, box_bottom+text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    #     cv2.putText(frame, f'All Series: {DATA.series_names[1]}', (margin_w, box_bottom+2*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
    #     cv2.putText(frame, f'Displaying: {STATE.display_symbols}', (margin_w, box_bottom+3*text_offset_h), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)


    # landmarks
    for i in range(len(RESULT.landmarks)):
        if not RESULT.stale[i]: # stale threshold: 6 frames
            continue
        hand = RESULT.landmarks[i]
        for landmark in hand[4::4]: # fingertips
            cv2.circle(frame, (int(window_w*landmark.x), int(window_h*landmark.y)), 2, (230, 230, 230), -1)