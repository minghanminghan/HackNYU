import pandas as pd
import numpy as np
from typing import List, Tuple


class data:
    #TODO: add arima
    def __init__(self, *args): # assumptions: df's Series is time-series
        self.shape: Tuple[int, int, int] = (0, 0, 0)
        self.series_names: List[List[str], List[str]] = [[], []]
        self.datetimes: List[str] = []
        self.values: np.ndarray = np.zeros((1, 1, 1))
    
    def set_data(self, values: pd.DataFrame):
        self.series_names = [i.to_list() for i in values.columns.levels] # [inner, outer]
        if 'Volume' in self.series_names[0]:
            self.series_names[0].remove('Volume')
        self.datetimes = values.index.to_list()
        self.shape = (len(self.datetimes), len(self.series_names[0]), len(self.series_names[1]))
        self.values = values.values.reshape(self.shape)

        # print(self.shape)


class state:
    # TODO: add distances, velocity, & angle (e.g. index to thumb, pinky to thumb)
    def __init__(self, *args):
        self.left_index = 0
        self.display_length = 0
        self.whiteboard = []
        self.display_symbols = [False] # turns into [True, ..., True] when data loads
        self.display_names = ['']
        self.toggle_index = 0
        self.hand_mode = 0 # 0: neutral, 1: scrolling, 2: whiteboard, 3: toggling series, 4: menu select
        self.video_mode = 1
        self.cooldown = 0
    
    def set_state(self, data:data):
        self.display_length: int = data.values.shape[0]                    # [1, SHAPE[0]]
        self.display_symbols: List[bool] = [True] * data.values.shape[2]   # [True, ..., True]
        self.display_names = data.series_names[1]

class result:
    def __init__(self):
        self.gestures = ['None', 'None'] # [left, right]
        self.angles = [0, 0]
        self.distances = [[False for i in range(4)] for j in range(2)] # [thumb-index, thumb-middle, thumb-ring, thumb-pinky]
        self.landmarks = [[], []]


DATA = data()
STATE = state()