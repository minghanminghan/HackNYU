# import stuff from loader
import fire
import yfinance as yf
import numpy as np
import pandas as pd

from main import event_loop
from classes import DATA, STATE

def read(path: str):
    '''
    read csv file as DataFrame
    '''
    global DATA, STATE
    raw = pd.read_csv(path, header=[0, 1], index_col=[0])

    DATA.set_data(raw)
    STATE.set_state(DATA)

    event_loop()
    


if __name__ == '__main__':
    '''
    expose app to command line
    example 1: `python cli.py 5d 15m meta`
    example 2: `python cli.py --period=1mo --interval=1d meta goog`
    '''
    fire.Fire()