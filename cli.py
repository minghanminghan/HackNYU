# import stuff from loader
import fire
import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Tuple

from classes import data, state
from main import event_loop
from classes import DATA, STATE

VALID_TICKERS = {
    'META',
    'AMZN',
    'NFLX',
    'GOOG',
    'AAPL',
    'PLTR',
    'ORCL',
    'RBLX',
    'NVDA'
}

def load(period='1y', interval='1d', *symbols:list[str]): # translate to yfinance api
    global DATA, STATE
    symbols = [s.upper() for s in symbols]
    for s in symbols: # simple validation
        if s not in VALID_TICKERS:
            return f'Invalid ticker: "{s}"'

    # dropping Volume for now
    raw = yf.download(' '.join(symbols), period=period, interval=interval).drop('Volume', axis=1, level=0).fillna(0)
    # print(raw.shape)

    DATA.set_data(raw)
    STATE.set_state(DATA)
    
    # print(DATA)
    # print(STATE)

    event_loop()


if __name__ == '__main__':
    '''
    expose app to command line
    example 1: `python cli.py 5d 15m meta`
    example 2: `python cli.py --period=1mo --interval=1d meta goog`
    '''
    fire.Fire(load)