# import stuff from loader
import fire
import yfinance as yf
import main

VALID_TICKERS = {
    'META',
    'AMZN',
    'NFLX',
    'GOOG',
    'AAPL',
    'PLTR',
    'ORCL', # OpenAI
    'RBLX',
    'NVDA'
}

def load(period='1y', interval='1d', *symbols:list[str]): # translate to yfinance api
    symbols = [s.upper() for s in symbols]
    for s in symbols: # simple validation
        if s not in VALID_TICKERS:
            return f'Invalid ticker: "{s}"'

    data = yf.download(' '.join(symbols), period=period, interval=interval).drop(columns=[('Volume', s) for s in symbols]) # returns pd.DataFrame
        # (GOOG Close, META Close, GOOG High, META High, GOOG Low, META Low, GOOG Open, META Open)
    datetimes = data.index
    #print(datetimes)
    main.event_loop(data.to_numpy(), symbols, datetimes)


if __name__ == '__main__':
    '''
    expose app to command line
    example 1: `python cli.py 5d 15m meta`
    example 2: `python cli.py --period=1mo --interval=1d meta goog`
    '''
    fire.Fire(load)