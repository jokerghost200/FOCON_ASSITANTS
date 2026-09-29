import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 50:
            logger.warning('Not enough data to calculate indicators.')
            return df
        df = df.copy()
        try:
            df['EMA_50']  = df['close'].ewm(span=50,  adjust=False).mean()
            df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD']        = ema_12 - ema_26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist']   = df['MACD'] - df['MACD_signal']
            delta = df['close'].diff()
            gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss.replace(0, float('nan'))
            df['RSI_14'] = 100 - (100 / (1 + rs))
            high_low   = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close  = (df['low']  - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
            w = 5
            df['swing_high'] = df['high'][df['high'] == df['high'].rolling(window=w*2+1, center=True).max()]
            df['swing_low']  = df['low'][df['low']   == df['low'].rolling(window=w*2+1, center=True).min()]
            df['last_swing_high'] = df['swing_high'].ffill()
            df['last_swing_low']  = df['swing_low'].ffill()
            df['bullish_bias'] = (df['close'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_200'])
            df['bearish_bias'] = (df['close'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_200'])
            df = self._add_fractals(df, period=2)
            df = self._add_zigzag(df, deviation_atr_mult=1.0)
        except Exception as e:
            logger.error(f'Error calculating indicators: {e}', exc_info=True)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Williams Fractals
    # ─────────────────────────────────────────────────────────────────────────

    def _add_fractals(self, df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
        n = len(df)
        fractal_up   = [False] * n
        fractal_down = [False] * n
        highs = df['high'].values
        lows  = df['low'].values
        for i in range(period, n - period):
            wh = highs[i - period: i + period + 1]
            wl = lows[i  - period: i + period + 1]
            if highs[i] == wh.max():
                fractal_up[i] = True
            if lows[i] == wl.min():
                fractal_down[i] = True
        df['fractal_up']   = fractal_up
        df['fractal_down'] = fractal_down
        df['fractal_up_level']   = np.nan
        df['fractal_down_level'] = np.nan
        df.loc[df['fractal_up'],   'fractal_up_level']   = df.loc[df['fractal_up'],   'high']
        df.loc[df['fractal_down'], 'fractal_down_level'] = df.loc[df['fractal_down'], 'low']
        df['last_fractal_high'] = df['fractal_up_level'].ffill()
        df['last_fractal_low']  = df['fractal_down_level'].ffill()
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # ZigZag (ATR-based)
    # ─────────────────────────────────────────────────────────────────────────

    def _add_zigzag(self, df: pd.DataFrame, deviation_atr_mult: float = 1.0) -> pd.DataFrame:
        if 'ATR_14' not in df.columns:
            return df
        atr_mean = float(df['ATR_14'].mean())
        if atr_mean <= 0:
            return df
        min_move = deviation_atr_mult * atr_mean
        n = len(df)
        pivots     = [None]  * n
        values     = [np.nan] * n
        directions = [0]     * n
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        lpt = None
        lpp = closes[0]
        lpi = 0
        for i in range(1, n):
            h = highs[i]; l = lows[i]
            if lpt != 'H' and (h - lpp) >= min_move:
                pivots[lpi] = lpt or 'L'; values[lpi] = lpp; directions[lpi] = -1
                lpt = 'H'; lpp = h; lpi = i
            elif lpt != 'L' and (lpp - l) >= min_move:
                pivots[lpi] = lpt or 'H'; values[lpi] = lpp; directions[lpi] = +1
                lpt = 'L'; lpp = l; lpi = i
            else:
                if lpt == 'H' and h > lpp: lpp = h; lpi = i
                elif lpt == 'L' and l < lpp: lpp = l; lpi = i
        df['zigzag_pivot'] = pivots
        df['zigzag_value'] = values
        df['zigzag_dir']   = directions
        dir_s = pd.Series(directions, index=df.index).replace(0, np.nan)
        df['zigzag_trend'] = dir_s.ffill().fillna(0).astype(int)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────────────

    def extract_latest_features(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}
        latest = df.iloc[-1]
        features = {
            'close':             latest.get('close'),
            'ema_50':            latest.get('EMA_50'),
            'ema_200':           latest.get('EMA_200'),
            'rsi_14':            latest.get('RSI_14'),
            'macd':              latest.get('MACD'),
            'macd_hist':         latest.get('MACD_hist'),
            'atr_14':            latest.get('ATR_14'),
            'last_swing_high':   latest.get('last_swing_high'),
            'last_swing_low':    latest.get('last_swing_low'),
            'trend':             ('BULLISH' if latest.get('bullish_bias') else 'BEARISH' if latest.get('bearish_bias') else 'NEUTRAL'),
            'fractal_up':        bool(latest.get('fractal_up',   False)),
            'fractal_down':      bool(latest.get('fractal_down', False)),
            'last_fractal_high': latest.get('last_fractal_high'),
            'last_fractal_low':  latest.get('last_fractal_low'),
            'zigzag_pivot':      latest.get('zigzag_pivot'),
            'zigzag_value':      latest.get('zigzag_value'),
            'zigzag_trend':      int(latest.get('zigzag_trend', 0)),
        }
        features = {k: (v if isinstance(v, (str, bool)) else (float(v) if (v is not None and pd.notnull(v)) else None)) for k, v in features.items()}
        return features
