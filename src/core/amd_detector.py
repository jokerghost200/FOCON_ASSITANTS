import pandas as pd
import numpy as np

class AMDEngine:
    def __init__(self, lookback_period=20):
        self.lookback = lookback_period
        
    def detect_fvg(self, df):
        """
        Detects Fair Value Gaps in the last few candles.
        Bullish FVG: Low of candle 3 > High of candle 1
        Bearish FVG: High of candle 3 < Low of candle 1
        """
        if len(df) < 3:
            return "None"
            
        recent = df.tail(3).copy()
        recent.reset_index(drop=True, inplace=True)
        
        c1_high, c1_low = recent.loc[0, 'high'], recent.loc[0, 'low']
        c3_high, c3_low = recent.loc[2, 'high'], recent.loc[2, 'low']
        
        if c3_low > c1_high:
            return "Bullish FVG"
        elif c3_high < c1_low:
            return "Bearish FVG"
            
        return "None"

    def detect_swing_points(self, df, window=5):
        """
        Identifies recent Swing Highs and Swing Lows.
        """
        df = df.copy()
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(window, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(window, center=True).min())]
        return df

    def detect_bos_mss(self, df):
        """
        Detects Break of Structure (BOS) or Market Structure Shift (MSS).
        Did the latest close break a recent swing high/low?
        """
        df_swings = self.detect_swing_points(df)
        if len(df_swings) < 10:
            return "None"
            
        last_close = df_swings['close'].iloc[-1]
        
        # Get last known swing high and low
        recent_highs = df_swings['swing_high'].dropna()
        recent_lows = df_swings['swing_low'].dropna()
        
        if not recent_highs.empty and last_close > recent_highs.iloc[-1]:
            return "Bullish BOS/MSS"
        if not recent_lows.empty and last_close < recent_lows.iloc[-1]:
            return "Bearish BOS/MSS"
            
        return "None"
        
    def detect_liquidity_grab(self, df):
        """
        Detects Manipulation / Fakeout.
        Price wicked above a swing high but closed below it, or wicked below a swing low but closed above it.
        """
        df_swings = self.detect_swing_points(df)
        if len(df_swings) < 10:
            return "None"
            
        last_high = df_swings['high'].iloc[-1]
        last_low = df_swings['low'].iloc[-1]
        last_close = df_swings['close'].iloc[-1]
        
        recent_highs = df_swings['swing_high'].dropna()
        recent_lows = df_swings['swing_low'].dropna()
        
        if not recent_highs.empty:
            prev_high = recent_highs.iloc[-1]
            if last_high > prev_high and last_close < prev_high:
                return "Buy-side Liquidity Grab (Bearish Fakeout)"
                
        if not recent_lows.empty:
            prev_low = recent_lows.iloc[-1]
            if last_low < prev_low and last_close > prev_low:
                return "Sell-side Liquidity Grab (Bullish Fakeout)"
                
        return "None"

    def detect_amd_phase(self, df):
        """
        Determines the current AMD Phase: Accumulation, Manipulation, Distribution.
        Accumulation: Price is within a tight range (low volatility).
        Manipulation: Liquidity grab detected.
        Distribution: Strong directional movement post-manipulation.
        """
        if len(df) < self.lookback:
            return "Unknown"
            
        recent_df = df.tail(self.lookback)
        high_max = recent_df['high'].max()
        low_min = recent_df['low'].min()
        
        # Calculate recent ATR-like metric for volatility
        recent_df = recent_df.copy()
        recent_df['range'] = recent_df['high'] - recent_df['low']
        avg_range = recent_df['range'].mean()
        
        current_range = high_max - low_min
        
        # Check Manipulation first
        liq_grab = self.detect_liquidity_grab(df)
        if "Liquidity Grab" in liq_grab:
            return "Manipulation"
            
        # Accumulation: if the overall range over lookback is relatively small
        if current_range < (avg_range * 3):
            return "Accumulation"
            
        # Distribution: Trend is established (BOS occurred and range is expanding)
        bos = self.detect_bos_mss(df)
        if "BOS" in bos:
            return "Distribution"
            
        return "Consolidation"

    def analyze(self, df):
        """
        Runs the full AMD/ICT engine on the given dataframe.
        Returns a dictionary of context.
        """
        return {
            "Phase": self.detect_amd_phase(df),
            "FVG": self.detect_fvg(df),
            "BOS_MSS": self.detect_bos_mss(df),
            "Liquidity_Grab": self.detect_liquidity_grab(df)
        }
