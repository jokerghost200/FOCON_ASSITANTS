import pandas as pd
from typing import Optional
from src.core.mt5_connector import MT5Connector
from src.core.amd_detector import AMDEngine
import logging

logger = logging.getLogger(__name__)

class MarketDataEngine:
    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self.amd_engine = AMDEngine(lookback_period=20)

    def get_dataframe(self, symbol: str, timeframe: str, num_bars: int = 700) -> Optional[pd.DataFrame]:
        """
        Fetches historical data from MT5 and converts it to a pandas DataFrame.
        """
        rates = self.connector.get_rates(symbol, timeframe, num_bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates returned for {symbol} on {timeframe}")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time in seconds into datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        # Drop columns we don't strictly need (like real_volume if unavailable)
        # MT5 usually returns: time, open, high, low, close, tick_volume, spread, real_volume
        df = df[['open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df

    def get_latest_tick(self, symbol: str) -> Optional[dict]:
        """
        Retrieves the latest tick data for a symbol (Bid, Ask).
        """
        if not self.connector.check_symbol(symbol):
            return None
            
        tick = self.connector.get_symbol_info(symbol)
        if tick is None:
            return None
            
        return {
            'bid': tick.get('bid'),
            'ask': tick.get('ask'),
            'spread': tick.get('spread'),
            'time': tick.get('time')
        }

    def get_market_depth(self, symbol: str) -> dict:
        """
        Retrieves the Depth of Market (Order Book) for the symbol if available.
        Returns a summary of resting bids vs asks volume to indicate order flow pressure.
        """
        import MetaTrader5 as mt5
        
        if not self.connector.check_symbol(symbol):
            return {"available": False, "reason": "Symbol not available"}
            
        with MT5Connector._lock:
            if not mt5.market_book_add(symbol):
                return {"available": False, "reason": "Broker does not support DOM for this symbol"}

            try:
                items = mt5.market_book_get(symbol)
            finally:
                # Always release the book to not consume MT5 resources unnecessarily
                mt5.market_book_release(symbol)
        
        if not items:
            return {"available": False, "reason": "DOM is empty"}
            
        bids_volume = 0.0
        asks_volume = 0.0
        
        for item in items:
            # Type 1 = BOOK_TYPE_SELL, Type 2 = BOOK_TYPE_BUY
            if item.type == mt5.BOOK_TYPE_BUY:
                bids_volume += item.volume_real if hasattr(item, 'volume_real') and item.volume_real > 0 else item.volume
            elif item.type == mt5.BOOK_TYPE_SELL:
                asks_volume += item.volume_real if hasattr(item, 'volume_real') and item.volume_real > 0 else item.volume
                
        total_vol = bids_volume + asks_volume
        if total_vol == 0:
            return {"available": False, "reason": "Zero volume in DOM"}
            
        buy_pressure = (bids_volume / total_vol) * 100
        sell_pressure = 100 - buy_pressure
        
        return {
            "available": True,
            "bids_volume": bids_volume,
            "asks_volume": asks_volume,
            "buy_pressure": round(buy_pressure, 1),
            "sell_pressure": round(sell_pressure, 1)
        }

    def analyze_amd(self, df: pd.DataFrame) -> dict:
        """
        Runs the AMD/ICT engine on the given dataframe and returns the context dictionary.
        """
        if df is None or df.empty:
            return {"Phase": "Unknown", "FVG": "None", "BOS_MSS": "None", "Liquidity_Grab": "None"}
            
        return self.amd_engine.analyze(df)
