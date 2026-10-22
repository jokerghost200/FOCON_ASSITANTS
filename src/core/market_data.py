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

    def get_order_book_snapshot(self, symbol: str) -> dict:
        """
        Retrieves a full Level 2 Order Book (DOM) snapshot with bid and ask price ladders.
        Returns:
            dict with 'available', 'bids', 'asks', 'best_bid', 'best_ask', 'spread',
            'bids_volume', 'asks_volume', 'imbalance_ratio', 'buy_pressure', 'sell_pressure'.
        """
        import MetaTrader5 as mt5

        items = self.connector.get_market_depth_book(symbol)
        if items is None or len(items) == 0:
            return {"available": False, "reason": "DOM unavailable or unsupported by broker"}

        bids = []
        asks = []
        bids_volume = 0.0
        asks_volume = 0.0

        for item in items:
            vol = float(item.volume_real if hasattr(item, 'volume_real') and item.volume_real > 0 else item.volume)
            price = float(item.price)
            if item.type == getattr(mt5, 'BOOK_TYPE_BUY', 2):
                bids.append({"price": price, "volume": vol})
                bids_volume += vol
            elif item.type == getattr(mt5, 'BOOK_TYPE_SELL', 1):
                asks.append({"price": price, "volume": vol})
                asks_volume += vol

        # Sort bids descending (highest bid first) and asks ascending (lowest ask first)
        bids.sort(key=lambda x: x["price"], reverse=True)
        asks.sort(key=lambda x: x["price"])

        total_vol = bids_volume + asks_volume
        if total_vol == 0:
            return {"available": False, "reason": "Zero volume in DOM"}

        best_bid = bids[0]["price"] if bids else None
        best_ask = asks[0]["price"] if asks else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0
        buy_pressure = (bids_volume / total_vol) * 100.0
        sell_pressure = 100.0 - buy_pressure
        imbalance = (bids_volume - asks_volume) / total_vol  # -1.0 (100% asks) to +1.0 (100% bids)

        return {
            "available": True,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "bids_volume": round(bids_volume, 2),
            "asks_volume": round(asks_volume, 2),
            "imbalance_ratio": round(imbalance, 3),
            "buy_pressure": round(buy_pressure, 1),
            "sell_pressure": round(sell_pressure, 1),
        }

    def get_market_depth(self, symbol: str) -> dict:
        """
        Retrieves summary Depth of Market for backward compatibility.
        """
        snapshot = self.get_order_book_snapshot(symbol)
        if not snapshot.get("available"):
            return snapshot
        return {
            "available": True,
            "bids_volume": snapshot["bids_volume"],
            "asks_volume": snapshot["asks_volume"],
            "buy_pressure": snapshot["buy_pressure"],
            "sell_pressure": snapshot["sell_pressure"]
        }

    def get_ticks_dataframe(self, symbol: str, count: int = 1500) -> Optional[pd.DataFrame]:
        """
        Retrieves recent ticks from MT5 and builds an enriched DataFrame
        with aggressive side inference (Buy/Sell) and Delta for Order Flow analysis.
        """
        raw_ticks = self.connector.get_ticks(symbol, count=count)
        if raw_ticks is None or len(raw_ticks) == 0:
            return None

        df = pd.DataFrame(raw_ticks)
        if df.empty:
            return None

        # Convert timestamp to datetime
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            if 'time_msc' in df.columns:
                df['datetime'] = pd.to_datetime(df['time_msc'], unit='ms')
            df.set_index('datetime', inplace=True)

        # Standardize volume
        if 'volume_real' in df.columns and (df['volume_real'] > 0).any():
            df['vol'] = df['volume_real']
        elif 'volume' in df.columns:
            df['vol'] = df['volume']
        else:
            df['vol'] = 1.0
        # If volume is all 0 (common in some FX quote feeds), fallback to 1.0 unit per tick
        df['vol'] = df['vol'].apply(lambda x: 1.0 if x <= 0 else float(x))

        # Determine price (use last if > 0, else mid of bid/ask)
        if 'last' in df.columns and (df['last'] > 0).any():
            df['price'] = df['last'].where(df['last'] > 0, (df['bid'] + df['ask']) / 2.0)
        else:
            df['price'] = (df['bid'] + df['ask']) / 2.0

        # Infer aggressive side (Buy vs Sell)
        # 1. MT5 flags: TICK_FLAG_BUY = 32, TICK_FLAG_SELL = 64
        FLAG_BUY = 32
        FLAG_SELL = 64

        sides = []
        deltas = []
        prev_price = None

        flags_col = df['flags'] if 'flags' in df.columns else [0] * len(df)
        prices = df['price'].values
        volumes = df['vol'].values
        bids = df['bid'].values if 'bid' in df.columns else prices
        asks = df['ask'].values if 'ask' in df.columns else prices

        for i in range(len(df)):
            flg = int(flags_col.iloc[i]) if hasattr(flags_col, 'iloc') else int(flags_col[i])
            v = volumes[i]
            p = prices[i]
            b = bids[i]
            a = asks[i]

            if flg & FLAG_BUY:
                side = "BUY"
            elif flg & FLAG_SELL:
                side = "SELL"
            else:
                # Lee-Ready / Quote Rule fallback
                if p >= a:
                    side = "BUY"
                elif p <= b:
                    side = "SELL"
                elif prev_price is not None:
                    if p > prev_price:
                        side = "BUY"
                    elif p < prev_price:
                        side = "SELL"
                    else:
                        side = sides[-1] if sides else "BUY"
                else:
                    side = "BUY"

            prev_price = p
            sides.append(side)
            deltas.append(v if side == "BUY" else -v)

        df['side'] = sides
        df['delta'] = deltas
        return df

    def analyze_amd(self, df: pd.DataFrame) -> dict:
        """
        Runs the AMD/ICT engine on the given dataframe and returns the context dictionary.
        """
        if df is None or df.empty:
            return {"Phase": "Unknown", "FVG": "None", "BOS_MSS": "None", "Liquidity_Grab": "None"}
            
        return self.amd_engine.analyze(df)
