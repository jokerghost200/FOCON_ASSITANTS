import MetaTrader5 as mt5
import logging
import threading
from typing import Optional
from src.core.state_machine import StateManager, BotState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5Connector:
    # Verrou GLOBAL partage par toutes les instances : le package MetaTrader5
    # n'est pas thread-safe ; deux acces concurrents (scan + trade manager)
    # provoquent un segfault (fermeture brutale sans traceback).
    _lock = threading.RLock()

    def __init__(self):
        self.connected = False
        self.state_manager = StateManager()

    def check_connection(self) -> bool:
        """Ping MT5. If disconnected, try to reconnect automatically."""
        with MT5Connector._lock:
            if not self.connected:
                return self.connect()
            
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("[MT5] Connexion au terminal perdue. Tentative de reconnexion...")
                self.connected = False
                self.state_manager.set_state(BotState.PAUSED)
                if self.connect():
                    self.state_manager.set_state(BotState.READY)
                    return True
                return False
            
            if not terminal_info.connected:
                logger.warning("[MT5] Le terminal est deconnecte du broker. Attente de reconnexion...")
                self.state_manager.set_state(BotState.CONNECTING)
                # We can't fix broker connection from here usually, just wait.
                return False

            if self.state_manager.get_state() == BotState.CONNECTING:
                self.state_manager.set_state(BotState.READY)
            return True

    def connect(self, retries: int = 3, delay: float = 1.5) -> bool:
        """Connects to the MetaTrader 5 terminal with automatic retries on transient errors."""
        import time
        with MT5Connector._lock:
            for attempt in range(1, retries + 1):
                if mt5.initialize():
                    self.connected = True
                    logger.info(f"Connected to MT5, version: {mt5.version()}")
                    return True

                err = mt5.last_error()
                # Libère les ressources du terminal avant de retenter
                try:
                    mt5.shutdown()
                except Exception:
                    pass

                if attempt < retries:
                    err_code, err_msg = err if isinstance(err, (tuple, list)) and len(err) >= 2 else (0, str(err))
                    if err_code == -6 or "Authorization" in str(err_msg):
                        logger.warning(
                            f"⚠️ Tentative {attempt}/{retries} : Échec d'autorisation MT5 "
                            f"(connexion broker ou réseau en cours de rétablissement). Nouvelle tentative dans {delay}s..."
                        )
                    else:
                        logger.warning(f"Tentative {attempt}/{retries} de connexion à MT5 échouée ({err}). Nouvelle tentative dans {delay}s...")
                    time.sleep(delay)

            logger.error(
                f"initialize() failed after {retries} attempts, error code = {err}. "
                "Assurez-vous que le terminal MetaTrader 5 est bien ouvert, connecté à Internet "
                "et authentifié sur votre compte de trading."
            )
            return False


    def disconnect(self):
        """Disconnects from the MetaTrader 5 terminal."""
        with MT5Connector._lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MT5")

    def get_account_info(self):
        """Retrieves account information."""
        with MT5Connector._lock:
            if not self.connected:
                return None
            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"Failed to get account info, error code = {mt5.last_error()}")
                return None
            return account_info._asdict()

    def check_symbol(self, symbol: str) -> bool:
        """Checks if a symbol is available in MT5."""
        with MT5Connector._lock:
            if not self.connected:
                return False

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol {symbol} not found")
                return False

            if not symbol_info.visible:
                logger.info(f"Symbol {symbol} is not visible, trying to make it visible")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"symbol_select({symbol}) failed, error code = {mt5.last_error()}")
                    return False

            return True

    def get_symbol_info(self, symbol: str):
        """Retrieves symbol properties."""
        with MT5Connector._lock:
            if not self.check_symbol(symbol):
                return None
            return mt5.symbol_info(symbol)._asdict()

    def get_visible_symbols(self, allowed_types: list = None) -> list:
        """Returns a list of all symbols currently visible in the MT5 Market Watch."""
        with MT5Connector._lock:
            if not self.connected:
                return []
            symbols = mt5.symbols_get()
            if symbols is None:
                return []

            visible_symbols = []
            for s in symbols:
                if s.visible:
                    if allowed_types:
                        path_lower = s.path.lower() if hasattr(s, 'path') else ""
                        if any(allowed.lower() in path_lower for allowed in allowed_types):
                            visible_symbols.append(s.name)
                    else:
                        visible_symbols.append(s.name)

            return visible_symbols

    def is_symbol_in_allowed_types(self, symbol: str, allowed_types: list = None) -> bool:
        """
        Vérifie si le symbole appartient aux types de marchés autorisés
        (ex: 'Crypto', 'Forex', 'Indices', 'Metals', etc.)
        """
        if not allowed_types:
            return True
            
        with MT5Connector._lock:
            if not self.connected:
                return True
            info = mt5.symbol_info(symbol)
            if info is None:
                return True
            path = getattr(info, 'path', '') or ''

        path_lower = path.lower()
        if path_lower:
            return any(str(allowed).lower() in path_lower for allowed in allowed_types)

        # Fallback heuristique si le courtier ne fournit pas de chemin hiérarchique
        sym = symbol.upper()
        crypto_roots = {"BTC", "ETH", "SOL", "ADA", "DOGE", "AVAX", "DOT", "LTC", "LNK", "LINK", "XLM", "BCH", "BNB", "MATIC", "XRP", "SHIB", "NEAR", "UNI"}
        is_crypto = any(sym.startswith(c) or c in sym for c in crypto_roots)

        allowed_lower = [str(a).lower() for a in allowed_types]
        if "crypto" in allowed_lower and is_crypto:
            return True
        if "forex" in allowed_lower and not is_crypto:
            fiat_currencies = {"EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
            if len(sym) >= 6 and (sym[:3] in fiat_currencies or sym[3:6] in fiat_currencies):
                return True

        return False

    def is_market_open(self, symbol: str) -> bool:
        """
        Checks if the market for a given symbol is currently open for trading.
        """
        with MT5Connector._lock:
            info = mt5.symbol_info(symbol)
            if info is None:
                return False

            # Check if trade mode allows trading
            # mt5.SYMBOL_TRADE_MODE_DISABLED means no trading
            if getattr(info, 'trade_mode', 0) in [mt5.SYMBOL_TRADE_MODE_DISABLED, mt5.SYMBOL_TRADE_MODE_CLOSEONLY]:
                return False

            # Check last tick time. If the last tick is older than 30 minutes,
            # it's highly likely the market is closed (weekend or session break).
            import time
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                current_time = time.time()
                if (current_time - tick.time) > 1800:  # 30 minutes
                    return False

            return True

    def get_rates(self, symbol: str, timeframe: str, num_bars: int = 100):
        """
        Retrieves historical rates (OHLCV) for a symbol.
        timeframe should be one of: 'M1', 'M5', 'M15', 'H1', 'H4', 'D1', etc.
        """
        if not self.check_symbol(symbol):
            return None
            
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        mt5_tf = tf_map.get(timeframe.upper())
        if mt5_tf is None:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        with MT5Connector._lock:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
            if rates is None:
                logger.error(f"Failed to get rates for {symbol}, error code = {mt5.last_error()}")
                return None

            return rates

    def order_check(self, symbol: str, action: str, volume: float, price: float, sl: float, tp: float):
        """
        Validates an order request before sending it.
        action: 'BUY' or 'SELL'
        """
        if not self.check_symbol(symbol):
            return None
            
        order_type = mt5.ORDER_TYPE_BUY if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        
        symbol_info = mt5.symbol_info(symbol)
        filling_mode = getattr(symbol_info, 'filling_mode', 0) if symbol_info else 0
        if filling_mode & 1:
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            type_filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 123456,
            "comment": "AI Assistant Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }
        
        with MT5Connector._lock:
            result = mt5.order_check(request)
            if result is None:
                logger.error(f"order_check failed, error code = {mt5.last_error()}")
                return None

            return result._asdict()
        
    def get_open_positions(self) -> list:
        """
        Retrieves all currently open positions from MT5.
        Returns a list of dictionaries with position details.
        """
        if not self.connected:
            return []

        with MT5Connector._lock:
            positions = mt5.positions_get()
        if positions is None:
            logger.warning(f"Failed to get positions, error code = {mt5.last_error()}")
            return []
            
        pos_list = []
        for pos in positions:
            pos_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'magic': getattr(pos, 'magic', 0)
            })
        return pos_list

    def order_send(self, request_dict: dict):
        """Sends an order to the MT5 terminal."""
        with MT5Connector._lock:
            result = mt5.order_send(request_dict)
            if result is None:
                logger.error(f"order_send failed, error code = {mt5.last_error()}")
                return None
            return result._asdict()

    # ── Pending Order (Limit) Management ──────────────────────────────────

    def place_pending_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "AI OB Pending",
        expiration_seconds: int = 14400,
    ) -> dict:
        """
        Places a Buy Limit or Sell Limit pending order on the specified price level.
        
        Parameters
        ----------
        action : 'BUY' or 'SELL'
        price  : the limit price (must be below ask for BUY_LIMIT, above bid for SELL_LIMIT)
        expiration_seconds : auto-cancel after this many seconds (default 4h)
        
        Returns
        -------
        dict with 'success', 'ticket', and optional 'reason'.
        """
        if not self.connected:
            return {"success": False, "reason": "MT5 not connected"}
        if not self.check_symbol(symbol):
            return {"success": False, "reason": f"Symbol {symbol} not available"}

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT

        # Dynamic filling mode
        symbol_info = mt5.symbol_info(symbol)
        filling_mode = getattr(symbol_info, 'filling_mode', 0) if symbol_info else 0
        if filling_mode & 1:
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            type_filling = mt5.ORDER_FILLING_RETURN

        import datetime
        expiration_time = int((datetime.datetime.now() + datetime.timedelta(seconds=expiration_seconds)).timestamp())

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiration_time,
            "type_filling": type_filling,
        }

        with MT5Connector._lock:
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.error(f"place_pending_order failed for {symbol}, error = {err}")
                return {"success": False, "reason": f"MT5 order_send failed: {err}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"place_pending_order rejected for {symbol}: "
                f"{result.comment} (Code {result.retcode})"
            )
            return {"success": False, "reason": f"{result.comment} (Code {result.retcode})"}

        ticket = int(result.order)
        logger.info(f"✅ [PENDING] Ordre limite placé: {action} {volume} {symbol} @ {price} (SL={sl}, TP={tp}, ticket #{ticket})")
        return {"success": True, "ticket": ticket}

    def cancel_pending_order(self, ticket: int) -> bool:
        """Cancels a pending order by ticket."""
        if not self.connected:
            return False

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }

        with MT5Connector._lock:
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"cancel_pending_order failed for #{ticket}, error = {mt5.last_error()}")
                return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"cancel_pending_order rejected for #{ticket}: {result.comment} (Code {result.retcode})")
            return False

        logger.info(f"🗑️ [PENDING] Ordre limite #{ticket} annulé avec succès.")
        return True

    def get_pending_orders(self, symbol: str = None) -> list:
        """
        Returns all pending orders, optionally filtered by symbol.
        Each order is returned as a dict with key fields.
        """
        if not self.connected:
            return []

        with MT5Connector._lock:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()

        if orders is None:
            return []

        order_list = []
        for o in orders:
            if getattr(o, 'magic', 0) != 123456:
                continue  # Only our bot's orders
            o_type_name = 'BUY' if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) else 'SELL'
            order_list.append({
                'ticket': o.ticket,
                'symbol': o.symbol,
                'type': o_type_name,
                'type_raw': o.type,
                'volume': o.volume_current,
                'price_open': o.price_open,
                'sl': o.sl,
                'tp': o.tp,
                'time_setup': getattr(o, 'time_setup', 0),
                'time_expiration': getattr(o, 'time_expiration', 0),
                'comment': getattr(o, 'comment', ''),
            })
        return order_list

    def modify_position_sl(self, ticket: int, symbol: str, new_sl: float, new_tp: float = 0.0) -> bool:
        """
        Modifies the Stop Loss (and optionally Take Profit) of an open position.
        """
        if not self.connected:
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": float(new_tp) if new_tp > 0 else 0.0,
            "magic": 123456
        }
        
        with MT5Connector._lock:
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"modify_position_sl failed for {ticket}, error code = {mt5.last_error()}")
                return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"modify_position_sl failed for {ticket}, retcode = {result.retcode}, comment = {result.comment}")
            return False
            
        logger.info(f"Successfully modified SL for position {ticket} to {new_sl}")
        return True

    def close_position(self, ticket: int, symbol: str, pos_type: str, volume: float, price: float) -> bool:
        """Closes an open position by sending an opposite order."""
        if not self.connected:
            return False
            
        order_type = mt5.ORDER_TYPE_SELL if str(pos_type).upper() == 'BUY' else mt5.ORDER_TYPE_BUY

        with MT5Connector._lock:
            symbol_info = mt5.symbol_info(symbol)
            filling_mode = getattr(symbol_info, 'filling_mode', 0) if symbol_info else 0
            if filling_mode & 1:
                type_filling = mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:
                type_filling = mt5.ORDER_FILLING_IOC
            else:
                type_filling = mt5.ORDER_FILLING_RETURN

            # Récupère le dernier prix réel (Bid pour fermer BUY, Ask pour fermer SELL)
            close_price = float(price)
            digits = getattr(symbol_info, 'digits', 5) if symbol_info else 5
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                live_p = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                if live_p and live_p > 0:
                    close_price = live_p
            close_price = round(close_price, digits)
        
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": close_price,
                "deviation": 20,
                "magic": 123456,
                "comment": "AI Auto Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
        
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = mt5.last_error()
                comment = result.comment if result else "None"
                logger.error(f"Failed to close position {ticket}. Error: {err}, Comment: {comment}")
                return False

        logger.info(f"Successfully closed position {ticket} on {symbol} @ {close_price}")
        return True

    def get_daily_profit(self) -> float:
        """
        Calculates total closed profit for the current calendar day.
        Includes profit, swap, and commissions of closed deals.
        """
        if not self.connected:
            return 0.0
        import datetime
        now = datetime.datetime.now()
        start_of_day = datetime.datetime(now.year, now.month, now.day, 0, 0, 0)
        end_of_day = start_of_day + datetime.timedelta(days=1)
        try:
            with MT5Connector._lock:
                deals = mt5.history_deals_get(start_of_day, end_of_day)
            if not deals:
                return 0.0
            total_profit = sum(
                float(d.profit) + float(getattr(d, 'swap', 0)) + float(getattr(d, 'commission', 0))
                for d in deals
                if getattr(d, 'entry', 0) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY)
            )
            return round(total_profit, 2)
        except Exception as _e:
            logger.warning(f"Failed to calculate daily profit: {_e}")
            return 0.0

    def get_deal_for_ticket(self, ticket: int) -> Optional[dict]:
        """
        Retrieves the closing deal information for a given position ticket from MT5 history.
        """
        if not self.connected:
            return None
        ticket = int(ticket)
        try:
            with MT5Connector._lock:
                deals = mt5.history_deals_get(position=ticket)
                if not deals:
                    import datetime
                    now = datetime.datetime.now()
                    start = now - datetime.timedelta(days=7)
                    deals = mt5.history_deals_get(start, now)
            if deals:
                for d in reversed(deals):
                    pos_id = getattr(d, 'position_id', getattr(d, 'position', 0))
                    if int(pos_id) == ticket and getattr(d, 'entry', 0) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY):
                        return {
                            'deal_ticket': d.ticket,
                            'price': float(d.price),
                            'profit': round(float(d.profit) + float(getattr(d, 'swap', 0)) + float(getattr(d, 'commission', 0)), 2),
                            'time': getattr(d, 'time', 0),
                            'comment': getattr(d, 'comment', ''),
                        }
        except Exception as _e:
            logger.debug(f"get_deal_for_ticket failed for {ticket}: {_e}")
        return None


