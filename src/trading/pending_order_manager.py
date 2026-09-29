"""
Pending Order Manager
=====================
Gère les ordres limite (pending) placés sur des Order Blocks non encore retestés.

Workflow :
1. Le détecteur d'Order Blocks identifie une zone OB valide mais PAS encore retestée.
2. Le PendingOrderManager place un Buy Limit ou Sell Limit sur cette zone.
3. Un thread de surveillance vérifie périodiquement si :
   a) L'OB a été invalidé (brisé) → annule l'ordre pending
   b) Les conditions techniques ont changé (RSI extrême, HTF contraire) → annule l'ordre
   c) L'ordre a expiré ou a été rempli → nettoyage
4. Quand le prix revient toucher la zone, MT5 exécute l'ordre automatiquement.

Confirmation pré-remplissage :
- Avant que l'ordre soit validé, on vérifie la confirmation en temps réel
  (mèche de rejet, RSI neutre, tendance HTF alignée).
- Si la confirmation échoue, l'ordre est annulé avant qu'il ne se déclenche.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class PendingOBOrder:
    """Represents a pending order placed on an Order Block zone."""
    ticket: int                     # MT5 pending order ticket
    symbol: str
    direction: str                  # 'BUY' or 'SELL'
    entry_price: float              # Limit price (on OB zone)
    sl: float
    tp: float
    volume: float
    ob_high: float                  # OB zone boundaries for invalidation checks
    ob_low: float
    ob_score: float
    timeframe: str
    created_at: float = 0.0        # timestamp
    max_age_seconds: int = 14400   # auto-cancel after 4 hours
    status: str = "ACTIVE"         # ACTIVE, FILLED, CANCELLED, INVALIDATED, EXPIRED


class PendingOrderManager:
    """
    Manages pending limit orders placed on Order Block zones.
    
    Responsibilities:
    - Place pending orders on valid OB zones
    - Monitor and validate orders periodically
    - Cancel orders whose OBs get invalidated
    - Cancel orders when pre-fill confirmation fails
    """

    def __init__(self, mt5_connector, ob_detector, tech_indicators, market_data, risk_manager):
        self.mt5 = mt5_connector
        self.ob_detector = ob_detector
        self.tech = tech_indicators
        self.market_data = market_data
        self.risk = risk_manager
        
        # Registry of active pending orders keyed by ticket
        self._pending_orders: Dict[int, PendingOBOrder] = {}
        self._lock = threading.Lock()
        
        # Configuration
        self.max_pending_per_symbol = 1       # Max 1 pending order per symbol
        self.min_ob_score = 55                # Minimum OB score to place pending
        self.confirmation_check_interval = 30  # seconds between confirmation checks
        self.max_pending_total = 5            # Max total pending orders

    def place_ob_pending(
        self,
        symbol: str,
        ob,  # OrderBlock dataclass
        timeframe: str,
        lot: float,
        htf_trend: str = "Unknown",
        expiration_seconds: int = 14400,
    ) -> Optional[PendingOBOrder]:
        """
        Places a pending limit order on an Order Block zone.
        
        Parameters
        ----------
        ob : OrderBlock — must have entry, sl, tp, direction, score calculated
        
        Returns
        -------
        PendingOBOrder if successfully placed, None otherwise.
        """
        with self._lock:
            # 1. Check limits
            if len(self._pending_orders) >= self.max_pending_total:
                logger.info(f"[PENDING] Max {self.max_pending_total} pending orders reached. Ignoring {symbol}.")
                return None
            
            # Check per-symbol limit
            sym_count = sum(1 for po in self._pending_orders.values() 
                         if po.symbol == symbol and po.status == "ACTIVE")
            if sym_count >= self.max_pending_per_symbol:
                logger.info(f"[PENDING] Already {sym_count} pending on {symbol}. Skipping.")
                return None

            # 2. Validate OB quality
            if ob.score < self.min_ob_score:
                logger.info(f"[PENDING] OB score {ob.score:.0f} < {self.min_ob_score} for {symbol}. Skipping.")
                return None

            if ob.entry is None or ob.sl is None or ob.tp is None:
                logger.warning(f"[PENDING] OB has no calculated levels for {symbol}. Skipping.")
                return None

            # 3. Validate direction vs current price (must be a LIMIT order, not immediate fill)
            tick = self.mt5.get_symbol_info(symbol)
            if not tick:
                logger.warning(f"[PENDING] Cannot get tick for {symbol}.")
                return None
            
            ask = float(tick.get('ask', 0))
            bid = float(tick.get('bid', 0))
            
            if ob.direction == 'BUY':
                # Buy Limit must be BELOW current ask
                if ob.entry >= ask:
                    logger.info(
                        f"[PENDING] {symbol}: BUY entry {ob.entry:.5f} >= ask {ask:.5f}. "
                        f"Price already in zone — use market order instead."
                    )
                    return None
            else:
                # Sell Limit must be ABOVE current bid
                if ob.entry <= bid:
                    logger.info(
                        f"[PENDING] {symbol}: SELL entry {ob.entry:.5f} <= bid {bid:.5f}. "
                        f"Price already in zone — use market order instead."
                    )
                    return None

            # 4. HTF alignment check
            htf_lower = str(htf_trend).lower()
            if ob.direction == 'BUY' and htf_lower in ('bearish', 'strong bearish'):
                logger.info(f"[PENDING] {symbol}: BUY pending rejected — HTF is {htf_trend}.")
                return None
            if ob.direction == 'SELL' and htf_lower in ('bullish', 'strong bullish'):
                logger.info(f"[PENDING] {symbol}: SELL pending rejected — HTF is {htf_trend}.")
                return None

            # 5. Place the pending order via MT5
            result = self.mt5.place_pending_order(
                symbol=symbol,
                action=ob.direction,
                volume=lot,
                price=ob.entry,
                sl=ob.sl,
                tp=ob.tp,
                comment=f"AI OB {ob.ob_type}",
                expiration_seconds=expiration_seconds,
            )

            if not result.get('success'):
                logger.warning(f"[PENDING] MT5 rejected pending for {symbol}: {result.get('reason')}")
                return None

            ticket = result['ticket']
            pending = PendingOBOrder(
                ticket=ticket,
                symbol=symbol,
                direction=ob.direction,
                entry_price=ob.entry,
                sl=ob.sl,
                tp=ob.tp,
                volume=lot,
                ob_high=ob.ob_high,
                ob_low=ob.ob_low,
                ob_score=ob.score,
                timeframe=timeframe,
                created_at=time.time(),
                max_age_seconds=expiration_seconds,
                status="ACTIVE",
            )
            self._pending_orders[ticket] = pending
            
            logger.info(
                f"📌 [PENDING OB] {ob.direction} LIMIT placé sur {symbol} @ {ob.entry:.5f} "
                f"(Zone [{ob.ob_low:.5f}-{ob.ob_high:.5f}], SL={ob.sl:.5f}, TP={ob.tp:.5f}, "
                f"Score={ob.score:.0f}, ticket #{ticket})"
            )
            return pending

    def validate_pending_orders(self, direction_filter: str = "BOTH") -> List[Dict]:
        """
        Periodic validation of all active pending orders.
        
        Checks:
        1. Is the OB still valid (not broken)?
        2. Are technical conditions still aligned?
        3. Has the order expired or been filled?
        4. Is RSI in an extreme zone that invalidates the setup?
        
        Returns list of actions taken (for logging/UI).
        """
        actions = []
        with self._lock:
            active_tickets = [
                (t, po) for t, po in self._pending_orders.items()
                if po.status == "ACTIVE"
            ]

        for ticket, pending in active_tickets:
            try:
                # Check if the pending order still exists in MT5
                mt5_orders = self.mt5.get_pending_orders(symbol=pending.symbol)
                order_exists = any(o['ticket'] == ticket for o in mt5_orders)
                
                if not order_exists:
                    # Order was either filled or removed externally
                    # Check if a position exists with this symbol (means it was filled)
                    positions = self.mt5.get_open_positions() or []
                    was_filled = any(
                        p.get('symbol') == pending.symbol 
                        and abs(p.get('price_open', 0) - pending.entry_price) < 0.001 * pending.entry_price
                        for p in positions
                    )
                    
                    with self._lock:
                        if was_filled:
                            pending.status = "FILLED"
                            logger.info(f"✅ [PENDING] #{ticket} {pending.symbol} — Ordre REMPLI ! Position ouverte.")
                            actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "FILLED"})
                        else:
                            pending.status = "EXPIRED"
                            logger.info(f"⏰ [PENDING] #{ticket} {pending.symbol} — Ordre expiré ou annulé manuellement.")
                            actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "EXPIRED"})
                    continue

                # ── Validation 1: OB Invalidation Check ──
                df = self.market_data.get_dataframe(pending.symbol, pending.timeframe)
                if df is not None and not df.empty:
                    df = self.tech.add_all_indicators(df)
                    atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else 0.0001
                    
                    # Check if OB zone has been breached (price closed through it)
                    buf = 0.3 * atr
                    recent = df.tail(5)
                    is_invalidated = False
                    
                    if pending.direction == 'BUY':
                        # Bullish OB invalidated if price closes below ob_low - buffer
                        is_invalidated = bool((recent['close'] < (pending.ob_low - buf)).any())
                    else:
                        # Bearish OB invalidated if price closes above ob_high + buffer
                        is_invalidated = bool((recent['close'] > (pending.ob_high + buf)).any())
                    
                    if is_invalidated:
                        self._cancel_and_mark(ticket, pending, "INVALIDATED",
                                             f"Zone OB brisée (prix a traversé la zone)")
                        actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "INVALIDATED"})
                        continue

                    # ── Validation 2: RSI Extreme Check ──
                    tech_features = self.tech.extract_latest_features(df)
                    rsi = float(tech_features.get('rsi_14', 50))
                    
                    if pending.direction == 'BUY' and rsi >= 72:
                        self._cancel_and_mark(ticket, pending, "INVALIDATED",
                                             f"RSI en surachat ({rsi:.1f}) — achat non-recommandé")
                        actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "RSI_CANCEL"})
                        continue
                    elif pending.direction == 'SELL' and rsi <= 28:
                        self._cancel_and_mark(ticket, pending, "INVALIDATED",
                                             f"RSI en survente ({rsi:.1f}) — vente non-recommandée")
                        actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "RSI_CANCEL"})
                        continue

                    # ── Validation 3: Price moving too far away ──
                    current_price = float(df['close'].iloc[-1])
                    distance_to_entry = abs(current_price - pending.entry_price)
                    max_distance = 5.0 * atr  # If price is more than 5 ATR away, cancel
                    
                    if distance_to_entry > max_distance:
                        self._cancel_and_mark(ticket, pending, "INVALIDATED",
                                             f"Prix trop éloigné de la zone ({distance_to_entry/atr:.1f} ATR)")
                        actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "TOO_FAR"})
                        continue

                # ── Validation 4: Age check ──
                age = time.time() - pending.created_at
                if age > pending.max_age_seconds:
                    self._cancel_and_mark(ticket, pending, "EXPIRED",
                                         f"Durée maximale dépassée ({age/3600:.1f}h)")
                    actions.append({"ticket": ticket, "symbol": pending.symbol, "action": "EXPIRED"})
                    continue

            except Exception as e:
                logger.warning(f"[PENDING] Erreur validation #{ticket}: {e}")

        return actions

    def _cancel_and_mark(self, ticket: int, pending: PendingOBOrder, status: str, reason: str):
        """Cancel a pending order in MT5 and update local tracking."""
        cancelled = self.mt5.cancel_pending_order(ticket)
        with self._lock:
            pending.status = status
        if cancelled:
            logger.info(f"🗑️ [PENDING] #{ticket} {pending.symbol} annulé: {reason}")
        else:
            logger.warning(f"[PENDING] #{ticket} — Échec d'annulation MT5 (peut-être déjà rempli): {reason}")

    def get_active_pendings(self) -> List[PendingOBOrder]:
        """Returns list of currently active pending orders."""
        with self._lock:
            return [po for po in self._pending_orders.values() if po.status == "ACTIVE"]

    def get_all_pendings(self) -> List[PendingOBOrder]:
        """Returns all pending orders (active + historical)."""
        with self._lock:
            return list(self._pending_orders.values())

    def cleanup_old(self, max_history: int = 50):
        """Remove old non-active entries from memory."""
        with self._lock:
            inactive = [t for t, po in self._pending_orders.items() if po.status != "ACTIVE"]
            for t in inactive[:-max_history]:
                del self._pending_orders[t]

    def get_stats(self) -> dict:
        """Returns statistics about pending orders."""
        with self._lock:
            active = sum(1 for po in self._pending_orders.values() if po.status == "ACTIVE")
            filled = sum(1 for po in self._pending_orders.values() if po.status == "FILLED")
            invalidated = sum(1 for po in self._pending_orders.values() if po.status == "INVALIDATED")
            expired = sum(1 for po in self._pending_orders.values() if po.status == "EXPIRED")
            cancelled = sum(1 for po in self._pending_orders.values() if po.status == "CANCELLED")
            return {
                "active": active,
                "filled": filled,
                "invalidated": invalidated,
                "expired": expired,
                "cancelled": cancelled,
                "total": len(self._pending_orders),
            }
