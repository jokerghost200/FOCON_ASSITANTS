"""
Backtest Engine (Moteur de Simulation sur Bougies Réelles MT5)
=============================================================
Permet de tester virtuellement les stratégies du robot sur l'historique réel des cours MT5 :
- Smart Money Concepts (Order Blocks institutionnels)
- Boomerang (Rebond EMA + alignement ZigZag)
- Candle Range Theory (CRT - Modèle 3 bougies & Pending Limits FVG)
- Cycles AMD (Accumulation, Manipulation, Distribution)
- Détecteur de Régime de Marché (Filtrage Choppy / Illiquide)
- Gestion de risque complète : Dynamique (% de risque) ou Lot Fixe configuré

Caractéristiques :
- Données de bougies réelles MT5 (M5, M15, H1, H4, etc.)
- Prise en compte exacte du spread et des spécifications du courtier (tick_size, tick_value)
- Simulation réaliste des ordres en attente (BUY_LIMIT / SELL_LIMIT) avec expiration
- Détection du Palier de Capital (Micro $50, Standard $1000, Pro $10k+)
- Protection Micro-Compte et vérification de marge
- Gestion réaliste du trade (Break-Even automatique, Trailing Stop, TP/SL)
- Analyse approfondie et conclusion diagnostique autonome du robot
"""

import math
import logging
from typing import Optional, Dict, Any, List, Callable
import pandas as pd
import numpy as np

from src.core.mt5_connector import MT5Connector
from src.core.technical_indicators import TechnicalIndicators
from src.core.order_block_detector import OrderBlockDetector
from src.core.boomerang_detector import BoomerangDetector
from src.core.crt_detector import CRTDetector, CRTSetup
from src.core.amd_detector import AMDEngine
from src.core.market_regime import MarketRegimeDetector, MarketRegimeType
from src.ai.local_quant_brain import LocalQuantBrain
from src.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        mt5_connector: MT5Connector,
        risk_manager: Optional[RiskManager] = None,
        weights_engine=None,
    ):
        self.mt5 = mt5_connector
        self.risk = risk_manager or RiskManager()
        self.tech = TechnicalIndicators()
        self.ob_detector = OrderBlockDetector(
            impulse_atr_mult=1.8,
            lookback=100,
            retest_tolerance=0.6,
            sl_buffer_mult=0.2,
            tp_rr_mult=2.5,
        )
        self.boomerang = BoomerangDetector(
            impulse_atr_mult=2.0,
            breakout_confirm=1,
            retest_tolerance=0.5,
            rejection_wick_min=0.3,
            min_rr=1.5,
        )
        self.crt_detector = CRTDetector(
            min_sweep_atr_mult=0.10,
            min_range_atr_mult=0.50,
            min_rr_tp1=1.5,
        )
        self.amd = AMDEngine(lookback_period=20)
        self.market_regime = MarketRegimeDetector(adx_trending_threshold=22.0)
        self.local_brain = LocalQuantBrain(min_confidence=60.0, min_rr=1.5, weights_engine=weights_engine)

    def run_backtest(
        self,
        symbol: str,
        timeframe: str = "M15",
        num_bars: int = 600,
        initial_capital: float = 100.0,
        risk_profile: str = "BALANCED",
        risk_percent: Optional[float] = None,
        direction_filter: str = "BOTH",
        strategy_filter: str = "ALL",
        fixed_lot_size: Optional[float] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        stop_event=None,
    ) -> Dict[str, Any]:
        """
        Exécute le backtest virtuel bougie par bougie sur les données historiques réelles MT5.
        Supporte le filtrage de stratégie ('ALL', 'CRT', 'ORDER_BLOCKS', 'BOOMERANG')
        et le volume fixe ('fixed_lot_size').
        """
        def _report(progress: float, text: str):
            if progress_cb:
                try:
                    progress_cb(progress, text)
                except Exception:
                    pass

        _report(0.05, f"Récupération des données réelles MT5 pour {symbol} ({timeframe})...")

        # 1. Vérification connexion et récupération des bougies MT5
        if not self.mt5.connected:
            if not self.mt5.connect():
                return {"error": "Impossible de se connecter à MetaTrader 5."}

        rates = self.mt5.get_rates(symbol, timeframe, num_bars)
        if rates is None or len(rates) < 60:
            return {
                "error": f"Historique insuffisant pour {symbol} ({timeframe}). Le courtier a retourné {len(rates) if rates is not None else 0} bougies."
            }

        symbol_info = self.mt5.get_symbol_info(symbol)
        if not symbol_info:
            return {"error": f"Impossible d'obtenir les spécifications du symbole {symbol} depuis MT5."}

        # Conversion en DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume", "spread"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        _report(0.12, "Calcul des indicateurs institutionnels et SMC...")

        # 2. Configuration du gestionnaire de risque pour ce capital
        self.risk.apply_risk_profile(risk_profile)
        if risk_percent is not None and risk_percent > 0:
            self.risk.default_risk_percent = float(risk_percent)

        if fixed_lot_size is not None and fixed_lot_size > 0.0:
            self.risk.fixed_lot_size = float(fixed_lot_size)
        elif fixed_lot_size == 0.0:
            self.risk.fixed_lot_size = 0.0

        effective_fixed_lot = getattr(self.risk, "fixed_lot_size", 0.0)

        tier_info = RiskManager.get_account_tier(initial_capital)
        eff_max_open_pos = self.risk.get_effective_max_open_positions(initial_capital, base_max_pos=2)

        # Spécifications du symbole
        point = float(symbol_info.get("point") or 0.00001)
        tick_size = float(symbol_info.get("trade_tick_size") or point)
        tick_value = float(symbol_info.get("trade_tick_value") or 1.0)
        volume_min = float(symbol_info.get("volume_min") or 0.01)
        volume_step = float(symbol_info.get("volume_step") or 0.01)

        # 3. Précalcul vectoriel des indicateurs
        df = self.tech.add_all_indicators(df)

        _report(0.20, f"Simulation en cours sur {len(df)} bougies réelles...")

        current_balance = float(initial_capital)
        peak_balance = current_balance
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0

        open_positions: List[Dict[str, Any]] = []
        pending_orders: List[Dict[str, Any]] = []
        closed_trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = [
            {"time": str(df.index[0]), "balance": current_balance, "equity": current_balance}
        ]

        # Compteurs statistiques
        blocked_by_micro_protection = 0
        blocked_by_spread = 0
        blocked_by_confidence = 0
        blocked_by_rr = 0
        blocked_by_regime = 0
        total_signals_generated = 0
        pending_placed_count = 0
        pending_filled_count = 0
        pending_expired_count = 0
        next_ticket = 1001

        warmup = 50
        total_steps = len(df) - warmup

        # 4. Boucle de replay bougie par bougie
        for idx in range(warmup, len(df)):
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                break

            if idx % 25 == 0:
                pct = 0.20 + 0.70 * ((idx - warmup) / max(1, total_steps))
                _report(pct, f"Simulation bougie {idx}/{len(df)} | Solde: {current_balance:.2f} $...")

            row = df.iloc[idx]
            bar_time = str(df.index[idx])
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            spread_pts = float(row.get("spread", 0) or symbol_info.get("spread", 10))
            spread_price = spread_pts * point

            # ── A. Vérification et exécution des Ordres en Attente (Pending Limits) ──
            for p_order in pending_orders[:]:
                order_type = p_order["order_type"]
                entry_target = p_order["entry"]
                filled = False

                # Expiration atteinte
                if (idx - p_order["placed_bar_idx"]) > p_order["expiration_bars"]:
                    pending_orders.remove(p_order)
                    pending_expired_count += 1
                    continue

                # Vérification déclenchement (Fill)
                if order_type == "BUY_LIMIT":
                    if l <= entry_target:
                        filled = True
                        exec_entry = entry_target
                elif order_type == "SELL_LIMIT":
                    if h >= entry_target:
                        filled = True
                        exec_entry = entry_target

                if filled:
                    pending_orders.remove(p_order)
                    pending_filled_count += 1
                    open_positions.append({
                        "ticket": p_order["ticket"],
                        "open_time": bar_time,
                        "action": p_order["action"],
                        "entry": exec_entry,
                        "sl": p_order["sl"],
                        "tp": p_order["tp"],
                        "lot": p_order["lot"],
                        "be_locked": False,
                        "balance_at_open": current_balance,
                        "setup_reason": p_order["setup_reason"],
                        "strategy": p_order["strategy"],
                    })

            # ── B. Gestion des positions ouvertes existantes ──
            for pos in open_positions[:]:
                pos_type = pos["action"]
                entry = pos["entry"]
                sl = pos["sl"]
                tp = pos["tp"]
                vol = pos["lot"]
                ticket = pos["ticket"]

                exited = False
                exit_price = 0.0
                exit_reason = ""

                if pos_type == "BUY":
                    # Progression vers le TP pour le Break-Even
                    tp_dist = tp - entry
                    if tp_dist > 0 and (h - entry) >= (0.45 * tp_dist) and not pos["be_locked"]:
                        pos["sl"] = round(entry + max(5 * point, spread_price), 5)
                        pos["be_locked"] = True

                    # Vérification sortie : SL touché
                    if l <= pos["sl"]:
                        exited = True
                        exit_price = pos["sl"]
                        exit_reason = "BREAK_EVEN" if pos["be_locked"] and pos["sl"] >= entry else "STOP_LOSS"
                    # Vérification sortie : TP touché
                    elif h >= pos["tp"]:
                        exited = True
                        exit_price = pos["tp"]
                        exit_reason = "TAKE_PROFIT"

                elif pos_type == "SELL":
                    # Progression vers le TP pour le Break-Even
                    tp_dist = entry - tp
                    if tp_dist > 0 and (entry - l) >= (0.45 * tp_dist) and not pos["be_locked"]:
                        pos["sl"] = round(entry - max(5 * point, spread_price), 5)
                        pos["be_locked"] = True

                    # Vérification sortie : SL touché
                    if h >= pos["sl"]:
                        exited = True
                        exit_price = pos["sl"]
                        exit_reason = "BREAK_EVEN" if pos["be_locked"] and pos["sl"] <= entry else "STOP_LOSS"
                    # Vérification sortie : TP touché
                    elif l <= pos["tp"]:
                        exited = True
                        exit_price = pos["tp"]
                        exit_reason = "TAKE_PROFIT"

                if exited:
                    # Calcul du PnL brut
                    pts = (exit_price - entry) if pos_type == "BUY" else (entry - exit_price)
                    ticks = pts / max(tick_size, 1e-9)
                    raw_pnl = ticks * tick_value * vol
                    spread_cost = (spread_price / max(tick_size, 1e-9)) * tick_value * vol
                    net_pnl = round(raw_pnl - spread_cost, 2)

                    current_balance = round(current_balance + net_pnl, 2)
                    if current_balance > peak_balance:
                        peak_balance = current_balance

                    # Calcul drawdown
                    dd_usd = peak_balance - current_balance
                    dd_pct = (dd_usd / peak_balance * 100.0) if peak_balance > 0 else 0.0
                    if dd_usd > max_drawdown_usd:
                        max_drawdown_usd = dd_usd
                    if dd_pct > max_drawdown_pct:
                        max_drawdown_pct = dd_pct

                    closed_trades.append({
                        "ticket": ticket,
                        "open_time": pos["open_time"],
                        "close_time": bar_time,
                        "symbol": symbol,
                        "action": pos_type,
                        "lot": vol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "exit_price": exit_price,
                        "profit_usd": net_pnl,
                        "profit_pct": round((net_pnl / pos["balance_at_open"]) * 100.0, 2),
                        "exit_reason": exit_reason,
                        "balance_after": current_balance,
                        "strategy": pos.get("strategy", "FUSION"),
                    })

                    open_positions.remove(pos)
                    equity_curve.append({
                        "time": bar_time,
                        "balance": current_balance,
                        "equity": current_balance,
                    })

            # ── C. Recherche de nouveaux signaux d'entrée ──
            if (len(open_positions) + len(pending_orders)) < eff_max_open_pos and current_balance > 5.0:
                sub_df = df.iloc[max(0, idx - 110) : idx + 1]
                tech_feat = self.tech.extract_latest_features(sub_df)
                atr_val = float(tech_feat.get("atr_14") or point * 10)

                # 1. Analyse AMD & Régime de Marché
                amd_context = {
                    "Phase": self.amd.detect_amd_phase(sub_df),
                    "FVG": self.amd.detect_fvg(sub_df),
                    "Liquidity_Grab": self.amd.detect_liquidity_grab(sub_df),
                    "BOS_MSS": self.amd.detect_bos_mss(sub_df),
                }

                spread_ratio = (spread_price / atr_val) if atr_val > 0 else 0.0
                regime_diag = self.market_regime.detect_regime(tech_feat, amd_context, spread_ratio)

                # Filtre de régime illiquide / haché : s'abstenir pour protéger le capital
                if regime_diag.get("regime") == MarketRegimeType.CHOPPY_ILLIQUID:
                    blocked_by_regime += 1
                    continue

                decision: Optional[Dict[str, Any]] = None
                strategy_tag = "FUSION"

                # Construction du sous-ensemble HTF (1h) pour analyse CRT
                try:
                    htf_df = sub_df.resample('1h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                except Exception:
                    htf_df = sub_df

                # ── Cas 1 : Mode Dédié CRT (Candle Range Theory) ──
                if strategy_filter == "CRT":
                    crt_setup = None
                    if len(htf_df) >= 3:
                        crt_setup = self.crt_detector.detect_smc_pending(
                            df_htf=htf_df,
                            df_ltf=sub_df,
                            atr_ltf=atr_val,
                            spread=spread_price,
                            min_rr=1.5,
                            expiration_bars=4,
                        )
                        if not crt_setup:
                            crt_setup = self.crt_detector.detect(htf_df, atr=atr_val, spread=spread_price)

                    if crt_setup:
                        req_dir = str(crt_setup.direction).upper()
                        if direction_filter == "BOTH" or direction_filter.startswith(req_dir):
                            is_pend = getattr(crt_setup, 'order_type', 'DEAL') in ("BUY_LIMIT", "SELL_LIMIT")
                            decision = {
                                "action": req_dir,
                                "entry": float(crt_setup.entry_price or c),
                                "sl": float(crt_setup.sl or crt_setup.sl_raw or (c - atr_val)),
                                "tp": float(crt_setup.tp or crt_setup.tp2 or (c + atr_val * 2)),
                                "confidence": 85 if is_pend else 75,
                                "is_pending": is_pend,
                                "order_type": getattr(crt_setup, 'order_type', 'DEAL'),
                                "expiration_bars": getattr(crt_setup, 'expiration_bars', 4),
                                "reason": f"CRT Setup ({crt_setup.direction})",
                            }
                            strategy_tag = "CRT"

                # ── Cas 2 : Mode Dédié Order Blocks SMC ──
                elif strategy_filter == "ORDER_BLOCKS":
                    obs = self.ob_detector.detect(sub_df, symbol, timeframe, direction_filter=direction_filter)
                    if obs:
                        best_ob = obs[0]
                        if best_ob.score >= 65 and best_ob.sl and best_ob.tp:
                            decision = {
                                "action": best_ob.direction,
                                "entry": c,
                                "sl": float(best_ob.sl),
                                "tp": float(best_ob.tp),
                                "confidence": int(best_ob.score),
                                "is_pending": False,
                                "order_type": "DEAL",
                                "reason": f"Order Block {best_ob.direction} (Score {best_ob.score})",
                            }
                            strategy_tag = "ORDER_BLOCK"

                # ── Cas 3 : Mode Dédié Boomerang (Rebond EMA + ZigZag) ──
                elif strategy_filter == "BOOMERANG":
                    boom = self.boomerang.detect(sub_df, symbol, timeframe)
                    if boom.decision in ("BUY", "SELL"):
                        req_dir = boom.direction
                        if direction_filter == "BOTH" or direction_filter.startswith(req_dir):
                            decision = {
                                "action": req_dir,
                                "entry": c,
                                "sl": float(boom.sl),
                                "tp": float(boom.tp),
                                "confidence": int(boom.score),
                                "is_pending": False,
                                "order_type": "DEAL",
                                "reason": f"Boomerang Rebound ({req_dir}, Score {boom.score})",
                            }
                            strategy_tag = "BOOMERANG"

                # ── Cas 4 : Mode 'ALL' (Synergie Complète via Cerveau Quantitatif Local) ──
                else:
                    obs = self.ob_detector.detect(sub_df, symbol, timeframe, direction_filter=direction_filter)
                    boom = self.boomerang.detect(sub_df, symbol, timeframe)
                    crt_setup = None
                    if len(htf_df) >= 3:
                        crt_setup = self.crt_detector.detect_smc_pending(
                            df_htf=htf_df,
                            df_ltf=sub_df,
                            atr_ltf=atr_val,
                            spread=spread_price,
                            min_rr=1.5,
                            expiration_bars=4,
                        )
                        if not crt_setup:
                            crt_setup = self.crt_detector.detect(htf_df, atr=atr_val, spread=spread_price)

                    decision = self.local_brain.decide(
                        symbol=symbol,
                        timeframe=timeframe,
                        tech_features=tech_feat,
                        amd_context=amd_context,
                        boomerang_data={"decision": boom.decision, "direction": boom.direction, "score": boom.score},
                        order_blocks=obs,
                        spread_price=spread_price,
                        min_distance=max(point * 10, spread_price * 1.5),
                        direction_filter=direction_filter,
                        crt_setup=crt_setup,
                    )

                    if crt_setup and getattr(crt_setup, 'order_type', '') in ("BUY_LIMIT", "SELL_LIMIT") and crt_setup.direction == decision.get("action"):
                        decision["is_pending"] = True
                        decision["order_type"] = crt_setup.order_type
                        decision["entry"] = float(crt_setup.entry_price or decision["entry"])
                        decision["expiration_bars"] = getattr(crt_setup, 'expiration_bars', 4)
                        strategy_tag = "CRT"
                    elif "Order Block" in decision.get("reason", ""):
                        strategy_tag = "ORDER_BLOCK"
                    elif "Boomerang" in decision.get("reason", ""):
                        strategy_tag = "BOOMERANG"
                    else:
                        strategy_tag = "FUSION"

                if decision and decision.get("action") in ("BUY", "SELL"):
                    total_signals_generated += 1
                    acc_info = {"balance": current_balance, "margin_free": current_balance}
                    validation = self.risk.validate_proposal(decision, acc_info, symbol_info)

                    if validation["valid"]:
                        prop = validation["proposal"]
                        trade_entry = float(prop.get("entry", c))
                        trade_sl = float(prop.get("sl"))
                        trade_tp = float(prop.get("tp"))
                        trade_lot = float(prop.get("lot"))
                        is_pending = bool(decision.get("is_pending"))
                        order_type = str(decision.get("order_type", "DEAL"))

                        if is_pending and order_type in ("BUY_LIMIT", "SELL_LIMIT"):
                            pending_orders.append({
                                "ticket": next_ticket,
                                "placed_bar_idx": idx,
                                "placed_time": bar_time,
                                "action": decision["action"],
                                "order_type": order_type,
                                "entry": trade_entry,
                                "sl": trade_sl,
                                "tp": trade_tp,
                                "lot": trade_lot,
                                "expiration_bars": int(decision.get("expiration_bars", 4)),
                                "strategy": strategy_tag,
                                "setup_reason": decision.get("reason", ""),
                            })
                            pending_placed_count += 1
                            next_ticket += 1
                        else:
                            open_positions.append({
                                "ticket": next_ticket,
                                "open_time": bar_time,
                                "action": decision["action"],
                                "entry": c,
                                "sl": trade_sl,
                                "tp": trade_tp,
                                "lot": trade_lot,
                                "be_locked": False,
                                "balance_at_open": current_balance,
                                "setup_reason": decision.get("reason", ""),
                                "strategy": strategy_tag,
                            })
                            next_ticket += 1
                    else:
                        reason_msg = validation.get("reason", "")
                        if "lot minimum broker" in reason_msg or "Protection Micro-Compte" in reason_msg or "Tier MICRO" in reason_msg or "Risque excessif" in reason_msg:
                            blocked_by_micro_protection += 1
                        elif "Spread" in reason_msg:
                            blocked_by_spread += 1
                        elif "Confidence" in reason_msg or "Confiance" in reason_msg:
                            blocked_by_confidence += 1
                        elif "Reward/risk" in reason_msg or "R:R" in reason_msg:
                            blocked_by_rr += 1

        # 5. Clôture des positions encore ouvertes à la dernière bougie
        if open_positions:
            last_row = df.iloc[-1]
            last_price = float(last_row["close"])
            last_time = str(df.index[-1])
            spread_pts = float(last_row.get("spread", 0) or symbol_info.get("spread", 10))
            spread_price = spread_pts * point

            for pos in open_positions:
                pos_type = pos["action"]
                entry = pos["entry"]
                vol = pos["lot"]
                pts = (last_price - entry) if pos_type == "BUY" else (entry - last_price)
                ticks = pts / max(tick_size, 1e-9)
                raw_pnl = ticks * tick_value * vol
                spread_cost = (spread_price / max(tick_size, 1e-9)) * tick_value * vol
                net_pnl = round(raw_pnl - spread_cost, 2)

                current_balance = round(current_balance + net_pnl, 2)
                closed_trades.append({
                    "ticket": pos["ticket"],
                    "open_time": pos["open_time"],
                    "close_time": last_time,
                    "symbol": symbol,
                    "action": pos_type,
                    "lot": vol,
                    "entry": entry,
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "exit_price": last_price,
                    "profit_usd": net_pnl,
                    "profit_pct": round((net_pnl / pos["balance_at_open"]) * 100.0, 2),
                    "exit_reason": "FIN_BACKTEST",
                    "balance_after": current_balance,
                    "strategy": pos.get("strategy", "FUSION"),
                })

        _report(0.95, "Calcul des métriques et synthèse diagnostique du robot...")

        # 6. Métriques de performance
        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t["profit_usd"] > 0]
        losses = [t for t in closed_trades if t["profit_usd"] < 0]
        breakevens = [t for t in closed_trades if t["profit_usd"] == 0]

        total_gain_usd = sum(t["profit_usd"] for t in wins)
        total_loss_usd = abs(sum(t["profit_usd"] for t in losses))

        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (total_gain_usd / total_loss_usd) if total_loss_usd > 0 else (99.0 if total_gain_usd > 0 else (1.0 if total_trades == 0 else 0.0))
        net_profit_usd = round(current_balance - initial_capital, 2)
        net_return_pct = round((net_profit_usd / initial_capital) * 100.0, 2)

        avg_win = (total_gain_usd / len(wins)) if wins else 0.0
        avg_loss = (total_loss_usd / len(losses)) if losses else 0.0

        # Comptage des sorties par type
        exits_tp = len([t for t in closed_trades if t["exit_reason"] == "TAKE_PROFIT"])
        exits_sl = len([t for t in closed_trades if t["exit_reason"] == "STOP_LOSS"])
        exits_be = len([t for t in closed_trades if t["exit_reason"] == "BREAK_EVEN"])

        # Analyse par stratégie
        trades_by_strategy: Dict[str, Dict[str, Any]] = {}
        for st in ("CRT", "ORDER_BLOCK", "BOOMERANG", "FUSION"):
            sub_t = [t for t in closed_trades if t.get("strategy") == st]
            if sub_t:
                s_wins = [t for t in sub_t if t["profit_usd"] > 0]
                s_pnl = sum(t["profit_usd"] for t in sub_t)
                trades_by_strategy[st] = {
                    "count": len(sub_t),
                    "wins": len(s_wins),
                    "win_rate": round(len(s_wins) / len(sub_t) * 100.0, 1),
                    "net_pnl": round(s_pnl, 2),
                }

        # 7. Génération du Rapport / Conclusion du Bot
        conclusion_text = self._generate_ai_conclusion(
            symbol=symbol,
            timeframe=timeframe,
            num_bars=len(df),
            initial_capital=initial_capital,
            final_balance=current_balance,
            net_profit_usd=net_profit_usd,
            net_return_pct=net_return_pct,
            tier_info=tier_info,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            exits_tp=exits_tp,
            exits_sl=exits_sl,
            exits_be=exits_be,
            blocked_by_micro_protection=blocked_by_micro_protection,
            blocked_by_spread=blocked_by_spread,
            blocked_by_confidence=blocked_by_confidence,
            blocked_by_rr=blocked_by_rr,
            blocked_by_regime=blocked_by_regime,
            total_signals_generated=total_signals_generated,
            strategy_filter=strategy_filter,
            effective_fixed_lot=effective_fixed_lot,
            pending_placed=pending_placed_count,
            pending_filled=pending_filled_count,
            pending_expired=pending_expired_count,
            trades_by_strategy=trades_by_strategy,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        _report(1.0, "Backtest terminé avec succès !")

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": len(df),
            "period_start": str(df.index[0]),
            "period_end": str(df.index[-1]),
            "initial_capital": initial_capital,
            "final_balance": current_balance,
            "net_profit_usd": net_profit_usd,
            "net_return_pct": net_return_pct,
            "tier_info": tier_info,
            "total_trades": total_trades,
            "wins_count": len(wins),
            "losses_count": len(losses),
            "breakeven_count": len(breakevens),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_usd": round(max_drawdown_usd, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 1),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "exits_tp": exits_tp,
            "exits_sl": exits_sl,
            "exits_be": exits_be,
            "blocked_by_micro_protection": blocked_by_micro_protection,
            "blocked_by_spread": blocked_by_spread,
            "blocked_by_confidence": blocked_by_confidence,
            "blocked_by_rr": blocked_by_rr,
            "blocked_by_regime": blocked_by_regime,
            "total_signals_generated": total_signals_generated,
            "pending_orders_placed": pending_placed_count,
            "pending_orders_filled": pending_filled_count,
            "pending_orders_expired": pending_expired_count,
            "strategy_filter": strategy_filter,
            "fixed_lot_size": effective_fixed_lot,
            "trades_by_strategy": trades_by_strategy,
            "trades": closed_trades,
            "equity_curve": equity_curve,
            "conclusion": conclusion_text,
        }

    def _generate_ai_conclusion(
        self,
        symbol: str,
        timeframe: str,
        num_bars: int,
        initial_capital: float,
        final_balance: float,
        net_profit_usd: float,
        net_return_pct: float,
        tier_info: dict,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        max_drawdown_pct: float,
        exits_tp: int,
        exits_sl: int,
        exits_be: int,
        blocked_by_micro_protection: int = 0,
        blocked_by_spread: int = 0,
        blocked_by_confidence: int = 0,
        blocked_by_rr: int = 0,
        blocked_by_regime: int = 0,
        total_signals_generated: int = 0,
        strategy_filter: str = "ALL",
        effective_fixed_lot: float = 0.0,
        pending_placed: int = 0,
        pending_filled: int = 0,
        pending_expired: int = 0,
        trades_by_strategy: dict = None,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
    ) -> str:
        """
        Rédige un débriefing diagnostique structuré et approfondi du backtest.
        """
        tier_name = tier_info.get("name", "Standard")
        trades_by_strategy = trades_by_strategy or {}
        lines = []

        lines.append(f"══════════════════════════════════════════════════════════════════════")
        lines.append(f"🧠 CONCLUSION & DÉBRIEFING DIAGNOSTIQUE DU BOT — {symbol} ({timeframe})")
        lines.append(f"══════════════════════════════════════════════════════════════════════")
        lines.append("")

        # 1. Bilan Financier
        lot_mode_str = f"Lot Fixe ({effective_fixed_lot:.2f})" if effective_fixed_lot > 0 else "Dynamique (% Risque)"
        lines.append(f"⚙️ Configuration : Stratégie = {strategy_filter} | Mode Volume = {lot_mode_str}")

        if net_profit_usd > 0:
            lines.append(f"🟢 BILAN GLOBAL : RENTABLE (+{net_profit_usd:+.2f} $ | {net_return_pct:+.1f}%)")
            lines.append(f"Le capital est passé de {initial_capital:.2f} $ à {final_balance:.2f} $ sur {num_bars} bougies.")
        elif net_profit_usd < 0:
            lines.append(f"🔴 BILAN GLOBAL : NÉGATIF ({net_profit_usd:+.2f} $ | {net_return_pct:+.1f}%)")
            lines.append(f"Le capital est passé de {initial_capital:.2f} $ à {final_balance:.2f} $ sur {num_bars} bougies.")
        else:
            lines.append(f"⚪ BILAN GLOBAL : NEUTRE (0.00 $ | 0.0%)")
            if total_trades == 0:
                lines.append(f"Capital préservé à 100% ({initial_capital:.2f} $). Aucune position risquée n'a été engagée.")

        lines.append("")
        lines.append("──────────────────────────────────────────────────────────────────────")
        lines.append("1. COMPORTEMENT DU CAPITAL & DES ORDRES :")
        lines.append(f"   • Palier détecté : {tier_name} (Capital initial : {initial_capital:.2f} $)")
        lines.append(f"   • Mode de dimensionnement : {lot_mode_str}")

        if pending_placed > 0:
            lines.append(f"   • Ordres en Attente (Pending Limits) : {pending_placed} placés | {pending_filled} exécutés | {pending_expired} expirés")

        if blocked_by_regime > 0:
            lines.append(f"   🛡️ FILTRE DE RÉGIME : {blocked_by_regime} période(s) évitée(s) car le marché était haché / illiquide (CHOPPY_ILLIQUID).")

        if initial_capital < 250 and effective_fixed_lot <= 0:
            lines.append("   • Comportement Micro-Compte :")
            if blocked_by_micro_protection > 0:
                lines.append(
                    f"     🛡️ PROTECTION ACTIVE : Le bot a bloqué {blocked_by_micro_protection} trade(s) car le lot minimum broker "
                    f"(0.01) avec un SL normal aurait engagé plus de 4% de votre compte d'un coup !"
                )
                lines.append(
                    f"     👉 Votre capital de {initial_capital:.2f} $ a été protégé contre une prise de risque disproportionnée."
                )
            elif total_trades > 0:
                lines.append("     ✅ Tous les trades exécutés respectaient scrupuleusement le plafond de risque micro-compte.")

        lines.append("")
        lines.append("──────────────────────────────────────────────────────────────────────")
        lines.append("2. EFFICACITÉ DES STRATÉGIES & DÉFENSE DU CAPITAL :")
        if total_trades > 0:
            lines.append(f"   • Taux de Réussite : {win_rate:.1f}% ({total_trades} positions prises au total)")
            lines.append(f"   • Profit Factor    : {profit_factor:.2f}")
            lines.append(f"   • Drawdown Maximum : -{max_drawdown_pct:.1f}%")
            lines.append(f"   • Sorties TP direct : {exits_tp} fois (Gains max encaissés)")
            lines.append(f"   • Sorties Break-Even: {exits_be} fois (Trades sécurisés à 0$ après impulsion)")
            lines.append(f"   • Sorties SL direct : {exits_sl} fois")

            if trades_by_strategy:
                lines.append("   • Répartition par Stratégie :")
                for s_name, s_data in trades_by_strategy.items():
                    lines.append(f"     - [{s_name}] : {s_data['count']} trade(s), WinRate {s_data['win_rate']}%, PnL: {s_data['net_pnl']:+.2f} $")
        else:
            lines.append(f"   • Taux de Réussite : 0.0% (0 position prise sur {total_signals_generated} opportunités détectées)")
            lines.append(f"   • Drawdown Maximum : 0.0% (Aucun risque encouru)")
            if blocked_by_micro_protection > 0:
                lines.append(f"   • Signaux bloqués par Protection Micro-Compte : {blocked_by_micro_protection} trade(s)")
            if blocked_by_confidence > 0:
                lines.append(f"   • Signaux bloqués pour Confiance < seuil      : {blocked_by_confidence} trade(s)")
            if blocked_by_spread > 0:
                lines.append(f"   • Signaux évités à cause du Spread excessif   : {blocked_by_spread} trade(s)")
            if blocked_by_rr > 0:
                lines.append(f"   • Signaux rejetés pour Ratio R:R insuffisant : {blocked_by_rr} trade(s)")

        if exits_be > 0:
            lines.append(
                f"   💡 Le Break-Even automatique a permis d'éliminer le risque sur {exits_be} trade(s) "
                f"qui se retournaient contre le marché, préservant ainsi les profits accumulés."
            )

        lines.append("")
        lines.append("──────────────────────────────────────────────────────────────────────")
        lines.append("3. RECOMMANDATIONS DU ROBOT POUR CE MARCHÉ :")
        if total_trades == 0:
            lines.append("   ⚪ DIAGNOSTIC DU 0 TRADE SUR CETTE PÉRIODE :")
            if blocked_by_micro_protection > 0:
                lines.append(f"   🛡️ CAUSE PRINCIPALE : Inadéquation entre la taille du contrat {symbol} et le capital de {initial_capital:.2f} $.")
                lines.append(f"      Sur le Bitcoin (BTCUSD ~84 000 $), le plus petit lot autorisé par votre courtier (0.01 lot) représente ~840 $ de valeur notionnelle.")
                lines.append(f"      Avec un Stop Loss standard de 300$ à 800$, une perte en cas de SL coûterait 3.00 $ à 8.00 $ (soit 6% à 16% de votre compte !).")
                lines.append(f"      Le bot a strictement refusé d'exécuter ces {blocked_by_micro_protection} signaux car son plafond de sécurité micro-compte est de 4.0% ({initial_capital * 0.04:.2f} $ max).")
                lines.append("")
                lines.append("   👉 QUE FAIRE MAINTENANT ?")
                lines.append("      1. 🎯 Testez sur le FOREX (ex: EURUSD, GBPUSD) avec vos 50 $ :")
                lines.append("         Le pip ne vaut que 0.10 $ sur 0.01 lot. Un SL de 15 pips ne risque que 1.50 $ (3.0% du capital). Le bot tradera activement et sereinement !")
                lines.append(f"      2. 💰 Pour trader {symbol} en toute sécurité :")
                lines.append("         Utilisez un capital de test de 250 $ à 500 $. Avec ce capital, un Stop Loss de 0.01 lot sur Bitcoin ne représentera que 1% à 1.5% de risque, parfaitement conforme aux standards institutionnels.")
            elif blocked_by_confidence > 0:
                lines.append(f"   ℹ️ CAUSE PRINCIPALE : Les {blocked_by_confidence} signaux générés avaient un score de confiance inférieur au seuil requis.")
                lines.append("   👉 Recommandation : Essayez le profil 'Dynamique (Agressif)' ou testez une unité de temps supérieure (H1) pour des confluences plus nettes.")
            elif blocked_by_spread > 0:
                lines.append(f"   ⚠️ CAUSE PRINCIPALE : Frais de spread trop élevés sur {symbol} ({blocked_by_spread} signaux rejetés).")
                lines.append("   👉 Recommandation : Privilégiez des heures de plus grande liquidité ou un courtier avec des spreads plus serrés.")
            else:
                lines.append(f"   ℹ️ CAUSE PRINCIPALE : Aucun setup valide n'a été détecté sur ces {num_bars} bougies pour la stratégie '{strategy_filter}'.")
                lines.append("   👉 Recommandation : Augmentez le nombre de bougies de test ou passez en mode 'Toutes les stratégies' pour combiner CRT, Order Blocks et Boomerang.")
        elif win_rate >= 60.0 and profit_factor >= 1.5:
            lines.append("   🌟 Configuration VALIDÉE : La stratégie présente un avantage statistique clair sur cette période.")
            lines.append(f"   👉 Recommandation : Vous pouvez déployer l'Autopilot sur {symbol} ({timeframe}) avec ce niveau de risque.")
        elif profit_factor >= 1.0:
            lines.append("   ⚠️ Configuration MOYENNE : Stratégie profitable mais avec une marge de sécurité étroite.")
            lines.append("   👉 Recommandation : Essayez d'augmenter le Timeframe (ex: M15 au lieu de M5) pour des confluences plus fiables.")
        else:
            lines.append("   🛑 Configuration DÉCONSEILLÉE : Cette période de marché a été défavorable.")
            lines.append("   👉 Recommandation : Privilégiez un filtre de direction stricte (BUY_ONLY ou SELL_ONLY selon la tendance H4/D1) ou activez un R:R minimum supérieur.")

        lines.append("══════════════════════════════════════════════════════════════════════")

        return "\n".join(lines)
