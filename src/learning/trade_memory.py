"""
Moteur de mémoire épisodique et journal long-terme des trades.
Enregistre chaque trade avec son contexte complet de marché à l'ouverture,
son parcours en temps réel (MFE/MAE) et déclenche l'auto-critique à la clôture.
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    ticket: int
    symbol: str
    action: str
    lot: float
    entry_price: float
    sl: float
    tp: float
    open_time: float
    open_time_str: str
    snapshot: Dict[str, Any]
    
    # Suivi en cours
    max_profit_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    last_profit_usd: float = 0.0
    
    # Clôture
    status: str = "OPEN"  # OPEN | CLOSED
    close_time: Optional[float] = None
    close_time_str: Optional[str] = None
    close_price: Optional[float] = None
    final_profit_usd: Optional[float] = None
    exit_reason: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Réflexion & Auto-critique
    critique: Optional[str] = None
    lesson: Optional[str] = None
    tags: Optional[List[str]] = None


class TradeMemoryEngine:
    """
    Système central de mémoire des trades et d'auto-critique.
    Persiste les snapshots dans 'data/trade_memory.json' et orchestre l'apprentissage.
    """
    def __init__(
        self,
        file_path: str = "data/trade_memory.json",
        critic: Optional[ReflectiveCritic] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
    ):
        self.file_path = file_path
        self.critic = critic or ReflectiveCritic()
        self.weights_engine = weights_engine or AdaptiveWeightsEngine()
        self.trades: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Charge l'historique complet de la mémoire depuis le disque."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    trades_list = data.get("trades", [])
                    for t in trades_list:
                        ticket = t.get("ticket")
                        if ticket:
                            self.trades[int(ticket)] = t
                logger.info(f"🧠 [Mémoire Épisodique] {len(self.trades)} trades chargés depuis la mémoire long terme.")
            except Exception as e:
                logger.warning(f"Erreur chargement trade_memory.json ({e}).")
        else:
            self._save()

    def _save(self):
        """Sauvegarde les trades dans un fichier JSON de manière sécurisée."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "trades": list(self.trades.values()),
                }, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        except Exception as e:
            logger.error(f"Erreur sauvegarde trade_memory.json : {e}")

    def record_entry(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot: float,
        price: float,
        sl: float,
        tp: float,
        snapshot: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Enregistre le snapshot complet au moment exact de l'ouverture du trade.
        """
        now = time.time()
        import datetime
        now_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        is_pending = bool(snapshot.get('is_pending')) if snapshot else False
        initial_status = "PENDING" if is_pending else "OPEN"
        order_type_str = str(snapshot.get('order_type', action)).upper() if snapshot else action.upper()

        record = TradeRecord(
            ticket=int(ticket),
            symbol=symbol,
            action=action.upper(),
            lot=float(lot),
            entry_price=float(price),
            sl=float(sl),
            tp=float(tp),
            open_time=now,
            open_time_str=now_str,
            snapshot=snapshot or {},
            max_profit_usd=0.0,
            max_drawdown_usd=0.0,
            last_profit_usd=0.0,
            status=initial_status,
        )
        rec_dict = asdict(record)
        self.trades[int(ticket)] = rec_dict
        self._save()

        if is_pending:
            logger.info(f"🧠 [Mémoire] Ordre en attente #{ticket} sur {symbol} ({order_type_str}) mémorisé en statut PENDING.")
        else:
            logger.info(f"🧠 [Mémoire] Position ouverte #{ticket} sur {symbol} ({action}) enregistrée avec son contexte de marché.")
        return rec_dict

    def update_excursion(self, ticket: int, current_profit_usd: float):
        """
        Suit en continu l'excursion du trade pour analyser le comportement (MFE / MAE).
        """
        ticket = int(ticket)
        if ticket not in self.trades:
            return

        t = self.trades[ticket]
        if t.get("status") != "OPEN":
            return

        current_profit_usd = float(current_profit_usd or 0.0)
        t["last_profit_usd"] = current_profit_usd

        prev_max_p = float(t.get("max_profit_usd", 0.0) or 0.0)
        t["max_profit_usd"] = round(max(prev_max_p, current_profit_usd), 2)

        prev_max_dd = float(t.get("max_drawdown_usd", 0.0) or 0.0)
        t["max_drawdown_usd"] = round(min(prev_max_dd, current_profit_usd), 2)

    def record_exit(
        self,
        ticket: int,
        close_price: float,
        final_profit_usd: float,
        exit_reason: str = "MANUAL",
        trigger_recalibration: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Clôture le trade en mémoire, déclenche l'auto-critique et le recalibrage des poids.
        """
        ticket = int(ticket)
        import datetime
        now = time.time()
        now_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        if ticket not in self.trades:
            # Trade ouvert avant ou manuellement sur MT5 sans snapshot initial
            self.trades[ticket] = {
                "ticket": ticket,
                "symbol": "UNKNOWN",
                "action": "UNKNOWN",
                "lot": 0.01,
                "entry_price": float(close_price),
                "sl": 0.0,
                "tp": 0.0,
                "open_time": now,
                "open_time_str": now_str,
                "snapshot": {},
                "max_profit_usd": float(final_profit_usd),
                "max_drawdown_usd": 0.0,
                "last_profit_usd": float(final_profit_usd),
                "status": "OPEN",
            }

        t = self.trades[ticket]
        t["status"] = "CLOSED"
        t["close_time"] = now
        t["close_time_str"] = now_str
        t["close_price"] = float(close_price)
        t["final_profit_usd"] = round(float(final_profit_usd), 2)
        t["exit_reason"] = exit_reason

        open_time = float(t.get("open_time") or now)
        t["duration_seconds"] = round(now - open_time, 1)

        # ── 1. Lancement de l'Auto-Critique (Post-Mortem) ──
        try:
            analysis = self.critic.critique_trade(t)
            t["critique"] = analysis.get("critique")
            t["lesson"] = analysis.get("lesson")
            t["tags"] = analysis.get("tags", [])
            logger.info(
                f"🧠 [Auto-Critique Trade #{ticket} {t.get('symbol')}] "
                f"Gain: {t['final_profit_usd']:+.2f} $ | Leçon: {t['lesson']}"
            )
        except Exception as _ce:
            logger.warning(f"Erreur critique sur trade #{ticket}: {_ce}")

        self._save()

        # ── 2. Recalibrage des poids si demandé ──
        if trigger_recalibration and self.weights_engine:
            closed_trades = self.get_closed_trades()
            self.weights_engine.recalibrate(closed_trades)

        return t

    def get_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retourne les trades clôturés du plus récent au plus ancien."""
        closed = [t for t in self.trades.values() if t.get("status") == "CLOSED"]
        closed.sort(key=lambda x: x.get("close_time", 0) or 0, reverse=True)
        return closed[:limit]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Retourne la liste des trades actuellement ouverts en mémoire."""
        return [t for t in self.trades.values() if t.get("status") == "OPEN"]

    def get_recent_lessons(self, symbol: Optional[str] = None, limit: int = 4) -> List[str]:
        """
        Extrait les leçons apprises récemment, filtrables par symbole,
        à injecter directement dans le prompt de l'IA.
        """
        closed = self.get_closed_trades(limit=30)
        lessons = []
        for t in closed:
            if symbol and t.get("symbol") != symbol:
                continue
            lesson = t.get("lesson")
            critique = t.get("critique")
            sym = t.get("symbol", "")
            profit = t.get("final_profit_usd", 0.0)
            if lesson and lesson not in lessons:
                sign = "+" if profit >= 0 else ""
                lessons.append(f"[{sym} | {sign}{profit:.2f}$] {lesson}")
                if len(lessons) >= limit:
                    break
        return lessons

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques globales de performance et d'apprentissage."""
        closed = self.get_closed_trades(limit=200)
        if not closed:
            return {
                "total_closed": 0,
                "wins_count": 0,
                "losses_count": 0,
                "win_rate": 0.0,
                "total_profit_usd": 0.0,
                "best_trade_usd": 0.0,
                "worst_trade_usd": 0.0,
            }

        total = len(closed)
        wins = [t for t in closed if float(t.get("final_profit_usd", 0) or 0) > 0.05]
        losses = [t for t in closed if float(t.get("final_profit_usd", 0) or 0) < -0.05]
        profits = [float(t.get("final_profit_usd", 0) or 0) for t in closed]

        return {
            "total_closed": total,
            "wins_count": len(wins),
            "losses_count": len(losses),
            "win_rate": round((len(wins) / total) * 100, 1),
            "total_profit_usd": round(sum(profits), 2),
            "best_trade_usd": round(max(profits) if profits else 0.0, 2),
            "worst_trade_usd": round(min(profits) if profits else 0.0, 2),
        }

    def check_expectancy(self, symbol: str, session_filter: str, strategy: str) -> dict:
        """
        Calcule l'Expectancy (R-multiple) pour une combinaison (Symbole, Session, Stratégie).
        Expectancy = (WinRate * AvgWinR) + (LossRate * AvgLossR)
        Si Expectancy < 0, la stratégie n'est pas rentable pour ce contexte.
        """
        closed = self.get_closed_trades(limit=500)
        
        # Filtrer par symbole, session, et stratégie (tags/reason)
        filtered_trades = []
        for t in closed:
            if symbol and t.get("symbol") != symbol:
                continue
            
            snap = t.get("snapshot", {})
            t_session = snap.get("session", "UNKNOWN")
            if session_filter and session_filter != "UNKNOWN" and t_session != session_filter:
                continue
                
            has_ob = snap.get("has_ob", False)
            fvg = snap.get("fvg", "None")
            
            # Logique de matching de stratégie (noms internes et standards)
            matched_strategy = False
            strat_upper = str(strategy or "").upper()
            if strat_upper in ("OB+FVG", "OB_FVG_RETEST"):
                if has_ob and fvg != "None":
                    matched_strategy = True
            elif strat_upper in ("OB", "ORDER_BLOCK"):
                if has_ob:
                    matched_strategy = True
            elif strat_upper in ("BOOMERANG",):
                if snap.get("has_boomerang", False):
                    matched_strategy = True
            elif strat_upper in ("LIQUIDITY_SWEEP_MSS", "SWEEP"):
                if str(snap.get("sweep", "None")) != "None":
                    matched_strategy = True
            elif strat_upper in ("TREND_PULLBACK",):
                if snap.get("ema_dist_atr", 0.0) != 0.0 or snap.get("htf_trend", "NEUTRAL") != "NEUTRAL":
                    matched_strategy = True
            else:
                matched_strategy = True  # Fallback si chaîne générique ou inconnue
                
            if matched_strategy:
                filtered_trades.append(t)
                
        if len(filtered_trades) < 3:
            return {"expectancy_r": 0.0, "status": "NEUTRAL", "trades_count": len(filtered_trades)}
            
        wins = []
        losses = []
        for t in filtered_trades:
            profit_usd = float(t.get("final_profit_usd", 0) or 0)
            initial_rr = float(t.get("snapshot", {}).get("initial_rr", 1.0))
            if initial_rr <= 0:
                initial_rr = 1.0 # fallback
                
            # Compute roughly the R realized (if profit > 0, it won some R; if < 0 it lost 1R assuming full SL)
            # Pour faire simple: profit / risk. Mais on n'a pas le risk USD direct, 
            # donc on peut estimer que 1R = |sl - entry| * lot * (contract size etc).
            # Si on a stocké `risk_usd` dans snapshot, c'est mieux.
            risk_usd = float(t.get("snapshot", {}).get("risk_usd", 0.0))
            if risk_usd <= 0:
                # Approximation naive si le trade gagne, il gagne R, s'il perd, il perd 1R.
                r_realized = (profit_usd / max(abs(profit_usd), 1.0)) if profit_usd != 0 else 0
            else:
                r_realized = profit_usd / risk_usd
                
            if r_realized > 0:
                wins.append(r_realized)
            else:
                losses.append(r_realized)
                
        win_rate = len(wins) / len(filtered_trades)
        loss_rate = len(losses) / len(filtered_trades)
        
        avg_win_r = sum(wins) / len(wins) if wins else 0.0
        avg_loss_r = sum(losses) / len(losses) if losses else 0.0
        
        expectancy_r = (win_rate * avg_win_r) + (loss_rate * avg_loss_r)
        
        status = "NEUTRAL"
        if expectancy_r < -0.1:
            status = "DISABLED"
        elif expectancy_r > 0.1:
            status = "ACTIVE"
            
        return {
            "expectancy_r": round(expectancy_r, 2),
            "status": status,
            "trades_count": len(filtered_trades),
            "win_rate": round(win_rate * 100, 1)
        }

    def get_symbol_affinity(self, symbol: str, session: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule l'affinité statistique du robot avec une paire (et sa session horaire)
        d'après l'historique réel des trades clôturés :
        - Booste (+6% confiance) les paires "Championnes" (Winrate >= 60%, PnL > 0, >= 3 trades)
        - Pénalise (-12% confiance) les paires "Toxiques" (Winrate <= 30%, PnL < 0, >= 3 trades)
        """
        closed = self.get_closed_trades(limit=500)
        sym_upper = str(symbol or "").upper()
        sym_trades = [t for t in closed if str(t.get("symbol", "")).upper() == sym_upper]

        if len(sym_trades) < 3:
            return {
                "symbol": symbol,
                "tier": "NEUTRAL",
                "confidence_bonus": 0.0,
                "win_rate": 0.0,
                "net_profit_usd": 0.0,
                "trades_count": len(sym_trades),
            }

        profits = [float(t.get("final_profit_usd", 0.0) or 0.0) for t in sym_trades]
        wins = [p for p in profits if p > 0.05]
        win_rate = (len(wins) / len(sym_trades)) * 100.0
        net_pnl = sum(profits)

        bonus = 0.0
        tier = "NEUTRAL"
        if win_rate >= 60.0 and net_pnl > 0:
            bonus = 6.0
            tier = "CHAMPION"
        elif win_rate >= 52.0 and net_pnl > 0:
            bonus = 3.0
            tier = "FAVORABLE"
        elif win_rate <= 30.0 and net_pnl < 0:
            bonus = -12.0
            tier = "TOXIC"
        elif win_rate < 42.0 and net_pnl < 0:
            bonus = -6.0
            tier = "WEAK"

        # Affinage par session horaire si disponible
        if session:
            sess_trades = [
                t for t in sym_trades
                if str((t.get("snapshot") or {}).get("session", "")).upper() == str(session).upper()
            ]
            if len(sess_trades) >= 3:
                s_profits = [float(t.get("final_profit_usd", 0.0) or 0.0) for t in sess_trades]
                s_wins = [p for p in s_profits if p > 0.05]
                s_wr = (len(s_wins) / len(sess_trades)) * 100.0
                if s_wr >= 65.0 and sum(s_profits) > 0:
                    bonus += 2.0
                elif s_wr <= 25.0 and sum(s_profits) < 0:
                    bonus -= 4.0

        return {
            "symbol": symbol,
            "tier": tier,
            "confidence_bonus": round(bonus, 1),
            "win_rate": round(win_rate, 1),
            "net_profit_usd": round(net_pnl, 2),
            "trades_count": len(sym_trades),
        }
