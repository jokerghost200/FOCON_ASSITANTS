"""
Base Strategy Module pour les Stratégies des Traders Professionnels.
Définit la structure commune des signaux, des filtres et des calculs géométriques/temporels
(Fibonacci, Swings, Imbalances, Sessions, ATR, OPR).
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProSignal:
    """Structure standardisée d'un signal émis par un profil de trader pro."""
    trader: str                      # Nom du trader (ex: "Nada FX", "Corentin Trading")
    profile_id: str                  # Identifiant normalisé ("nada", "corentin", "benjamin", "romain")
    direction: str                   # "BUY" ou "SELL"
    entry: float                     # Prix d'entrée recommandé (ou limite)
    sl: float                        # Stop Loss
    tp: float                        # Take Profit
    rr: float                        # Ratio Risque / Rendement
    is_pending: bool = False         # True si ordre limite en attente
    order_type: str = "MARKET"       # "MARKET", "BUY_LIMIT", "SELL_LIMIT"
    expiration_bars: int = 4         # Expiration en barres si pending
    expiration_seconds: int = 3600   # Expiration en secondes si pending
    cluster_orders: Optional[List[Dict[str, Any]]] = None  # Ordres en grappe pour Benjamin
    reason: str = ""                 # Explication détaillée du setup
    confluences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    be_at_r: Optional[float] = None  # Déclencheur du Break Even en multiple de R (Romain)
    trail: bool = False              # Trailing stop actif (Romain)
    decay_risk_mult: float = 1.0     # Multiplicateur de risque dégressif (Benjamin)


class BaseTraderStrategy:
    """Classe de base fournissant les utilitaires mathématiques, temporels et structurels."""

    def __init__(self, profile_id: str, name: str, config: Dict[str, Any]):
        self.profile_id = profile_id
        self.name = name
        self.config = config or {}

    def is_symbol_supported(self, symbol: str) -> bool:
        """Vérifie si le symbole fait partie des actifs ciblés par la stratégie."""
        symbols = self.config.get("symbols", [])
        if not symbols:
            return True
        sym_clean = str(symbol).upper().replace(".R", "").replace("M", "").strip()
        for s in symbols:
            s_clean = str(s).upper().replace(".R", "").replace("M", "").strip()
            if s_clean in sym_clean or sym_clean in s_clean:
                return True
        return False

    # ─── UTILITAIRES TEMPORELS ET SESSIONS ───

    @staticmethod
    def parse_time_str(time_str: str) -> time:
        """Convertit 'HH:MM' en objet datetime.time."""
        parts = [int(p) for p in time_str.strip().split(":")]
        return time(hour=parts[0], minute=parts[1] if len(parts) > 1 else 0)

    @classmethod
    def in_time_window(cls, current_dt: datetime, start_str: str, end_str: str) -> bool:
        """Vérifie si current_dt se situe entre start_str et end_str (format 'HH:MM')."""
        try:
            t_curr = current_dt.time()
            t_start = cls.parse_time_str(start_str)
            t_end = cls.parse_time_str(end_str)
            if t_start <= t_end:
                return t_start <= t_curr <= t_end
            else:  # Cas chevauchement de minuit
                return t_curr >= t_start or t_curr <= t_end
        except Exception as e:
            logger.debug(f"[BaseTraderStrategy] Erreur calcul fenêtre horaire : {e}")
            return True

    @classmethod
    def minutes_since_time(cls, current_dt: datetime, ref_time_str: str) -> float:
        """Retourne le nombre de minutes écoulées depuis ref_time_str aujourd'hui."""
        try:
            ref_t = cls.parse_time_str(ref_time_str)
            ref_dt = current_dt.replace(hour=ref_t.hour, minute=ref_t.minute, second=0, microsecond=0)
            diff = (current_dt - ref_dt).total_seconds() / 60.0
            return diff
        except Exception:
            return 999.0

    # ─── CALCULS GÉOMÉTRIQUES : FIBONACCI ───

    @staticmethod
    def calculate_fibonacci_levels(
        swing_low: float, swing_high: float, direction: str
    ) -> Dict[str, float]:
        """
        Calcule les niveaux de retracement de Fibonacci pour une impulsion :
        - Pour un mouvement haussier (direction='BUY') : le retracement descend depuis le swing_high.
          0.50  = high - 0.50 * diff
          0.618 = high - 0.618 * diff (Golden Pocket)
        - Pour un mouvement baissier (direction='SELL') : le retracement monte depuis le swing_low.
          0.50  = low + 0.50 * diff
          0.618 = low + 0.618 * diff
        """
        diff = max(0.0, swing_high - swing_low)
        if diff <= 0:
            return {
                "0.0": swing_high, "0.382": swing_high, "0.50": swing_high,
                "0.618": swing_high, "0.786": swing_high, "1.0": swing_low
            }

        dir_upper = str(direction).upper()
        if dir_upper == "BUY":
            return {
                "0.0": swing_high,
                "0.382": swing_high - (0.382 * diff),
                "0.50": swing_high - (0.50 * diff),
                "0.618": swing_high - (0.618 * diff),
                "0.786": swing_high - (0.786 * diff),
                "1.0": swing_low,
                "diff": diff,
            }
        else:
            return {
                "0.0": swing_low,
                "0.382": swing_low + (0.382 * diff),
                "0.50": swing_low + (0.50 * diff),
                "0.618": swing_low + (0.618 * diff),
                "0.786": swing_low + (0.786 * diff),
                "1.0": swing_high,
                "diff": diff,
            }

    # ─── INDICATEURS STRUCTURELS : SWINGS, IMBALANCES, SESSIONS ───

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'Average True Range (ATR) sur le dataframe."""
        if df is None or len(df) < period + 1:
            return 0.001
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            tr_list = []
            for i in range(1, len(df)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i - 1])
                lc = abs(low[i] - close[i - 1])
                tr_list.append(max(hl, hc, lc))
            if not tr_list:
                return 0.001
            atr = float(pd.Series(tr_list).rolling(period).mean().iloc[-1])
            return atr if not np.isnan(atr) and atr > 0 else 0.001
        except Exception:
            return 0.001

    @staticmethod
    def detect_swings(df: pd.DataFrame, lookback: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Détecte les Swing Highs et Swing Lows fractals récents."""
        swing_highs = []
        swing_lows = []
        if df is None or len(df) < (lookback * 2 + 1):
            return {"highs": swing_highs, "lows": swing_lows}

        highs = df['high'].values
        lows = df['low'].values
        times = df['time'].values if 'time' in df.columns else list(range(len(df)))

        for i in range(lookback, len(df) - lookback):
            # Check swing high
            is_sh = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_sh = False
                    break
            if is_sh:
                swing_highs.append({"index": i, "price": float(highs[i]), "time": times[i]})

            # Check swing low
            is_sl = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_sl = False
                    break
            if is_sl:
                swing_lows.append({"index": i, "price": float(lows[i]), "time": times[i]})

        return {"highs": swing_highs, "lows": swing_lows}

    @staticmethod
    def detect_session_range(
        df: pd.DataFrame, start_hour: int, end_hour: int
    ) -> Optional[Dict[str, float]]:
        """
        Extrait le plus haut, le plus bas et le prix d'ouverture d'une session donnée
        (ex: session de Londres entre 08:00 et 15:00).
        """
        if df is None or df.empty or 'time' not in df.columns:
            return None
        try:
            # Filtrer les barres du jour dont l'heure est comprise entre start_hour et end_hour
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy['time']):
                df_copy['dt'] = pd.to_datetime(df_copy['time'])
            else:
                df_copy['dt'] = df_copy['time']

            # Garder la session la plus récente
            latest_date = df_copy['dt'].iloc[-1].date()
            day_mask = (df_copy['dt'].dt.date == latest_date)
            hour_mask = (df_copy['dt'].dt.hour >= start_hour) & (df_copy['dt'].dt.hour < end_hour)
            session_bars = df_copy[day_mask & hour_mask]

            if session_bars.empty:
                # Fallback : chercher sur le jour précédent si la session du jour n'a pas encore eu lieu
                unique_dates = df_copy['dt'].dt.date.unique()
                if len(unique_dates) >= 2:
                    prev_date = unique_dates[-2]
                    session_bars = df_copy[(df_copy['dt'].dt.date == prev_date) & hour_mask]

            if session_bars.empty:
                return None

            return {
                "high": float(session_bars['high'].max()),
                "low": float(session_bars['low'].min()),
                "open": float(session_bars['open'].iloc[0]),
                "close": float(session_bars['close'].iloc[-1]),
                "range": float(session_bars['high'].max() - session_bars['low'].min()),
            }
        except Exception as e:
            logger.debug(f"[BaseTraderStrategy] Erreur calcul range de session : {e}")
            return None

    @staticmethod
    def detect_imbalances_fvg(df: pd.DataFrame, lookback: int = 30) -> List[Dict[str, Any]]:
        """
        Détecte les Fair Value Gaps (FVG) / Imbalances en 3 bougies :
        - Bullish FVG : low de la bougie i > high de la bougie i-2. Zone = [high[i-2], low[i]]
        - Bearish FVG : high de la bougie i < low de la bougie i-2. Zone = [high[i], low[i-2]]
        """
        imbalances = []
        if df is None or len(df) < 3:
            return imbalances

        start_idx = max(2, len(df) - lookback)
        for i in range(start_idx, len(df)):
            high_prev2 = float(df['high'].iloc[i - 2])
            low_prev2 = float(df['low'].iloc[i - 2])
            low_curr = float(df['low'].iloc[i])
            high_curr = float(df['high'].iloc[i])

            # Bullish FVG
            if low_curr > high_prev2:
                gap = low_curr - high_prev2
                imbalances.append({
                    "type": "BULLISH_FVG",
                    "direction": "BUY",
                    "top": low_curr,
                    "bottom": high_prev2,
                    "midpoint": (low_curr + high_prev2) / 2.0,
                    "gap_size": gap,
                    "bar_index": i,
                })

            # Bearish FVG
            elif high_curr < low_prev2:
                gap = low_prev2 - high_curr
                imbalances.append({
                    "type": "BEARISH_FVG",
                    "direction": "SELL",
                    "top": low_prev2,
                    "bottom": high_curr,
                    "midpoint": (low_prev2 + high_curr) / 2.0,
                    "gap_size": gap,
                    "bar_index": i,
                })

        return imbalances

    def evaluate(
        self,
        symbol: str,
        df_primary: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        df_ltf: Optional[pd.DataFrame] = None,
        spread: float = 0.0,
        min_distance: float = 0.0,
        tech_features: Optional[Dict[str, Any]] = None,
        amd_context: Optional[Dict[str, Any]] = None,
        order_flow_confluence: Optional[Any] = None,
        current_dt: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Optional[ProSignal]:
        """
        Méthode principale à surcharger par chaque stratégie de trader.
        Retourne un ProSignal si un setup valide est identifié, sinon None.
        """
        raise NotImplementedError
