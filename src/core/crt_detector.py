"""
CRT Detector (Candle Range Theory)
==================================
Module de détection algorithmique conforme au guide CRT (Candle Range Theory).
Modèle à 3 bougies :
  C1 : Bougie de range (référence HTF : H4, H1 ou Daily)
  C2 : Bougie de manipulation (sweep d'une extrémité + clôture stricte DANS le range de C1)
  C3 : Bougie de distribution (visée des 50% Equilibrium puis de la borne opposée)

Garanties de robustesse :
- Bougies closes uniquement (index MT5 1 et 2, soit iloc[-2] et iloc[-3], 0 repaint)
- Rejet systématique des doubles dépassements ambigus (mèches des deux côtés)
- Filtres de taille minimale du sweep et du range (multiples ATR)
- Horaires des Killzones basés sur zoneinfo ("America/New_York")
- Mode d'exécution Conservateur (attente clôture complète de C2)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")


@dataclass
class CRTSetup:
    direction: str          # "BUY" ou "SELL"
    c1_high: float          # Borne haute du range C1
    c1_low: float           # Borne basse du range C1
    c1_mid: float           # 50% Equilibrium du range C1
    c2_high: float          # Extrême haut de C2
    c2_low: float           # Extrême bas de C2
    c2_close: float         # Clôture de C2
    sweep_price: float      # Prix extrême du sweep
    sweep_depth: float      # Profondeur du sweep
    sl_raw: float           # SL brut (mèche de C2)
    tp1: float              # 50% EQ de C1
    tp2: float              # Borne opposée de C1
    in_killzone: bool       # Si le sweep a eu lieu pendant une Killzone
    killzone_name: str      # Nom de la killzone ("London", "New York", "None")
    c1_time: Any = None
    c2_time: Any = None


class CRTDetector:
    """
    Détecteur algorithmique des setups Candle Range Theory (CRT).
    """

    def __init__(
        self,
        min_sweep_atr_mult: float = 0.10,
        min_range_atr_mult: float = 0.50,
        min_rr_tp1: float = 1.5,
    ):
        self.min_sweep_atr_mult = min_sweep_atr_mult
        self.min_range_atr_mult = min_range_atr_mult
        self.min_rr_tp1 = min_rr_tp1

    @staticmethod
    def check_killzone(dt: Optional[datetime]) -> Dict[str, Any]:
        """
        Vérifie si une heure donnée se situe dans une Killzone institutionnelle
        en heure officielle de New York (gère le décalage été/hiver automatiquement).
        - Londres : 02:00 à 05:00 NY
        - New York : 07:00 à 10:00 NY
        """
        if dt is None:
            return {"in_killzone": False, "name": "None"}

        # Si le datetime est naïf, on l'assume UTC
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
        else:
            dt_utc = dt.astimezone(ZoneInfo("UTC"))

        dt_ny = dt_utc.astimezone(NY_TZ)
        hour = dt_ny.hour
        minute = dt_ny.minute
        time_decimal = hour + minute / 60.0

        if 2.0 <= time_decimal <= 5.0:
            return {"in_killzone": True, "name": "London"}
        elif 7.0 <= time_decimal <= 10.0:
            return {"in_killzone": True, "name": "New York"}

        return {"in_killzone": False, "name": "None"}

    def detect(
        self,
        df_htf: pd.DataFrame,
        atr: Optional[float] = None,
        spread: float = 0.0,
    ) -> Optional[CRTSetup]:
        """
        Analyse un DataFrame HTF (H4, H1 ou Daily).
        Exige au moins 3 bougies.
        
        RÈGLE SÉCURITÉ BOUGIER FERMÉES :
        df_htf.iloc[-1] = Bougie courante en cours de formation (non close, index MT5 0) -> JAMAIS évaluée pour C2.
        df_htf.iloc[-2] = Bougie C2 (dernière bougie close, index MT5 1).
        df_htf.iloc[-3] = Bougie C1 (bougie de référence close, index MT5 2).
        """
        if df_htf is None or len(df_htf) < 3:
            return None

        # Extraction des deux bougies closes consécutives
        c1 = df_htf.iloc[-3]
        c2 = df_htf.iloc[-2]

        c1_high = float(c1['high'])
        c1_low = float(c1['low'])
        c1_close = float(c1['close'])
        c1_range = c1_high - c1_low

        c2_high = float(c2['high'])
        c2_low = float(c2['low'])
        c2_close = float(c2['close'])

        if c1_range <= 0:
            return None

        # ─── 1. FILTRE DE TAILLE DU RANGE (évite les ranges écrasés sans liquidité)
        if atr and atr > 0:
            if c1_range < (self.min_range_atr_mult * atr):
                logger.debug(f"[CRT] Range C1 trop petit ({c1_range:.5f} < {self.min_range_atr_mult * atr:.5f})")
                return None

        # ─── 2. REJET SI DOUBLE DÉPASSEMENT (AMBIGUÏTÉ) ──────────────────────
        # Si C2 a balayé à la fois le High ET le Low de C1, signal invalide/ambigu
        if (c2_low < c1_low) and (c2_high > c1_high):
            logger.debug("[CRT] Rejet: C2 a dépassé à la fois le High et le Low de C1 (Double sweep ambigu)")
            return None

        # ─── 3. ANALYSE DU SWEEP & CLÔTURE STRICTE DANS LE RANGE ──────────────
        swept_low = (c2_low < c1_low) and (c2_close > c1_low) and (c2_close < c1_high)
        swept_high = (c2_high > c1_high) and (c2_close < c1_high) and (c2_close > c1_low)

        if not swept_low and not swept_high:
            # Pas de sweep ou clôture hors du range (cassure réelle / invalidation)
            return None

        # ─── 4. PROFONDEUR MINIMALE DU SWEEP (anti micro-mèche / bruit) ───────
        if swept_low:
            sweep_depth = c1_low - c2_low
            direction = "BUY"
            sl_raw = c2_low
            tp1 = (c1_high + c1_low) / 2.0
            tp2 = c1_high
            sweep_price = c2_low
        else:
            sweep_depth = c2_high - c1_high
            direction = "SELL"
            sl_raw = c2_high
            tp1 = (c1_high + c1_low) / 2.0
            tp2 = c1_low
            sweep_price = c2_high

        if atr and atr > 0:
            if sweep_depth < (self.min_sweep_atr_mult * atr):
                logger.debug(f"[CRT] Sweep trop superficiel ({sweep_depth:.5f} < {self.min_sweep_atr_mult * atr:.5f})")
                return None

        # ─── 5. KILLZONE CHECK ────────────────────────────────────────────────
        c2_time = getattr(c2, 'name', None)
        if not isinstance(c2_time, datetime) and 'time' in c2:
            c2_time = c2['time']

        kz_info = self.check_killzone(c2_time if isinstance(c2_time, datetime) else None)

        c1_time = getattr(c1, 'name', None)
        if not isinstance(c1_time, datetime) and 'time' in c1:
            c1_time = c1['time']

        mid_eq = round((c1_high + c1_low) / 2.0, 5)

        return CRTSetup(
            direction=direction,
            c1_high=round(c1_high, 5),
            c1_low=round(c1_low, 5),
            c1_mid=mid_eq,
            c2_high=round(c2_high, 5),
            c2_low=round(c2_low, 5),
            c2_close=round(c2_close, 5),
            sweep_price=round(sweep_price, 5),
            sweep_depth=round(sweep_depth, 5),
            sl_raw=round(sl_raw, 5),
            tp1=mid_eq,
            tp2=round(tp2, 5),
            in_killzone=kz_info["in_killzone"],
            killzone_name=kz_info["name"],
            c1_time=c1_time,
            c2_time=c2_time,
        )
