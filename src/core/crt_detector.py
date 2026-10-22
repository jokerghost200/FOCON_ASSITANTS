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
    # ── Extensions Guide Universel Pending Orders (ICT/SMC) ──
    order_type: str = "DEAL"             # "BUY_LIMIT", "SELL_LIMIT" ou "DEAL"
    entry_price: Optional[float] = None  # Milieu du FVG
    sl: Optional[float] = None           # Stop-Loss avec marge ATR/spread
    tp: Optional[float] = None           # Take-Profit validé (opposé ou 2R)
    tp_fallback: Optional[float] = None  # Take-Profit 2R de remplacement
    rr: Optional[float] = None           # Ratio R:R effectif
    has_mss: bool = False                # Si MSS confirmé
    mss_price: Optional[float] = None    # Niveau de cassure MSS
    has_fvg: bool = False                # Si FVG valide présent
    fvg_low: Optional[float] = None      # Borne basse du FVG
    fvg_high: Optional[float] = None     # Borne haute du FVG
    expiration_bars: int = 4             # Expiration en nombre de bougies LTF
    expiration_seconds: int = 3600       # Expiration en secondes
    is_pending: bool = False             # True si ordre limite en attente
    reason: str = ""
    # ── Extensions Order Flow / Microstructure ──
    order_flow_confluence: Optional[Any] = None  # Synthèse de confluence Order Flow
    order_flow_confirmed: bool = False           # True si Order Flow valide le setup
    order_flow_score: float = 0.0                # Score 0 - 100 de confluence Order Flow


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

    def find_confirmed_swings(self, df: pd.DataFrame, lookback: int = 2) -> list:
        """
        Trouve les Swing Highs et Swing Lows confirmés avec `lookback` bougies de chaque côté.
        RÈGLE SÉCURITÉ ZÉRO-LOOKAHEAD :
        Un swing à l'indice i n'est confirmé qu'après la clôture de la bougie (i + lookback).
        """
        n = len(df)
        swings = []
        if n < (lookback * 2 + 1):
            return swings

        for i in range(lookback, n - lookback):
            confirm_idx = i + lookback
            h_i = float(df['high'].iloc[i])
            l_i = float(df['low'].iloc[i])

            # Swing High
            is_high = True
            for k in range(1, lookback + 1):
                if float(df['high'].iloc[i - k]) >= h_i or float(df['high'].iloc[i + k]) >= h_i:
                    is_high = False
                    break
            if is_high:
                swings.append({
                    "type": "HIGH",
                    "price": h_i,
                    "index": i,
                    "confirmed_index": confirm_idx,
                })

            # Swing Low
            is_low = True
            for k in range(1, lookback + 1):
                if float(df['low'].iloc[i - k]) <= l_i or float(df['low'].iloc[i + k]) <= l_i:
                    is_low = False
                    break
            if is_low:
                swings.append({
                    "type": "LOW",
                    "price": l_i,
                    "index": i,
                    "confirmed_index": confirm_idx,
                })

        return swings

    def detect_smc_pending(
        self,
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        atr_ltf: Optional[float] = None,
        spread: float = 0.0,
        swing_lookback: int = 2,
        mss_max_bars: int = 3,
        min_rr: float = 1.5,
        expiration_bars: int = 4,
    ) -> Optional[CRTSetup]:
        """
        Détection complète du modèle universel CRT + ICT/SMC pour ordres en attente (Pending Limits).
        Conforme au Guide Universel (Sections 3, 4 et 7) :
        1. Range HTF de référence (H, L, M) sur la dernière bougie HTF close.
        2. Balayage (sweep) en LTF de H ou L, puis réintégration en clôture.
        3. Confirmation MSS dans les 3 bougies LTF suivant la réintégration.
        4. FVG créé par le déplacement impulsif du MSS (milieu du FVG = prix d'entrée LIMIT).
        5. Stop-Loss au-delà du sweep + marge max(spread, 0.1 * ATR).
        6. Take-Profit à la liquidité opposée (ou 2R fallback) avec filtre R:R >= 1.5.
        7. Expiration paramétrable (4 bougies LTF par défaut).
        """
        if df_htf is None or len(df_htf) < 2 or df_ltf is None or len(df_ltf) < 8:
            return None

        # 1. RANGE HTF DE RÉFÉRENCE (dernière bougie HTF clôturée : iloc[-2] si la dernière est active, sinon iloc[-1])
        htf_ref = df_htf.iloc[-2] if len(df_htf) >= 3 else df_htf.iloc[-1]
        c1_high = float(htf_ref['high'])
        c1_low = float(htf_ref['low'])
        c1_range = c1_high - c1_low
        if c1_range <= 0:
            return None
        c1_mid = round((c1_high + c1_low) / 2.0, 5)

        # 2. BOUGIES LTF FERMÉES UNIQUEMENT (exclure la bougie courante en cours de formation)
        closed_ltf = df_ltf.iloc[:-1].copy() if len(df_ltf) >= 2 else df_ltf.copy()
        if len(closed_ltf) < 6:
            return None

        # 3. SWINGS CONFIRMÉS SUR LTF
        swings = self.find_confirmed_swings(closed_ltf, lookback=swing_lookback)

        # 4. RECHERCHE DU SWEEP ET DE LA RÉINTÉGRATION
        search_window = min(len(closed_ltf), 24)
        sub_ltf = closed_ltf.iloc[-search_window:].reset_index(drop=True)
        
        sweep_low_val = None
        sweep_low_idx = None
        reintegrate_buy_idx = None

        sweep_high_val = None
        sweep_high_idx = None
        reintegrate_sell_idx = None

        for idx in range(len(sub_ltf)):
            row = sub_ltf.iloc[idx]
            h = float(row['high'])
            l = float(row['low'])
            c = float(row['close'])

            # Balayage du bas
            if l < c1_low:
                if sweep_low_val is None or l < sweep_low_val:
                    sweep_low_val = l
                    sweep_low_idx = idx

            # Réintégration haussière
            if sweep_low_val is not None and sweep_low_idx is not None and idx > sweep_low_idx:
                if c > c1_low and reintegrate_buy_idx is None:
                    reintegrate_buy_idx = idx

            # Balayage du haut
            if h > c1_high:
                if sweep_high_val is None or h > sweep_high_val:
                    sweep_high_val = h
                    sweep_high_idx = idx

            # Réintégration baissière
            if sweep_high_val is not None and sweep_high_idx is not None and idx > sweep_high_idx:
                if c < c1_high and reintegrate_sell_idx is None:
                    reintegrate_sell_idx = idx

        # Rejet si double-sweep non résolu ou aucun sweep réintégré
        if reintegrate_buy_idx is not None and reintegrate_sell_idx is not None:
            return None

        direction = None
        sweep_price = None
        reintegrate_idx = None
        if reintegrate_buy_idx is not None:
            direction = "BUY"
            sweep_price = sweep_low_val
            reintegrate_idx = reintegrate_buy_idx
        elif reintegrate_sell_idx is not None:
            direction = "SELL"
            sweep_price = sweep_high_val
            reintegrate_idx = reintegrate_sell_idx
        else:
            return None

        # 5. DÉTECTION DU MSS DANS LES mss_max_bars (3 BOUGIES SUIVANTES)
        bars_after_reintegrate = len(sub_ltf) - 1 - reintegrate_idx
        if bars_after_reintegrate > (mss_max_bars + 3):
            return None

        has_mss = False
        mss_bar_idx = None
        mss_price = None

        if direction == "BUY":
            relevant_highs = [
                s for s in swings 
                if s['type'] == 'HIGH' and s['confirmed_index'] <= (len(closed_ltf) - search_window + reintegrate_idx)
            ]
            if not relevant_highs:
                mss_price = float(sub_ltf['high'].iloc[max(0, reintegrate_idx - 5):reintegrate_idx].max())
            else:
                mss_price = float(relevant_highs[-1]['price'])

            for k in range(reintegrate_idx, min(len(sub_ltf), reintegrate_idx + mss_max_bars + 1)):
                if float(sub_ltf['close'].iloc[k]) > mss_price:
                    has_mss = True
                    mss_bar_idx = k
                    break

        elif direction == "SELL":
            relevant_lows = [
                s for s in swings 
                if s['type'] == 'LOW' and s['confirmed_index'] <= (len(closed_ltf) - search_window + reintegrate_idx)
            ]
            if not relevant_lows:
                mss_price = float(sub_ltf['low'].iloc[max(0, reintegrate_idx - 5):reintegrate_idx].min())
            else:
                mss_price = float(relevant_lows[-1]['price'])

            for k in range(reintegrate_idx, min(len(sub_ltf), reintegrate_idx + mss_max_bars + 1)):
                if float(sub_ltf['close'].iloc[k]) < mss_price:
                    has_mss = True
                    mss_bar_idx = k
                    break

        if not has_mss or mss_bar_idx is None:
            return None

        # 6. DÉTECTION DU FVG CRÉÉ PAR LE DÉPLACEMENT IMPULSIF DU MSS
        has_fvg = False
        fvg_low = None
        fvg_high = None

        for end_i in range(mss_bar_idx, max(mss_bar_idx - 3, 1), -1):
            if end_i < 2:
                continue
            b1 = sub_ltf.iloc[end_i - 2]
            b3 = sub_ltf.iloc[end_i]

            if direction == "BUY":
                if float(b3['low']) > float(b1['high']):
                    fvg_low = float(b1['high'])
                    fvg_high = float(b3['low'])
                    has_fvg = True
                    break
            else:
                if float(b3['high']) < float(b1['low']):
                    fvg_low = float(b3['high'])
                    fvg_high = float(b1['low'])
                    has_fvg = True
                    break

        if not has_fvg or fvg_low is None or fvg_high is None or (fvg_high - fvg_low) <= 0:
            return None

        # Invalidation du FVG si une clôture ultérieure a traversé son extrémité opposée
        for post_i in range(mss_bar_idx + 1, len(sub_ltf)):
            post_close = float(sub_ltf['close'].iloc[post_i])
            if direction == "BUY" and post_close < fvg_low:
                return None
            elif direction == "SELL" and post_close > fvg_high:
                return None

        # 7. CALCUL DES NIVEAUX DE PRIX SELON LE GUIDE UNIVERSEL
        atr = atr_ltf if (atr_ltf and atr_ltf > 0) else 0.0006
        marge = max(spread, 0.1 * atr)
        entry_price = round((fvg_low + fvg_high) / 2.0, 5)

        if direction == "BUY":
            sl = round(sweep_price - marge, 5)
            risk = entry_price - sl
            if risk <= 0:
                return None
            tp_principal = round(c1_high, 5)
            tp_fallback = round(entry_price + 2.0 * risk, 5)
            rr_principal = (tp_principal - entry_price) / risk
            order_type = "BUY_LIMIT"
        else:
            sl = round(sweep_price + marge, 5)
            risk = sl - entry_price
            if risk <= 0:
                return None
            tp_principal = round(c1_low, 5)
            tp_fallback = round(entry_price - 2.0 * risk, 5)
            rr_principal = (entry_price - tp_principal) / risk
            order_type = "SELL_LIMIT"

        if rr_principal >= min_rr:
            tp_final = tp_principal
            rr_final = round(rr_principal, 2)
        elif 2.0 >= min_rr:
            tp_final = tp_fallback
            rr_final = 2.0
        else:
            return None

        exp_seconds = expiration_bars * 900  # M15 par défaut (15m * 4 = 60m)

        c2_time = getattr(sub_ltf.iloc[reintegrate_idx], 'name', None)
        if not isinstance(c2_time, datetime) and 'time' in sub_ltf.iloc[reintegrate_idx]:
            c2_time = sub_ltf.iloc[reintegrate_idx]['time']
        kz = self.check_killzone(c2_time if isinstance(c2_time, datetime) else None)

        c2_ref = sub_ltf.iloc[-1]
        reason = (
            f"CRT Universal SMC ({order_type}) : Sweep à {sweep_price:.5f}, "
            f"MSS confirmé à {mss_price:.5f}, Entrée milieu FVG à {entry_price:.5f}, "
            f"SL {sl:.5f}, TP {tp_final:.5f} (R:R {rr_final:.2f})"
        )

        return CRTSetup(
            direction=direction,
            c1_high=c1_high,
            c1_low=c1_low,
            c1_mid=c1_mid,
            c2_high=float(c2_ref['high']),
            c2_low=float(c2_ref['low']),
            c2_close=float(c2_ref['close']),
            sweep_price=sweep_price,
            sweep_depth=round(abs(c1_low - sweep_price if direction == 'BUY' else sweep_price - c1_high), 5),
            sl_raw=sweep_price,
            tp1=c1_mid,
            tp2=tp_final,
            in_killzone=kz["in_killzone"],
            killzone_name=kz["name"],
            order_type=order_type,
            entry_price=entry_price,
            sl=sl,
            tp=tp_final,
            tp_fallback=tp_fallback,
            rr=rr_final,
            has_mss=True,
            mss_price=round(mss_price, 5),
            has_fvg=True,
            fvg_low=round(fvg_low, 5),
            fvg_high=round(fvg_high, 5),
            expiration_bars=expiration_bars,
            expiration_seconds=exp_seconds,
            is_pending=True,
            reason=reason,
        )
