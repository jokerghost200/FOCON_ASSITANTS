"""
Order Flow & Market Microstructure Engine
=========================================
Module complet d'analyse avancée du carnet d'ordres et des flux de transactions.
Permet au bot d'analyser nativement les indicateurs institutionnels autrefois réservés
à Bookmap, ATAS, Sierra Chart ou Quantower :
  1. Carnet d'ordres DOM (Depth of Market L2) et détection des murs de liquidité
  2. Cumulative Volume Delta (CVD) et détection des divergences d'absorption
  3. Profil de Volume (Volume Profile : POC, VAH, VAL, zones HVN et LVN)
  4. Graphiques Footprint et déséquilibres bid/ask (Stacked Imbalances)
  5. Détection de l'absorption passive aux extrêmes de sweep CRT
  6. Synthèse de confluence Order Flow pour confirmation du setup CRT + ICT/SMC
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    poc: float                          # Point of Control (prix au volume max)
    vah: float                          # Value Area High (70% du volume total)
    val: float                          # Value Area Low (70% du volume total)
    high_volume_nodes: List[float] = field(default_factory=list) # HVN (zones magnétiques)
    low_volume_nodes: List[float] = field(default_factory=list)  # LVN (zones de rejet / vides)
    total_volume: float = 0.0
    bins: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CVDResult:
    current_cvd: float                  # Valeur cumulée actuelle du Delta
    cvd_trend: str                      # "BULLISH", "BEARISH", "NEUTRAL"
    divergence: str                     # "BEARISH_DIVERGENCE", "BULLISH_DIVERGENCE", "NONE"
    buy_volume_total: float = 0.0       # Volume total acheteur agressif
    sell_volume_total: float = 0.0      # Volume total vendeur agressif
    delta_ratio: float = 0.0            # (Buy - Sell) / (Buy + Sell) [-1.0 à +1.0]
    series: Optional[List[float]] = None


@dataclass
class DOMLiquidityResult:
    available: bool                     # True si le courtier fournit le carnet L2
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bids_volume: float = 0.0
    asks_volume: float = 0.0
    imbalance_ratio: float = 0.0        # -1.0 à +1.0
    bid_walls: List[Dict[str, Any]] = field(default_factory=list)  # Gros blocs d'achat limites
    ask_walls: List[Dict[str, Any]] = field(default_factory=list)  # Gros blocs de vente limites
    nearest_bid_wall: Optional[float] = None
    nearest_ask_wall: Optional[float] = None
    reason: str = ""


@dataclass
class FootprintResult:
    stacked_imbalances: List[Dict[str, Any]] = field(default_factory=list)
    buying_imbalances_count: int = 0
    selling_imbalances_count: int = 0
    unfinished_auction: bool = False
    dominant_side: str = "NEUTRAL"      # "BUYERS", "SELLERS", "BALANCED"


@dataclass
class AbsorptionResult:
    is_absorbed: bool = False
    absorption_type: str = "NONE"       # "BUYERS_ABSORBED" (bearish), "SELLERS_ABSORBED" (bullish)
    level: float = 0.0
    absorbed_volume: float = 0.0
    confidence: float = 0.0
    reason: str = ""


@dataclass
class OrderFlowConfluence:
    symbol: str
    is_confirmed: bool                  # Confirmation globale Order Flow
    confluence_score: float             # Score composite 0 - 100
    cvd_confirmed: bool = False
    absorption_confirmed: bool = False
    dom_confirmed: bool = False
    volume_profile_confirmed: bool = False
    footprint_confirmed: bool = False
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    details: List[str] = field(default_factory=list)
    summary_badge: str = ""
    # ── Métadonnées de Qualité & Fiabilité des Données ──
    data_quality: str = "ESTIMATED_QUOTE_RULE"   # "REAL_OBSERVED", "ESTIMATED_QUOTE_RULE", "DEGRADED_STALE", "UNAVAILABLE"
    is_dom_available: bool = False               # True si carnet L2 fourni par le courtier
    is_real_volume: bool = False                 # True si volume réel en contrats (pas seulement tick count)
    is_synthetic: bool = False                   # True si ticks synthétisés à partir de bougies
    damping_applied: float = 1.0                 # Coefficient réducteur anti-surapprentissage
    dimensions_breakdown: Dict[str, float] = field(default_factory=dict) # Détail par dimension



class OrderFlowEngine:
    """
    Moteur unifié d'analyse des flux d'ordres et de microstructure de marché.
    """

    def __init__(
        self,
        vp_bins: int = 50,
        vp_value_area_pct: float = 0.70,
        imbalance_ratio: float = 3.0,
        dom_wall_multiplier: float = 2.0,
        min_confluence_score: float = 60.0,
    ):
        self.vp_bins = vp_bins
        self.vp_value_area_pct = vp_value_area_pct
        self.imbalance_ratio = imbalance_ratio
        self.dom_wall_multiplier = dom_wall_multiplier
        self.min_confluence_score = min_confluence_score

    # ─────────────────────────────────────────────────────────────
    # 1. PROFIL DE VOLUME (Volume Profile : POC, VAH, VAL, HVN, LVN)
    # ─────────────────────────────────────────────────────────────
    def calculate_volume_profile(
        self,
        data: pd.DataFrame,
        num_bins: Optional[int] = None,
        value_area_pct: Optional[float] = None,
    ) -> VolumeProfileResult:
        """
        Calcule la distribution du volume par tranche de prix.
        Fonctionne avec un DataFrame de ticks ('price', 'vol') ou de bougies ('high', 'low', 'close', 'volume').
        """
        bins_count = num_bins or self.vp_bins
        va_pct = value_area_pct or self.vp_value_area_pct

        if data is None or data.empty:
            return VolumeProfileResult(poc=0.0, vah=0.0, val=0.0)

        # ── Cas A: DataFrame de Ticks ──
        if 'price' in data.columns and 'vol' in data.columns:
            prices = data['price'].values
            volumes = data['vol'].values
            sides = data['side'].values if 'side' in data.columns else ['BUY'] * len(data)

            min_p = np.min(prices)
            max_p = np.max(prices)
            if min_p == max_p:
                return VolumeProfileResult(poc=float(min_p), vah=float(max_p), val=float(min_p), total_volume=float(np.sum(volumes)))

            bin_edges = np.linspace(min_p, max_p, bins_count + 1)
            bin_indices = np.digitize(prices, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bins_count - 1)

            bin_vols = np.zeros(bins_count)
            bin_buy_vols = np.zeros(bins_count)
            bin_sell_vols = np.zeros(bins_count)

            for i in range(len(prices)):
                idx = bin_indices[i]
                v = volumes[i]
                bin_vols[idx] += v
                if sides[i] == 'BUY':
                    bin_buy_vols[idx] += v
                else:
                    bin_sell_vols[idx] += v

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # ── Cas B: DataFrame de Bougies OHLCV ──
        else:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values if 'volume' in data.columns else data.get('tick_volume', pd.Series(np.ones(len(data)))).values

            min_p = np.min(lows)
            max_p = np.max(highs)
            if min_p == max_p:
                return VolumeProfileResult(poc=float(min_p), vah=float(max_p), val=float(min_p), total_volume=float(np.sum(volumes)))

            bin_edges = np.linspace(min_p, max_p, bins_count + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_vols = np.zeros(bins_count)
            bin_buy_vols = np.zeros(bins_count)
            bin_sell_vols = np.zeros(bins_count)

            # Heuristique TPO / Volume pour bougies : distribution du volume de chaque barre sur sa hauteur
            for i in range(len(data)):
                h = highs[i]
                l = lows[i]
                c = closes[i]
                v = volumes[i]
                if h == l:
                    idx = np.clip(np.digitize([c], bin_edges)[0] - 1, 0, bins_count - 1)
                    bin_vols[idx] += v
                else:
                    # Distribuer uniformément sur les tranches couvertes
                    mask = (bin_centers >= l) & (bin_centers <= h)
                    active_count = np.sum(mask)
                    if active_count > 0:
                        allocated = v / active_count
                        bin_vols[mask] += allocated
                        if c >= (h + l) / 2.0:
                            bin_buy_vols[mask] += allocated * 0.6
                            bin_sell_vols[mask] += allocated * 0.4
                        else:
                            bin_buy_vols[mask] += allocated * 0.4
                            bin_sell_vols[mask] += allocated * 0.6

        total_vol = float(np.sum(bin_vols))
        if total_vol <= 0:
            return VolumeProfileResult(poc=0.0, vah=0.0, val=0.0)

        # 1. Point of Control (POC)
        poc_idx = int(np.argmax(bin_vols))
        poc_price = float(bin_centers[poc_idx])

        # 2. Value Area (VAH / VAL couvrant va_pct, par défaut 70% du volume)
        target_vol = total_vol * va_pct
        accumulated_vol = bin_vols[poc_idx]
        up_idx = poc_idx
        down_idx = poc_idx

        while accumulated_vol < target_vol and (up_idx < bins_count - 1 or down_idx > 0):
            next_up_vol = bin_vols[up_idx + 1] if up_idx < bins_count - 1 else 0.0
            next_down_vol = bin_vols[down_idx - 1] if down_idx > 0 else 0.0

            if next_up_vol >= next_down_vol and next_up_vol > 0:
                up_idx += 1
                accumulated_vol += next_up_vol
            elif next_down_vol > 0:
                down_idx -= 1
                accumulated_vol += next_down_vol
            else:
                if up_idx < bins_count - 1:
                    up_idx += 1
                if down_idx > 0:
                    down_idx -= 1
                break

        val_price = float(bin_centers[down_idx])
        vah_price = float(bin_centers[up_idx])

        # 3. High Volume Nodes (HVN) et Low Volume Nodes (LVN)
        mean_vol = np.mean(bin_vols)
        std_vol = np.std(bin_vols)
        hvn_threshold = mean_vol + (0.8 * std_vol)
        lvn_threshold = max(0.0, mean_vol - (0.8 * std_vol))

        hvns = [float(bin_centers[i]) for i in range(bins_count) if bin_vols[i] >= hvn_threshold]
        lvns = [float(bin_centers[i]) for i in range(bins_count) if bin_vols[i] <= lvn_threshold and bin_vols[i] > 0]

        profile_bins = []
        for i in range(bins_count):
            profile_bins.append({
                "price": round(float(bin_centers[i]), 5),
                "volume": round(float(bin_vols[i]), 2),
                "buy_vol": round(float(bin_buy_vols[i]), 2),
                "sell_vol": round(float(bin_sell_vols[i]), 2),
            })

        return VolumeProfileResult(
            poc=round(poc_price, 5),
            vah=round(vah_price, 5),
            val=round(val_price, 5),
            high_volume_nodes=hvns,
            low_volume_nodes=lvns,
            total_volume=round(total_vol, 2),
            bins=profile_bins,
        )

    # ─────────────────────────────────────────────────────────────
    # 2. CUMULATIVE VOLUME DELTA (CVD & DIVERGENCES)
    # ─────────────────────────────────────────────────────────────
    def calculate_cvd(self, ticks_df: pd.DataFrame) -> CVDResult:
        """
        Calcule le Cumulative Volume Delta (CVD) et détecte les divergences avec le prix.
        """
        if ticks_df is None or ticks_df.empty or 'delta' not in ticks_df.columns:
            return CVDResult(current_cvd=0.0, cvd_trend="NEUTRAL", divergence="NONE")

        deltas = ticks_df['delta'].values
        prices = ticks_df['price'].values
        cvd_series = np.cumsum(deltas)
        current_cvd = float(cvd_series[-1])

        # Volumes totaux acheteur / vendeur
        buy_vol = float(np.sum(ticks_df.loc[ticks_df['side'] == 'BUY', 'vol'])) if 'side' in ticks_df.columns else float(np.sum(np.maximum(0, deltas)))
        sell_vol = float(np.sum(ticks_df.loc[ticks_df['side'] == 'SELL', 'vol'])) if 'side' in ticks_df.columns else float(np.sum(np.maximum(0, -deltas)))
        tot = buy_vol + sell_vol
        delta_ratio = float((buy_vol - sell_vol) / tot) if tot > 0 else 0.0

        # Tendance du CVD (pente sur la dernière moitié des données)
        n = len(cvd_series)
        split = max(2, n // 2)
        cvd_first_half = np.mean(cvd_series[:split])
        cvd_second_half = np.mean(cvd_series[split:])
        if cvd_second_half > cvd_first_half * 1.05:
            cvd_trend = "BULLISH"
        elif cvd_second_half < cvd_first_half * 0.95:
            cvd_trend = "BEARISH"
        else:
            cvd_trend = "NEUTRAL"

        # ── Détection de divergence CVD vs Prix ──
        # Segment 1 (antérieur) vs Segment 2 (récent)
        p1 = np.mean(prices[:split])
        p2 = np.mean(prices[split:])
        c1 = cvd_first_half
        c2 = cvd_second_half

        divergence = "NONE"
        # 1. Bearish Divergence: Prix monte (p2 > p1) mais CVD baisse (c2 < c1)
        # Signifie que les acheteurs s'épuisent ou que des ordres limites vendeurs absorbent tout
        if p2 > p1 and c2 < c1:
            divergence = "BEARISH_DIVERGENCE"
        # 2. Bullish Divergence: Prix baisse (p2 < p1) mais CVD monte (c2 > c1)
        # Signifie que les vendeurs agressifs se font absorber par des acheteurs limites
        elif p2 < p1 and c2 > c1:
            divergence = "BULLISH_DIVERGENCE"

        return CVDResult(
            current_cvd=round(current_cvd, 2),
            cvd_trend=cvd_trend,
            divergence=divergence,
            buy_volume_total=round(buy_vol, 2),
            sell_volume_total=round(sell_vol, 2),
            delta_ratio=round(delta_ratio, 3),
            series=cvd_series.tolist()[-100:] if len(cvd_series) > 0 else None,
        )

    # ─────────────────────────────────────────────────────────────
    # 3. DOM & CARNET D'ORDRES (Liquidity Walls & Imbalance)
    # ─────────────────────────────────────────────────────────────
    def analyze_dom(
        self,
        dom_snapshot: dict,
        wall_std_multiplier: Optional[float] = None,
    ) -> DOMLiquidityResult:
        """
        Analyse le carnet d'ordres L2, calcule le ratio de pression et repère les 'Murs de Liquidité'.
        """
        if not dom_snapshot or not dom_snapshot.get("available"):
            return DOMLiquidityResult(
                available=False,
                reason=dom_snapshot.get("reason", "DOM indisponible")
            )

        mult = wall_std_multiplier or self.dom_wall_multiplier
        bids = dom_snapshot.get("bids", [])
        asks = dom_snapshot.get("asks", [])
        bids_vol = float(dom_snapshot.get("bids_volume", 0.0))
        asks_vol = float(dom_snapshot.get("asks_volume", 0.0))
        imbalance = float(dom_snapshot.get("imbalance_ratio", 0.0))
        best_bid = dom_snapshot.get("best_bid")
        best_ask = dom_snapshot.get("best_ask")

        # Analyse des murs de liquidité par côté (bids vs asks séparés pour ne pas masquer un mur opposé)
        bid_vols = [b["volume"] for b in bids if b.get("volume", 0) > 0]
        ask_vols = [a["volume"] for a in asks if a.get("volume", 0) > 0]
        bid_walls = []
        ask_walls = []

        if bid_vols:
            b_mean = np.mean(bid_vols)
            b_std = np.std(bid_vols)
            b_thresh = b_mean + (mult * b_std) if b_std > 0 else b_mean * 1.5
            for b in bids:
                if b["volume"] >= b_thresh and b["volume"] > 0:
                    bid_walls.append({"price": b["price"], "volume": b["volume"], "type": "BID_WALL"})

        if ask_vols:
            a_mean = np.mean(ask_vols)
            a_std = np.std(ask_vols)
            a_thresh = a_mean + (mult * a_std) if a_std > 0 else a_mean * 1.5
            for a in asks:
                if a["volume"] >= a_thresh and a["volume"] > 0:
                    ask_walls.append({"price": a["price"], "volume": a["volume"], "type": "ASK_WALL"})

        nearest_bid_wall = bid_walls[0]["price"] if bid_walls else None
        nearest_ask_wall = ask_walls[0]["price"] if ask_walls else None

        return DOMLiquidityResult(
            available=True,
            best_bid=best_bid,
            best_ask=best_ask,
            bids_volume=bids_vol,
            asks_volume=asks_vol,
            imbalance_ratio=imbalance,
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            nearest_bid_wall=nearest_bid_wall,
            nearest_ask_wall=nearest_ask_wall,
            reason="DOM analysé avec succès",
        )

    # ─────────────────────────────────────────────────────────────
    # 4. FOOTPRINT & DÉSÉQUILIBRES (Stacked Imbalances)
    # ─────────────────────────────────────────────────────────────
    def calculate_footprint(
        self,
        ticks_df: pd.DataFrame,
        imbalance_ratio: Optional[float] = None,
        num_price_bins: int = 20,
    ) -> FootprintResult:
        """
        Regroupe les ticks par niveau de prix pour comparer le volume Bid vs Ask
        et repérer les déséquilibres diagonaux (Stacked Imbalances).
        """
        ratio_thresh = imbalance_ratio or self.imbalance_ratio

        if ticks_df is None or ticks_df.empty:
            return FootprintResult()

        prices = ticks_df['price'].values
        volumes = ticks_df['vol'].values
        sides = ticks_df['side'].values if 'side' in ticks_df.columns else ['BUY'] * len(ticks_df)

        min_p = np.min(prices)
        max_p = np.max(prices)
        if min_p == max_p:
            return FootprintResult()

        bin_edges = np.linspace(min_p, max_p, num_price_bins + 1)
        bin_indices = np.clip(np.digitize(prices, bin_edges) - 1, 0, num_price_bins - 1)

        bids_by_level = np.zeros(num_price_bins)
        asks_by_level = np.zeros(num_price_bins)

        for i in range(len(prices)):
            idx = bin_indices[i]
            v = volumes[i]
            if sides[i] == 'BUY':
                asks_by_level[idx] += v  # Acheteur agressif frappe le Ask
            else:
                bids_by_level[idx] += v  # Vendeur agressif frappe le Bid

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        stacked_imbalances = []
        buy_imb_cnt = 0
        sell_imb_cnt = 0

        # Vérification des déséquilibres diagonaux (Ask(level+1) vs Bid(level))
        for i in range(num_price_bins - 1):
            ask_vol_next = asks_by_level[i + 1]
            bid_vol_curr = bids_by_level[i]

            # Imbalance Acheteuse : Ask agressif très supérieur au Bid passif
            if bid_vol_curr > 0 and (ask_vol_next / bid_vol_curr) >= ratio_thresh:
                stacked_imbalances.append({
                    "price": round(float(bin_centers[i + 1]), 5),
                    "type": "BUYING_IMBALANCE",
                    "ratio": round(float(ask_vol_next / bid_vol_curr), 2),
                })
                buy_imb_cnt += 1

            # Imbalance Vendeuse : Bid agressif très supérieur à l'Ask passif
            elif ask_vol_next > 0 and (bid_vol_curr / ask_vol_next) >= ratio_thresh:
                stacked_imbalances.append({
                    "price": round(float(bin_centers[i]), 5),
                    "type": "SELLING_IMBALANCE",
                    "ratio": round(float(bid_vol_curr / ask_vol_next), 2),
                })
                sell_imb_cnt += 1

        dom_side = "BALANCED"
        if buy_imb_cnt > sell_imb_cnt * 1.5:
            dom_side = "BUYERS"
        elif sell_imb_cnt > buy_imb_cnt * 1.5:
            dom_side = "SELLERS"

        return FootprintResult(
            stacked_imbalances=stacked_imbalances,
            buying_imbalances_count=buy_imb_cnt,
            selling_imbalances_count=sell_imb_cnt,
            dominant_side=dom_side,
        )

    # ─────────────────────────────────────────────────────────────
    # 5. DÉTECTION D'ABSORPTION (Absorption aux Niveaux Clés)
    # ─────────────────────────────────────────────────────────────
    def detect_absorption(
        self,
        ticks_df: pd.DataFrame,
        key_level: float,
        direction: str,
        tolerance: float = 0.0005,
    ) -> AbsorptionResult:
        """
        Détecte si un volume agressif important a été absorbé passivement à proximité d'un niveau clé
        (par exemple le Sweep High ou Sweep Low de la bougie C2 du setup CRT).
        direction: 'BUY' (recherche absorption des vendeurs au creux) ou 'SELL' (absorption des acheteurs au sommet)
        """
        if ticks_df is None or ticks_df.empty or key_level is None:
            return AbsorptionResult()

        near_mask = np.abs(ticks_df['price'].values - key_level) <= tolerance
        near_ticks = ticks_df[near_mask]

        if near_ticks.empty or len(near_ticks) < 5:
            return AbsorptionResult(is_absorbed=False, level=key_level)

        near_vol = float(np.sum(near_ticks['vol']))
        near_deltas = near_ticks['delta'].values
        sum_delta = float(np.sum(near_deltas))

        # Rebond du prix par rapport au niveau clé dans les ticks finaux
        recent_prices = ticks_df['price'].values[-10:]
        price_moved_away = False

        if direction == "BUY":
            # Pour un achat CRT, les vendeurs agressifs poussent vers le bas (delta négatif fort)
            # mais le prix ne descend plus et remonte au-dessus du niveau clé
            current_p = float(recent_prices[-1])
            price_moved_away = current_p >= key_level
            if sum_delta < 0 and price_moved_away:
                conf = min(100.0, 50.0 + (abs(sum_delta) / (near_vol + 1e-6)) * 50.0)
                return AbsorptionResult(
                    is_absorbed=True,
                    absorption_type="SELLERS_ABSORBED",
                    level=key_level,
                    absorbed_volume=round(near_vol, 2),
                    confidence=round(conf, 1),
                    reason=f"Absorption vendeuse: delta négatif ({sum_delta:.1f}) absorbé au creux {key_level:.5f}, rebond vers {current_p:.5f}",
                )

        elif direction == "SELL":
            # Pour une vente CRT, les acheteurs agressifs poussent vers le haut (delta positif fort)
            # mais le prix est bloqué et retombe sous le niveau clé
            current_p = float(recent_prices[-1])
            price_moved_away = current_p <= key_level
            if sum_delta > 0 and price_moved_away:
                conf = min(100.0, 50.0 + (sum_delta / (near_vol + 1e-6)) * 50.0)
                return AbsorptionResult(
                    is_absorbed=True,
                    absorption_type="BUYERS_ABSORBED",
                    level=key_level,
                    absorbed_volume=round(near_vol, 2),
                    confidence=round(conf, 1),
                    reason=f"Absorption acheteuse: delta positif ({sum_delta:.1f}) absorbé au sommet {key_level:.5f}, rejet vers {current_p:.5f}",
                )

        return AbsorptionResult(is_absorbed=False, level=key_level)

    # ─────────────────────────────────────────────────────────────
    # 6. CONFLUENCE ORDER FLOW POUR LA STRATÉGIE CRT + ICT/SMC
    # ─────────────────────────────────────────────────────────────
    def evaluate_setup_confluence(
        self,
        symbol: str,
        crt_setup: Any,
        ticks_df: Optional[pd.DataFrame] = None,
        dom_data: Optional[dict] = None,
        candles_df: Optional[pd.DataFrame] = None,
        atr: float = 0.001,
    ) -> OrderFlowConfluence:
        """
        Synthèse multidimensionnelle Order Flow pour le setup CRT.
        Mesure 4 dimensions orthogonales (non redondantes) :
          1. Flux & Absorption (CVD + Absorption au sweep dé-collinéarisées)
          2. Liquidité Passive (Carnet DOM L2 & Murs d'ordres)
          3. Zone de Valeur (Volume Profile POC / VAH / VAL)
          4. Micro-Enchères (Footprint stacked imbalances)

        Intègre la détection de qualité des données (Observed vs Estimated vs Stale)
        et applique un coefficient réducteur (damping) si les données sont estimées.
        """
        if crt_setup is None:
            return OrderFlowConfluence(
                symbol=symbol,
                is_confirmed=False,
                confluence_score=0.0,
                data_quality="UNAVAILABLE",
                summary_badge="OF 0/100 [NON DISPO]",
            )

        direction = getattr(crt_setup, 'direction', 'BUY')
        sweep_price = getattr(crt_setup, 'sweep_price', None)
        entry_price = getattr(crt_setup, 'entry_price', None) or getattr(crt_setup, 'c2_close', 0.0)

        details = []
        cvd_confirmed = False
        absorption_confirmed = False
        dom_confirmed = False
        vp_confirmed = False
        fp_confirmed = False

        # ── A. Inspection de la Qualité et de la Fraîcheur des Données ──
        data_quality = "UNAVAILABLE"
        has_real_vol = False
        is_synthetic = False
        stale_sec = 0.0

        if ticks_df is not None and not ticks_df.empty:
            data_quality = ticks_df.attrs.get("data_quality", "ESTIMATED_QUOTE_RULE")
            has_real_vol = bool(ticks_df.attrs.get("has_real_volume", False))
            is_synthetic = bool(ticks_df.attrs.get("is_synthetic", False))
            stale_sec = float(ticks_df.attrs.get("stale_seconds", 0.0))

            # Auto-détection de secours si attrs absents
            if "data_quality" not in ticks_df.attrs:
                has_real_vol = 'vol' in ticks_df.columns and (ticks_df['vol'] > 1.0).any()
                if 'time' in ticks_df.columns:
                    import datetime
                    now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
                    stale_sec = max(0.0, now_ts - float(ticks_df['time'].iloc[-1]))
                    if stale_sec > 180:
                        data_quality = "DEGRADED_STALE"
                    elif has_real_vol:
                        data_quality = "REAL_OBSERVED"
                    else:
                        data_quality = "ESTIMATED_QUOTE_RULE"
                elif has_real_vol:
                    data_quality = "REAL_OBSERVED"
                else:
                    data_quality = "ESTIMATED_QUOTE_RULE"

        # Damping : modère l'impact des indicateurs de ticks si le flux est estimé (Quote-Rule Forex)
        damping = 1.0
        if data_quality == "DEGRADED_STALE":
            damping = 0.0
            details.append(f"⚠️ Données ticks périmées ({stale_sec:.0f}s sans tick) : confirmation verrouillée")
        elif data_quality == "ESTIMATED_QUOTE_RULE":
            damping = 0.65  # Damping 35% sur ticks estimés pour éviter le sur-ajustement
            details.append("ℹ️ Ticks estimés (Quote Rule / sans volume réel) : pondération modérée 65%")
        elif data_quality == "REAL_OBSERVED":
            damping = 1.0
            details.append("✅ Flux de transactions observé (volumes réels et flags d'agressivité)")

        if is_synthetic:
            details.append("⚠️ Ticks synthétiques de test : simulation sans garantie de rentabilité réelle")

        # ── DIMENSION 1 : Flux & Absorption (CVD + Absorption au Sweep dé-collinéarisés) ──
        # Dé-collinéarisation : une divergence CVD et une absorption au sweep mesurent le même
        # phénomène sous-jacent. On évite le cumul redondant (+25 + +20 = +45) en plafonnant à +30 avec synergie.
        flow_dim = 0.0
        has_cvd_divergence = False
        has_sweep_absorption = False

        cvd_res = self.calculate_cvd(ticks_df) if ticks_df is not None and not ticks_df.empty else None
        if cvd_res:
            if direction == "BUY":
                if cvd_res.divergence == "BULLISH_DIVERGENCE":
                    has_cvd_divergence = True
                    cvd_confirmed = True
                    details.append("CVD: Divergence haussière (absorption vendeuse)")
                elif cvd_res.cvd_trend == "BULLISH":
                    flow_dim += 12.0
                    cvd_confirmed = True
                    details.append(f"CVD: Flux haussier aligné (Delta: {cvd_res.delta_ratio:+.2f})")
                elif cvd_res.divergence == "BEARISH_DIVERGENCE":
                    flow_dim -= 20.0
                    details.append("CVD Alerte: Divergence baissière adverse")
            elif direction == "SELL":
                if cvd_res.divergence == "BEARISH_DIVERGENCE":
                    has_cvd_divergence = True
                    cvd_confirmed = True
                    details.append("CVD: Divergence baissière (absorption acheteuse)")
                elif cvd_res.cvd_trend == "BEARISH":
                    flow_dim += 12.0
                    cvd_confirmed = True
                    details.append(f"CVD: Flux vendeur aligné (Delta: {cvd_res.delta_ratio:+.2f})")
                elif cvd_res.divergence == "BULLISH_DIVERGENCE":
                    flow_dim -= 20.0
                    details.append("CVD Alerte: Divergence haussière adverse")

        if ticks_df is not None and not ticks_df.empty and sweep_price is not None:
            abs_res = self.detect_absorption(ticks_df, sweep_price, direction, tolerance=atr * 0.25)
            if abs_res.is_absorbed:
                has_sweep_absorption = True
                absorption_confirmed = True
                details.append(f"Absorption confirmée au sweep ({abs_res.absorption_type})")

        # Fusion dé-collinéarisée CVD + Absorption au Sweep
        if has_cvd_divergence and has_sweep_absorption:
            flow_dim += 30.0  # Plafond combiné (25 de base + 5 synergie, au lieu de 45 cumulés)
            details.append("Synergie Flow: Divergence CVD + Absorption confirmées ensemble (score combiné)")
        elif has_cvd_divergence:
            flow_dim += 25.0
        elif has_sweep_absorption:
            flow_dim += 20.0

        flow_dim = float(np.clip(flow_dim, -20.0, 30.0))

        # ── DIMENSION 2 : Liquidité Passive & Carnet DOM L2 ──
        dom_dim = 0.0
        dom_res = self.analyze_dom(dom_data) if dom_data else None
        is_dom_available = bool(dom_res and dom_res.available)

        if is_dom_available:
            if direction == "BUY":
                if dom_res.imbalance_ratio > 0.15:
                    dom_dim += 12.0
                    dom_confirmed = True
                    details.append(f"DOM L2: Pression acheteuse (+{dom_res.imbalance_ratio*100:.0f}% bids)")
                elif dom_res.imbalance_ratio < -0.20:
                    dom_dim -= 10.0
                    details.append(f"DOM Alerte: Pression vendeuse adverse ({dom_res.imbalance_ratio*100:.0f}% asks)")
                if dom_res.nearest_bid_wall and dom_res.nearest_bid_wall <= entry_price:
                    dom_dim += 8.0
                    dom_confirmed = True
                    details.append(f"DOM L2: Mur d'achat protecteur @ {dom_res.nearest_bid_wall:.5f}")
            elif direction == "SELL":
                if dom_res.imbalance_ratio < -0.15:
                    dom_dim += 12.0
                    dom_confirmed = True
                    details.append(f"DOM L2: Pression vendeuse ({dom_res.imbalance_ratio*100:.0f}% asks)")
                elif dom_res.imbalance_ratio > 0.20:
                    dom_dim -= 10.0
                    details.append(f"DOM Alerte: Pression acheteuse adverse (+{dom_res.imbalance_ratio*100:.0f}% bids)")
                if dom_res.nearest_ask_wall and dom_res.nearest_ask_wall >= entry_price:
                    dom_dim += 8.0
                    dom_confirmed = True
                    details.append(f"DOM L2: Mur de vente protecteur @ {dom_res.nearest_ask_wall:.5f}")
            dom_dim = float(np.clip(dom_dim, -10.0, 20.0))
        else:
            details.append("DOM L2: Non disponible sur ce symbole/courtier (dimension neutre)")

        # ── DIMENSION 3 : Zone de Valeur Structurelle (Volume Profile) ──
        vp_dim = 0.0
        source_vp = ticks_df if ticks_df is not None and not ticks_df.empty else candles_df
        vp_res = self.calculate_volume_profile(source_vp) if source_vp is not None else None
        poc = vp_res.poc if vp_res else None
        vah = vp_res.vah if vp_res else None
        val = vp_res.val if vp_res else None

        if vp_res and vp_res.poc > 0:
            if direction == "BUY":
                if entry_price <= vp_res.val or abs(entry_price - vp_res.poc) <= atr * 0.3:
                    vp_dim += 15.0
                    vp_confirmed = True
                    details.append(f"VP: Entrée sous VAL ({vp_res.val:.5f}) ou sur POC ({vp_res.poc:.5f})")
            elif direction == "SELL":
                if entry_price >= vp_res.vah or abs(entry_price - vp_res.poc) <= atr * 0.3:
                    vp_dim += 15.0
                    vp_confirmed = True
                    details.append(f"VP: Entrée sur VAH ({vp_res.vah:.5f}) ou sous POC ({vp_res.poc:.5f})")
            vp_dim = float(np.clip(vp_dim, 0.0, 15.0))

        # ── DIMENSION 4 : Micro-Enchères (Footprint Stacked Imbalances) ──
        fp_dim = 0.0
        if ticks_df is not None and len(ticks_df) >= 30:
            fp_res = self.calculate_footprint(ticks_df)
            if direction == "BUY" and fp_res.dominant_side == "BUYERS":
                fp_dim += 10.0
                fp_confirmed = True
                details.append(f"Footprint: Déséquilibres acheteurs ({fp_res.buying_imbalances_count} stacked)")
            elif direction == "SELL" and fp_res.dominant_side == "SELLERS":
                fp_dim += 10.0
                fp_confirmed = True
                details.append(f"Footprint: Déséquilibres vendeurs ({fp_res.selling_imbalances_count} stacked)")
            fp_dim = float(np.clip(fp_dim, 0.0, 10.0))

        # Application du damping sur les dimensions issues du flux de ticks
        damped_flow = flow_dim * damping
        damped_fp = fp_dim * damping

        raw_score = 50.0 + damped_flow + dom_dim + vp_dim + damped_fp
        final_score = float(np.clip(raw_score, 0.0, 100.0))

        # Sécurité : Si les données sont périmées ou indisponibles, interdire la confirmation positive
        if data_quality in ("DEGRADED_STALE", "UNAVAILABLE"):
            final_score = min(final_score, 50.0)
            is_confirmed = False
        else:
            is_confirmed = final_score >= self.min_confluence_score

        badge = f"OF {final_score:.0f}/100"
        if is_confirmed:
            badge += " [CONFIRMÉ]"
        elif data_quality == "DEGRADED_STALE":
            badge += " [PÉRIMÉ]"
        elif data_quality == "ESTIMATED_QUOTE_RULE":
            badge += " [ESTIMÉ]"

        breakdown = {
            "flow_dim": round(damped_flow, 1),
            "dom_dim": round(dom_dim, 1),
            "vp_dim": round(vp_dim, 1),
            "fp_dim": round(damped_fp, 1),
        }

        return OrderFlowConfluence(
            symbol=symbol,
            is_confirmed=is_confirmed,
            confluence_score=round(final_score, 1),
            cvd_confirmed=cvd_confirmed,
            absorption_confirmed=absorption_confirmed,
            dom_confirmed=dom_confirmed,
            volume_profile_confirmed=vp_confirmed,
            footprint_confirmed=fp_confirmed,
            poc=poc,
            vah=vah,
            val=val,
            details=details,
            summary_badge=badge,
            data_quality=data_quality,
            is_dom_available=is_dom_available,
            is_real_volume=has_real_vol,
            is_synthetic=is_synthetic,
            damping_applied=damping,
            dimensions_breakdown=breakdown,
        )
