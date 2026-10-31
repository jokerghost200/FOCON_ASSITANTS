# -*- coding: utf-8 -*-
"""
FOCON — Analyse Scientifique des Facteurs de Performance SMC Order Block
========================================================================
Étudie les 67 Order Blocks Bearish et 55 Order Blocks Bullish pour identifier :
1. Pourquoi les Bearish OB surperforment (+0.520 R vs +0.108 R, PF 3.38 vs 2.01)
2. Pourquoi les Bullish OB subissent un MAE plus lourd (-2.74 $ vs -1.70 $)
3. Les corrélations entre variables (HTF, Âge OB, Impulsion, Spread, Session) et succès
4. Formulation et simulation d'expériences contrôlées en Sandbox In-Sample / Out-of-Sample
5. Aucune modification en production (sécurité El Centinela)
"""

import json
import re
import math
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def parse_trade_features(t: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait l'ensemble des variables contextuelles et quantitatives d'un trade."""
    snap = t.get("snapshot") or {}
    reason = str(snap.get("reason", ""))
    action = str(t.get("action", "")).upper()
    symbol = str(t.get("symbol", "UNKNOWN"))
    entry = float(t.get("entry_price", 0.0) or 0.0)
    sl = float(t.get("sl", 0.0) or 0.0)
    tp = float(t.get("tp", 0.0) or 0.0)
    pnl = float(t.get("final_profit_usd", 0.0) or 0.0)
    mfe = float(t.get("max_profit_usd", 0.0) or 0.0)
    mae = float(t.get("max_drawdown_usd", 0.0) or 0.0)
    spread = float(snap.get("spread", 0.0) or 0.0)
    confidence = float(snap.get("confidence", 0.0) or 0.0)
    htf = str(snap.get("htf_trend", "NEUTRAL")).upper()

    # Détection sous-famille OB
    ob_type = "UNKNOWN"
    if "BULLISH OB" in reason.upper() or (action == "BUY" and "ORDER BLOCK" in reason.upper()):
        ob_type = "BULLISH"
    elif "BEARISH OB" in reason.upper() or (action == "SELL" and "ORDER BLOCK" in reason.upper()):
        ob_type = "BEARISH"

    # Extraction Score
    score_match = re.search(r"Score\s+(\d+)/100", reason, re.I)
    ob_score = float(score_match.group(1)) if score_match else confidence

    # Extraction Âge en bougies
    age_match = re.search(r"(\d+)\s+bougies", reason, re.I)
    ob_age = int(age_match.group(1)) if age_match else 0

    # Extraction Impulsion ATR
    imp_match = re.search(r"Impulsion\s+([\d\.]+)\s*x\s*ATR", reason, re.I)
    impulse_atr = float(imp_match.group(1)) if imp_match else 1.5

    # Extraction R:R
    rr_match = re.search(r"R:R\s+([\d\.]+)", reason, re.I)
    rr_ratio = float(rr_match.group(1)) if rr_match else 2.0

    # Alignement HTF
    if action == "BUY":
        aligned = (htf in ("BULLISH", "STRONG BULLISH"))
        counter = (htf in ("BEARISH", "STRONG BEARISH"))
    else:
        aligned = (htf in ("BEARISH", "STRONG BEARISH"))
        counter = (htf in ("BULLISH", "STRONG BULLISH"))
    neutral = (htf == "NEUTRAL")

    # Session horaire (UTC)
    open_str = t.get("open_time_str", "")
    session = "OTHER"
    if open_str:
        try:
            dt = datetime.strptime(open_str, "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
            if 0 <= hour < 7:
                session = "ASIAN"
            elif 7 <= hour < 12:
                session = "LONDON"
            elif 12 <= hour < 16:
                session = "NY_OVERLAP"
            elif 16 <= hour < 21:
                session = "NY_AFTERNOON"
            else:
                session = "ROLLOVER"
        except Exception:
            pass

    # Risque 1R
    exit_reason = str(t.get("exit_reason", "")).upper()
    if exit_reason == "SL_HIT" and pnl < 0:
        r_risk = max(abs(pnl), 0.50)
    elif abs(mae) > 0:
        r_risk = max(abs(mae), 0.50)
    else:
        r_risk = max(abs(entry - sl) / entry * 1000.0 if entry > 0 else 1.5, 1.0)

    result_r = round(pnl / r_risk, 3)
    mfe_r = round(mfe / r_risk, 3)
    mae_r = round(mae / r_risk, 3)

    return {
        "ticket": t.get("ticket"),
        "timestamp": t.get("open_time", 0.0),
        "open_time_str": open_str,
        "symbol": symbol,
        "action": action,
        "ob_type": ob_type,
        "htf_trend": htf,
        "aligned_htf": aligned,
        "counter_htf": counter,
        "neutral_htf": neutral,
        "ob_score": ob_score,
        "ob_age": ob_age,
        "impulse_atr": impulse_atr,
        "rr_ratio": rr_ratio,
        "spread": spread,
        "confidence": confidence,
        "session": session,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sl_dist": abs(entry - sl),
        "pnl_usd": pnl,
        "result_r": result_r,
        "r_risk": r_risk,
        "mfe_usd": mfe,
        "mae_usd": mae,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "exit_reason": exit_reason,
        "is_win": pnl > 0,
        "is_loss": pnl < 0,
    }


def compute_group_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule les statistiques moyennes et distributions d'un groupe."""
    if not trades:
        return {}
    n = len(trades)
    wins = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if t["is_loss"]]
    gross_win = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = round(gross_win / max(gross_loss, 1e-5), 2) if gross_loss > 0 else 99.0
    wr = round(len(wins) / n * 100.0, 1)
    exp_r = round(sum(t["result_r"] for t in trades) / n, 3)
    net_pnl = round(gross_win - gross_loss, 2)

    avg_mfe = round(sum(t["mfe_usd"] for t in trades) / n, 2)
    avg_mae = round(sum(t["mae_usd"] for t in trades) / n, 2)
    avg_score = round(sum(t["ob_score"] for t in trades) / n, 1)
    avg_age = round(sum(t["ob_age"] for t in trades) / n, 1)
    avg_imp = round(sum(t["impulse_atr"] for t in trades) / n, 2)
    avg_spread = round(sum(t["spread"] for t in trades) / n, 2)

    aligned_pct = round(sum(1 for t in trades if t["aligned_htf"]) / n * 100.0, 1)
    neutral_pct = round(sum(1 for t in trades if t["neutral_htf"]) / n * 100.0, 1)
    counter_pct = round(sum(1 for t in trades if t["counter_htf"]) / n * 100.0, 1)

    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "profit_factor": pf,
        "expectancy_r": exp_r,
        "net_pnl": net_pnl,
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0.0,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_score": avg_score,
        "avg_age": avg_age,
        "avg_impulse": avg_imp,
        "avg_spread": avg_spread,
        "aligned_pct": aligned_pct,
        "neutral_pct": neutral_pct,
        "counter_pct": counter_pct,
    }


def analyze_differences(bearish_trades: List[Dict[str, Any]], bullish_trades: List[Dict[str, Any]]):
    """Analyse comparative détaillée entre OB Bearish et OB Bullish."""
    b_stats = compute_group_stats(bearish_trades)
    u_stats = compute_group_stats(bullish_trades)

    print("=" * 80)
    print("1. COMPARAISON GLOBALE : OB BEARISH (67) vs OB BULLISH (55)")
    print("=" * 80)
    print(f"{'Métrique':<25} | {'OB Bearish (67)':<22} | {'OB Bullish (55)':<22} | {'Écart / Diagnostic':<20}")
    print("-" * 80)
    print(f"{'Win Rate':<25} | {b_stats['win_rate']:>6.1f} %                | {u_stats['win_rate']:>6.1f} %                | {u_stats['win_rate'] - b_stats['win_rate']:+6.1f} pts (Bearish >)")
    print(f"{'Profit Factor':<25} | {b_stats['profit_factor']:>6.2f}                  | {u_stats['profit_factor']:>6.2f}                  | {u_stats['profit_factor'] - b_stats['profit_factor']:+6.2f} (Bearish >)")
    print(f"{'Expectancy (R)':<25} | {b_stats['expectancy_r']:>+6.3f} R               | {u_stats['expectancy_r']:>+6.3f} R               | {u_stats['expectancy_r'] - b_stats['expectancy_r']:+6.3f} R")
    print(f"{'Net PnL':<25} | {b_stats['net_pnl']:>+7.2f} $              | {u_stats['net_pnl']:>+7.2f} $              | {u_stats['net_pnl'] - b_stats['net_pnl']:+7.2f} $")
    print(f"{'MAE moyen (Drawdown)':<25} | {b_stats['avg_mae']:>6.2f} $               | {u_stats['avg_mae']:>6.2f} $               | {u_stats['avg_mae'] - b_stats['avg_mae']:+6.2f} $ (Bullish pire)")
    print(f"{'MFE moyen (Max gain)':<25} | {b_stats['avg_mfe']:>6.2f} $               | {u_stats['avg_mfe']:>6.2f} $               | {u_stats['avg_mfe'] - b_stats['avg_mfe']:+6.2f} $")
    print(f"{'Aligné HTF (%)':<25} | {b_stats['aligned_pct']:>6.1f} %                | {u_stats['aligned_pct']:>6.1f} %                | {u_stats['aligned_pct'] - b_stats['aligned_pct']:+6.1f} %")
    print(f"{'Neutre HTF (%)':<25} | {b_stats['neutral_pct']:>6.1f} %                | {u_stats['neutral_pct']:>6.1f} %                | {u_stats['neutral_pct'] - b_stats['neutral_pct']:+6.1f} %")
    print(f"{'Contre-tendance HTF':<25} | {b_stats['counter_pct']:>6.1f} %                | {u_stats['counter_pct']:>6.1f} %                | {u_stats['counter_pct'] - b_stats['counter_pct']:+6.1f} %")
    print(f"{'Âge moyen (bougies)':<25} | {b_stats['avg_age']:>6.1f} bougies          | {u_stats['avg_age']:>6.1f} bougies          | {u_stats['avg_age'] - b_stats['avg_age']:+6.1f} bougies")
    print(f"{'Impulsion moyenne':<25} | {b_stats['avg_impulse']:>6.2f} x ATR            | {u_stats['avg_impulse']:>6.2f} x ATR            | {u_stats['avg_impulse'] - b_stats['avg_impulse']:+6.2f} ATR")
    print(f"{'Spread moyen':<25} | {b_stats['avg_spread']:>6.2f} pts              | {u_stats['avg_spread']:>6.2f} pts              | {u_stats['avg_spread'] - b_stats['avg_spread']:+6.2f} pts")


def analyze_bullish_win_vs_loss(bullish_trades: List[Dict[str, Any]]):
    """Compare les Bullish gagnants vs perdants pour isoler la cause des pertes."""
    wins = [t for t in bullish_trades if t["is_win"]]
    losses = [t for t in bullish_trades if t["is_loss"]]

    w_stats = compute_group_stats(wins)
    l_stats = compute_group_stats(losses)

    print("\n" + "=" * 80)
    print("2. DISSECTION DES OB BULLISH : GAGNANTS (18) vs PERDANTS (37)")
    print("=" * 80)
    print(f"{'Facteur':<25} | {'Bullish GAGNANTS (18)':<22} | {'Bullish PERDANTS (37)':<22} | {'Différence clé':<20}")
    print("-" * 80)
    print(f"{'Alignement HTF (%)':<25} | {w_stats['aligned_pct']:>6.1f} %                | {l_stats['aligned_pct']:>6.1f} %                | {w_stats['aligned_pct'] - l_stats['aligned_pct']:+6.1f} % (Crucial !)")
    print(f"{'Neutre HTF (%)':<25} | {w_stats['neutral_pct']:>6.1f} %                | {l_stats['neutral_pct']:>6.1f} %                | {w_stats['neutral_pct'] - l_stats['neutral_pct']:+6.1f} %")
    print(f"{'Âge moyen (bougies)':<25} | {w_stats['avg_age']:>6.1f} bougies          | {l_stats['avg_age']:>6.1f} bougies          | {l_stats['avg_age'] - w_stats['avg_age']:+6.1f} (Perdants plus vieux)")
    print(f"{'Impulsion ATR':<25} | {w_stats['avg_impulse']:>6.2f} x ATR            | {l_stats['avg_impulse']:>6.2f} x ATR            | {w_stats['avg_impulse'] - l_stats['avg_impulse']:+6.2f} ATR (Gagnants plus forts)")
    print(f"{'Spread moyen':<25} | {w_stats['avg_spread']:>6.2f} pts              | {l_stats['avg_spread']:>6.2f} pts              | {l_stats['avg_spread'] - w_stats['avg_spread']:+6.2f} pts")
    print(f"{'MAE moyen':<25} | {w_stats['avg_mae']:>6.2f} $               | {l_stats['avg_mae']:>6.2f} $               | {l_stats['avg_mae'] - w_stats['avg_mae']:+6.2f} $")
    print(f"{'MFE moyen':<25} | {w_stats['avg_mfe']:>6.2f} $               | {l_stats['avg_mfe']:>6.2f} $               | {w_stats['avg_mfe'] - l_stats['avg_mfe']:+6.2f} $")


def test_controlled_experiments(bullish_trades: List[Dict[str, Any]]):
    """
    Simule 4 variantes expérimentales de filtres sur les 55 OB Bullish
    avec séparation stricte In-Sample (première moitié) / Out-of-Sample (seconde moitié).
    """
    n = len(bullish_trades)
    split_idx = n // 2
    is_trades = bullish_trades[:split_idx]
    oos_trades = bullish_trades[split_idx:]

    print("\n" + "=" * 80)
    print("3. EXPÉRIENCES CONTRÔLÉES EN SANDBOX SUR LES OB BULLISH (IN-SAMPLE vs OUT-OF-SAMPLE)")
    print("=" * 80)

    variants = [
        ("Base (Aucun filtre)", lambda t: True),
        ("Variante 1 : Filtre Anti-HTF Neutre (Exclut HTF NEUTRAL)", lambda t: not t["neutral_htf"]),
        ("Variante 2 : Filtre Fraîcheur (Âge OB <= 45 bougies)", lambda t: t["ob_age"] <= 45),
        ("Variante 3 : Filtre Impulsion Forte (Impulsion >= 2.0x ATR)", lambda t: t["impulse_atr"] >= 2.0),
        ("Variante 4 : Confluence Maîtresse (HTF non-neutre + Âge <= 50)", lambda t: not t["neutral_htf"] and t["ob_age"] <= 50),
    ]

    for name, flt in variants:
        is_sub = [t for t in is_trades if flt(t)]
        oos_sub = [t for t in oos_trades if flt(t)]
        all_sub = [t for t in bullish_trades if flt(t)]

        m_is = compute_group_stats(is_sub)
        m_oos = compute_group_stats(oos_sub)
        m_all = compute_group_stats(all_sub)

        exp_is = m_is.get("expectancy_r", 0.0)
        exp_oos = m_oos.get("expectancy_r", 0.0)
        wfe = round(exp_oos / exp_is, 2) if exp_is > 0 else 0.0

        print(f"[{name}]")
        print(f"  GLOBAL  ({m_all.get('n', 0)} trades) : WR {m_all.get('win_rate', 0)}% | PF {m_all.get('profit_factor', 0)} | Exp {m_all.get('expectancy_r', 0):+.3f}R | PnL {m_all.get('net_pnl', 0):+.2f}$ | MAE {m_all.get('avg_mae', 0):.2f}$")
        print(f"  IN-SAMPLE  ({m_is.get('n', 0)} trades) : WR {m_is.get('win_rate', 0)}% | PF {m_is.get('profit_factor', 0)} | Exp {exp_is:+.3f}R")
        print(f"  OUT-SAMPLE ({m_oos.get('n', 0)} trades) : WR {m_oos.get('win_rate', 0)}% | PF {m_oos.get('profit_factor', 0)} | Exp {exp_oos:+.3f}R | WFE {wfe}")
        print("-" * 80)


def run_full_ob_audit():
    with open("data/trade_memory.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    closed = [t for t in data.get("trades", []) if t.get("status") == "CLOSED"]
    parsed = [parse_trade_features(t) for t in closed]
    # Tri chronologique absolu
    parsed = sorted(parsed, key=lambda t: t["timestamp"])

    bearish_obs = [t for t in parsed if t["ob_type"] == "BEARISH"]
    bullish_obs = [t for t in parsed if t["ob_type"] == "BULLISH"]

    print(f"Dataset chargé : {len(bearish_obs)} OB Bearish et {len(bullish_obs)} OB Bullish.")
    analyze_differences(bearish_obs, bullish_obs)
    analyze_bullish_win_vs_loss(bullish_obs)
    test_controlled_experiments(bullish_obs)


if __name__ == "__main__":
    run_full_ob_audit()
