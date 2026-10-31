# -*- coding: utf-8 -*-
"""
FOCON — Audit Rigoureux du Currency Flow et du Decay Factor des Order Blocks
===========================================================================
Objectif :
1. Auditer les calculs et expliquer rigoureusement les exclusions de trades (notamment GBPNZD vs AUDCAD/NZDCAD/AUDNZD)
2. Exporter la totalité des 55 trades Bullish OB en CSV détaillé (traçabilité par ticket)
3. Approfondir le Decay Factor par tranches (0-10, 11-25, 26-40, >40 bougies) + balayage de sensibilité
4. Modéliser le Currency Strength Alignment (CSA) dynamique sans liste noire permanente
5. Définir l'architecture d'intégration entre les six cerveaux
6. Respect absolu : zéro modification des paramètres de production
"""

import json
import csv
import re
import math
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def parse_all_bullish_ob_trades() -> List[Dict[str, Any]]:
    """Parse et structure l'ensemble des 55 trades Bullish Order Block."""
    path = os.path.join("data", "trade_memory.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    closed = [t for t in data.get("trades", []) if t.get("status") == "CLOSED"]
    bullish = []

    for t in closed:
        snap = t.get("snapshot") or {}
        reason = str(snap.get("reason", ""))
        action = str(t.get("action", "")).upper()
        if "BULLISH OB" in reason.upper() or (action == "BUY" and "ORDER BLOCK" in reason.upper()):
            pnl = float(t.get("final_profit_usd", 0.0) or 0.0)
            mfe = float(t.get("max_profit_usd", 0.0) or 0.0)
            mae = float(t.get("max_drawdown_usd", 0.0) or 0.0)
            entry = float(t.get("entry_price", 0.0) or 0.0)
            sl = float(t.get("sl", 0.0) or 0.0)
            tp = float(t.get("tp", 0.0) or 0.0)
            spread = float(snap.get("spread", 0.0) or 0.0)
            confidence = float(snap.get("confidence", 0.0) or 0.0)
            symbol = str(t.get("symbol", "UNKNOWN"))

            # Âge en bougies
            age_m = re.search(r"(\d+)\s+bougies", reason, re.I)
            ob_age = int(age_m.group(1)) if age_m else 0

            # Impulsion
            imp_m = re.search(r"Impulsion\s+([\d\.]+)\s*x\s*ATR", reason, re.I)
            impulse_atr = float(imp_m.group(1)) if imp_m else 1.5

            # Score
            score_m = re.search(r"Score\s+(\d+)/100", reason, re.I)
            ob_score = float(score_m.group(1)) if score_m else confidence

            # 1R Risk
            exit_reason = str(t.get("exit_reason", "")).upper()
            if exit_reason == "SL_HIT" and pnl < 0:
                r_risk = max(abs(pnl), 0.50)
            elif abs(mae) > 0:
                r_risk = max(abs(mae), 0.50)
            else:
                r_risk = max(abs(entry - sl) / entry * 1000.0 if entry > 0 else 1.5, 1.0)

            result_r = round(pnl / r_risk, 3)

            # Catégorisation de l'âge
            if ob_age <= 10:
                age_cat = "0-10 (Ultra-frais)"
            elif ob_age <= 25:
                age_cat = "11-25 (Frais standard)"
            elif ob_age <= 40:
                age_cat = "26-40 (Vieillissant)"
            else:
                age_cat = ">40 (Ancien / Dégradé)"

            bullish.append({
                "ticket": t.get("ticket"),
                "symbol": symbol,
                "action": action,
                "open_time": t.get("open_time", 0.0),
                "open_time_str": t.get("open_time_str", ""),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "pnl_usd": pnl,
                "result_r": result_r,
                "r_risk": r_risk,
                "mfe_usd": mfe,
                "mae_usd": mae,
                "ob_age": ob_age,
                "age_cat": age_cat,
                "impulse_atr": impulse_atr,
                "ob_score": ob_score,
                "spread": spread,
                "exit_reason": exit_reason,
                "is_win": pnl > 0,
                "is_loss": pnl < 0,
            })

    # Tri chronologique absolu
    return sorted(bullish, key=lambda x: x["open_time"])


def audit_variant_1_exclusions(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Audit approfondi de la Variante 1 :
    Explique pourquoi 19 trades ont été exclus (AUDCAD, NZDCAD, AUDNZD)
    et pourquoi GBPNZD (2 trades perdants) n'était pas inclus dans l'exclusion.
    """
    symbols_breakdown = defaultdict(lambda: {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0})
    for t in trades:
        s = t["symbol"]
        symbols_breakdown[s]["total"] += 1
        symbols_breakdown[s]["pnl"] += t["pnl_usd"]
        if t["is_win"]:
            symbols_breakdown[s]["wins"] += 1
        elif t["is_loss"]:
            symbols_breakdown[s]["losses"] += 1

    # Règle de la Variante 1 : exclusion stricte du bloc matières premières océaniennes face au CAD
    excluded_symbols_v1 = {"AUDCAD", "NZDCAD", "AUDNZD"}
    excluded_trades_v1 = [t for t in trades if t["symbol"] in excluded_symbols_v1]
    kept_trades_v1 = [t for t in trades if t["symbol"] not in excluded_symbols_v1]

    # Vérification GBPNZD
    gbpnzd_trades = [t for t in trades if t["symbol"] == "GBPNZD"]

    return {
        "symbols_breakdown": dict(symbols_breakdown),
        "excluded_symbols_v1": list(excluded_symbols_v1),
        "excluded_count": len(excluded_trades_v1),
        "excluded_wins": sum(1 for t in excluded_trades_v1 if t["is_win"]),
        "excluded_losses": sum(1 for t in excluded_trades_v1 if t["is_loss"]),
        "excluded_pnl": sum(t["pnl_usd"] for t in excluded_trades_v1),
        "kept_count": len(kept_trades_v1),
        "kept_wins": sum(1 for t in kept_trades_v1 if t["is_win"]),
        "kept_losses": sum(1 for t in kept_trades_v1 if t["is_loss"]),
        "kept_pnl": sum(t["pnl_usd"] for t in kept_trades_v1),
        "gbpnzd_count": len(gbpnzd_trades),
        "gbpnzd_losses": sum(1 for t in gbpnzd_trades if t["is_loss"]),
        "gbpnzd_pnl": sum(t["pnl_usd"] for t in gbpnzd_trades),
    }


def analyze_decay_brackets(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse par tranche d'âge de l'Order Block : 0-10, 11-25, 26-40, >40 bougies."""
    brackets = {
        "0-10 (Ultra-frais)": [t for t in trades if t["ob_age"] <= 10],
        "11-25 (Frais standard)": [t for t in trades if 11 <= t["ob_age"] <= 25],
        "26-40 (Vieillissant)": [t for t in trades if 26 <= t["ob_age"] <= 40],
        ">40 (Ancien / Dégradé)": [t for t in trades if t["ob_age"] > 40],
    }

    report = {}
    for b_name, b_trades in brackets.items():
        n = len(b_trades)
        if n == 0:
            report[b_name] = {"count": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0, "pnl": 0.0}
            continue
        wins = [t for t in b_trades if t["is_win"]]
        losses = [t for t in b_trades if t["is_loss"]]
        gross_w = sum(t["pnl_usd"] for t in wins)
        gross_l = abs(sum(t["pnl_usd"] for t in losses))
        pf = round(gross_w / max(gross_l, 1e-5), 2) if gross_l > 0 else 99.0
        wr = round(len(wins) / n * 100.0, 1)
        exp_r = round(sum(t["result_r"] for t in b_trades) / n, 3)
        net_pnl = round(sum(t["pnl_usd"] for t in b_trades), 2)
        avg_mae = round(sum(t["mae_usd"] for t in b_trades) / n, 2)
        avg_mfe = round(sum(t["mfe_usd"] for t in b_trades) / n, 2)

        report[b_name] = {
            "count": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": wr,
            "profit_factor": pf,
            "expectancy_r": exp_r,
            "net_pnl": net_pnl,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
        }

    return report


def sweep_age_cutoffs(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Balayage de sensibilité du seuil d'âge de 10 à 45 bougies par pas de 5."""
    results = []
    for cutoff in range(10, 50, 5):
        filtered = [t for t in trades if t["ob_age"] <= cutoff]
        n = len(filtered)
        if n == 0:
            continue
        wins = [t for t in filtered if t["is_win"]]
        losses = [t for t in filtered if t["is_loss"]]
        gross_w = sum(t["pnl_usd"] for t in wins)
        gross_l = abs(sum(t["pnl_usd"] for t in losses))
        pf = round(gross_w / max(gross_l, 1e-5), 2) if gross_l > 0 else 99.0
        wr = round(len(wins) / n * 100.0, 1)
        exp_r = round(sum(t["result_r"] for t in filtered) / n, 3)
        pnl = round(sum(t["pnl_usd"] for t in filtered), 2)

        results.append({
            "cutoff_age": cutoff,
            "retained_trades": n,
            "win_rate": wr,
            "profit_factor": pf,
            "expectancy_r": exp_r,
            "net_pnl": pnl,
        })
    return results


def export_trades_csv(trades: List[Dict[str, Any]], filename: str = "data/audit_trades_currency_flow_decay.csv"):
    """Exporte la totalité des 55 trades analysés avec traçabilité complète de chaque ticket."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fieldnames = [
        "ticket", "symbol", "action", "open_time_str", "entry", "sl", "tp",
        "pnl_usd", "result_r", "r_risk", "mfe_usd", "mae_usd", "ob_age",
        "age_category", "impulse_atr", "ob_score", "spread", "exit_reason",
        "v1_excluded_knife", "v2_retained_fresh25", "v3_retained_combined"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            is_v1_excluded = t["symbol"] in ("AUDCAD", "NZDCAD", "AUDNZD")
            is_v2_retained = t["ob_age"] <= 25
            is_v3_retained = (not is_v1_excluded) and is_v2_retained

            writer.writerow({
                "ticket": t["ticket"],
                "symbol": t["symbol"],
                "action": t["action"],
                "open_time_str": t["open_time_str"],
                "entry": t["entry"],
                "sl": t["sl"],
                "tp": t["tp"],
                "pnl_usd": t["pnl_usd"],
                "result_r": t["result_r"],
                "r_risk": round(t["r_risk"], 2),
                "mfe_usd": t["mfe_usd"],
                "mae_usd": t["mae_usd"],
                "ob_age": t["ob_age"],
                "age_category": t["age_cat"],
                "impulse_atr": t["impulse_atr"],
                "ob_score": t["ob_score"],
                "spread": t["spread"],
                "exit_reason": t["exit_reason"],
                "v1_excluded_knife": "OUI" if is_v1_excluded else "NON",
                "v2_retained_fresh25": "OUI" if is_v2_retained else "NON",
                "v3_retained_combined": "OUI" if is_v3_retained else "NON",
            })
    print(f"Export CSV généré avec succès : {filename} ({len(trades)} lignes).")


def run_audit():
    trades = parse_all_bullish_ob_trades()
    print(f"Total trades Bullish OB chargés : {len(trades)}")

    # 1. Audit des exclusions
    v1_audit = audit_variant_1_exclusions(trades)
    print("\n" + "=" * 75)
    print("1. AUDIT DES EXCLUSIONS VARIANTE 1 (EXCLUSION DU BLOC AUD/NZD/CAD)")
    print("=" * 75)
    print(f"Trades totaux : {len(trades)}")
    print(f"Symboles exclus : {v1_audit['excluded_symbols_v1']}")
    print(f"Trades exclus : {v1_audit['excluded_count']} (Gagnants : {v1_audit['excluded_wins']}, Pertes : {v1_audit['excluded_losses']})")
    print(f"PnL des trades exclus : {v1_audit['excluded_pnl']:+.2f} $")
    print(f"Trades conservés : {v1_audit['kept_count']} (Gagnants : {v1_audit['kept_wins']}, Pertes : {v1_audit['kept_losses']})")
    print(f"PnL des trades conservés : {v1_audit['kept_pnl']:+.2f} $")
    print("-" * 75)
    print("Cas particulier de GBPNZD :")
    print(f"  GBPNZD a enregistré {v1_audit['gbpnzd_count']} trades BUY, tous les deux perdants ({v1_audit['gbpnzd_pnl']:+.2f} $).")
    print("  Explication : La Variante 1 excluait uniquement le sous-panier 'AUDCAD', 'NZDCAD', 'AUDNZD'.")
    print("  GBPNZD n'en faisait pas partie et a donc été CONSERVÉ dans la Variante 1.")

    # 2. Analyse par tranche d'âge
    brackets = analyze_decay_brackets(trades)
    print("\n" + "=" * 75)
    print("2. ANALYSE DU DECAY FACTOR PAR TRANCHES D'ÂGE D'ORDER BLOCK")
    print("=" * 75)
    for cat, b in brackets.items():
        print(f"Tranche : {cat:<24} | n={b['count']:2d} | WR: {b['win_rate']:>5.1f}% | PF: {b['profit_factor']:>5.2f} | Exp: {b['expectancy_r']:>+6.3f}R | Net: {b['net_pnl']:>+6.2f}$ | MAE: {b['avg_mae']:>5.2f}$ | MFE: {b['avg_mfe']:>5.2f}$")

    # 3. Balayage de sensibilité
    print("\n" + "=" * 75)
    print("3. BALAYAGE DE SENSIBILITÉ DU SEUIL D'ÂGE MAXIMUM")
    print("=" * 75)
    sweeps = sweep_age_cutoffs(trades)
    for s in sweeps:
        print(f"Âge max <= {s['cutoff_age']:2d} bougies | Retenus : {s['retained_trades']:2d}/55 | WR: {s['win_rate']:>5.1f}% | PF: {s['profit_factor']:>5.2f} | Exp: {s['expectancy_r']:>+6.3f}R | Net: {s['net_pnl']:>+6.2f}$")

    # 4. Export CSV
    export_trades_csv(trades)


if __name__ == "__main__":
    run_audit()
