#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOCON ASSISTANT — Lanceur CLI de Backtesting
============================================
Exécute des simulations virtuelles bougie par bougie sur l'historique réel MT5
avec l'ensemble des stratégies (CRT, SMC Order Blocks, Boomerang, Cycles AMD)
et la gestion de risque intégrée (Dynamique ou Lot Fixe).

Usage :
    python backtester.py
    python backtester.py --symbol BTCUSD --tf M15 --bars 800 --capital 500
    python backtester.py --symbol EURUSD --strategy CRT --fixed-lot 0.02
    python backtester.py --symbol GBPUSD --profile AGGRESSIVE --export trades.csv
"""

import sys
import os
import argparse
import logging
from typing import Optional

# Configuration du chemin pour importer src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.mt5_connector import MT5Connector
from src.trading.risk_manager import RiskManager
from src.trading.backtester import BacktestEngine


def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                      🦅 FOCON ASSISTANT — BACKTEST ENGINE                             ║
║      Simulation Institutionnelle SMC • CRT • Boomerang • Cycles AMD • Risk Mgmt       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def render_progress_bar(progress: float, text: str):
    bar_len = 35
    filled = int(round(bar_len * progress))
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(progress * 100)
    sys.stdout.write(f"\r  [{bar}] {pct:3d}% | {text[:45]:<45}")
    sys.stdout.flush()


def run_cli_backtest(args):
    setup_logger(verbose=args.verbose)
    print_banner()

    symbol = args.symbol.upper()
    timeframe = args.timeframe.upper()
    bars = int(args.bars)
    capital = float(args.capital)
    strategy = args.strategy.upper()
    profile = args.profile.upper()
    fixed_lot = float(args.fixed_lot) if args.fixed_lot is not None else 0.0
    direction = args.direction.upper()
    risk_pct = float(args.risk_pct) if args.risk_pct is not None else None

    print(f"  📊 Symbole        : \033[1m{symbol}\033[0m")
    print(f"  ⏱️  Timeframe      : \033[1m{timeframe}\033[0m")
    print(f"  🕯️  Bougies        : \033[1m{bars}\033[0m")
    print(f"  💰 Capital Initial: \033[1m{capital:.2f} $\033[0m")
    print(f"  🎯 Stratégie      : \033[1m{strategy}\033[0m (CRT, SMC Order Blocks, Boomerang)")
    print(f"  🛡️  Profil Risque  : \033[1m{profile}\033[0m")
    if fixed_lot > 0.0:
        print(f"  📦 Volume Fixe    : \033[1m{fixed_lot:.2f} Lot(s)\033[0m (Mode Forcé)")
    else:
        print(f"  📐 Volume         : \033[1mDynamique (AutoPilot)\033[0m")
    print(f"  🧭 Direction      : \033[1m{direction}\033[0m\n")

    print("  🔌 Connexion à MetaTrader 5...")
    mt5_conn = MT5Connector()
    if not mt5_conn.connect():
        print("\n  ❌ ERREUR : Impossible d'établir la liaison avec MetaTrader 5.")
        print("     Veuillez vérifier que le terminal MT5 est bien ouvert et connecté à un compte.")
        return 1

    engine = BacktestEngine(mt5_connector=mt5_conn)

    print("\n  🚀 Démarrage de la simulation...")
    results = engine.run_backtest(
        symbol=symbol,
        timeframe=timeframe,
        num_bars=bars,
        initial_capital=capital,
        risk_profile=profile,
        risk_percent=risk_pct,
        direction_filter=direction,
        strategy_filter=strategy,
        fixed_lot_size=fixed_lot,
        progress_cb=render_progress_bar,
    )
    print("\n")

    if "error" in results:
        print(f"  ❌ Échec de la simulation : {results['error']}")
        return 1

    # Affichage des métriques clés
    stats = results.get("stats", {})
    tot_trades = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0.0)
    pf = stats.get("profit_factor", 0.0)
    net_pnl = stats.get("net_profit_usd", 0.0)
    net_pnl_pct = stats.get("net_profit_pct", 0.0)
    max_dd_pct = stats.get("max_drawdown_pct", 0.0)
    max_dd_usd = stats.get("max_drawdown_usd", 0.0)
    final_balance = stats.get("final_balance", capital)

    pnl_color = "\033[32m" if net_pnl >= 0 else "\033[31m"
    reset_color = "\033[0m"

    print("╔═══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                              RÉSULTATS DE LA SIMULATION                               ║")
    print("╠═══════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Capital Initial : {capital:>10.2f} $       │  Capital Final   : {final_balance:>10.2f} $           ║")
    print(f"║  Profit Net      : {pnl_color}{net_pnl:>+10.2f} ${reset_color}       │  Rendement Total : {pnl_color}{net_pnl_pct:>+9.2f} %{reset_color}           ║")
    print(f"║  Trades Exécutés : {tot_trades:>10d}         │  Win Rate        : {win_rate:>9.1f} %           ║")
    print(f"║  Profit Factor   : {pf:>10.2f}          │  Max Drawdown    : {max_dd_pct:>9.2f} % ({max_dd_usd:.2f} $)  ║")
    print(f"║  Trades Gagnants : {stats.get('winning_trades', 0):>10d}         │  Trades Perdants : {stats.get('losing_trades', 0):>10d}           ║")
    print(f"║  Trades au BE    : {stats.get('break_even_trades', 0):>10d}         │  Ordres Limits   : {stats.get('pending_orders_placed', 0):>10d} ({stats.get('pending_orders_filled', 0)} remplis) ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════════════╝\n")

    # Affichage de la table des trades
    trades = results.get("trades", [])
    if trades:
        print("  📜 DÉTAIL DES DERNIERS TRADES EXÉCUTÉS :")
        print("  " + "─" * 86)
        print(f"  {'#':<4} {'Date/Heure':<17} {'Strat':<9} {'Type':<5} {'Lot':<5} {'Entrée':<9} {'Sortie':<9} {'PnL ($)':<9} {'Raison':<12}")
        print("  " + "─" * 86)
        display_trades = trades[-25:] if len(trades) > 25 else trades
        for t in display_trades:
            pnl = t.get("profit_usd", 0.0)
            col = "\033[32m" if pnl > 0 else ("\033[31m" if pnl < 0 else "")
            pnl_str = f"{col}{pnl:+7.2f} ${reset_color}" if col else f"{pnl:+7.2f} $"
            print(
                f"  {t.get('ticket', 0):<4} "
                f"{str(t.get('close_time', ''))[:16]:<17} "
                f"{t.get('strategy', 'FUSION')[:8]:<9} "
                f"{t.get('action', ''):<5} "
                f"{t.get('lot', 0.0):<5.2f} "
                f"{t.get('entry', 0.0):<9.4f} "
                f"{t.get('exit_price', 0.0):<9.4f} "
                f"{pnl_str:<18} "
                f"{t.get('exit_reason', ''):<12}"
            )
        if len(trades) > 25:
            print(f"  ... (+ {len(trades) - 25} trades antérieurs masqués)")
        print("  " + "─" * 86 + "\n")

    # Export CSV si demandé
    if args.export:
        import pandas as pd
        if trades:
            export_path = args.export
            df_trades = pd.DataFrame(trades)
            df_trades.to_csv(export_path, index=False)
            print(f"  💾 {len(trades)} trade(s) exporté(s) avec succès dans : {export_path}\n")
        else:
            print("  ℹ️ Aucun trade à exporter pour cette simulation.\n")

    # Rapport d'analyse autonome et diagnostic complet
    report = results.get("report_text", "")
    if report:
        print(report)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FOCON ASSISTANT — Moteur de Backtest Institutionnel MT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python backtester.py
  python backtester.py --symbol BTCUSD --tf M15 --bars 800 --capital 500
  python backtester.py --symbol EURUSD --strategy CRT --fixed-lot 0.02
  python backtester.py --symbol GBPUSD --profile AGGRESSIVE --export trades.csv
        """,
    )
    parser.add_argument("-s", "--symbol", type=str, default="EURUSD", help="Symbole MT5 (ex: EURUSD, BTCUSD, GBPUSD, XAUUSD)")
    parser.add_argument("-t", "--timeframe", "--tf", type=str, default="M15", help="Unité de temps (M5, M15, H1, H4, D1)")
    parser.add_argument("-b", "--bars", type=int, default=600, help="Nombre de bougies historiques à simuler (défaut: 600)")
    parser.add_argument("-c", "--capital", type=float, default=100.0, help="Capital initial de test en $ (défaut: 100.0)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ALL",
        choices=["ALL", "CRT", "ORDER_BLOCKS", "BOOMERANG"],
        help="Stratégie à tester : ALL (Fusion), CRT (Candle Range Theory), ORDER_BLOCKS (SMC), BOOMERANG",
    )
    parser.add_argument(
        "-p", "--profile",
        type=str,
        default="BALANCED",
        choices=["BALANCED", "AGGRESSIVE", "CONSERVATIVE", "DYNAMIC"],
        help="Profil de risque (BALANCED, AGGRESSIVE, CONSERVATIVE, DYNAMIC)",
    )
    parser.add_argument("--fixed-lot", type=float, default=0.0, help="Volume fixe obligatoire (ex: 0.01). 0.0 = Calcul dynamique.")
    parser.add_argument("--risk-pct", type=float, default=None, help="Pourcentage de risque personnalisé (ex: 1.5)")
    parser.add_argument(
        "--direction",
        type=str,
        default="BOTH",
        choices=["BOTH", "BUY_ONLY", "SELL_ONLY"],
        help="Filtre directionnel (BOTH, BUY_ONLY, SELL_ONLY)",
    )
    parser.add_argument("-e", "--export", type=str, default=None, help="Fichier CSV pour sauvegarder les trades (ex: trades.csv)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux (logs de débogage)")

    args = parser.parse_args()
    sys.exit(run_cli_backtest(args))


if __name__ == "__main__":
    main()
