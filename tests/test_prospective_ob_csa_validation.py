# -*- coding: utf-8 -*-
"""
Tests de Validation Prospective OB Decay et Currency Strength Alignment (FOCON)
================================================================================
Vérifie :
1. L'intégrité de l'échantillon des 122 Order Blocks et des 55 Bullish OB
2. La décomposition des 7 tranches d'âge et les intervalles de confiance de Wilson
3. La supériorité de la Combinaison C (OB + CSA) en termes de Profit Factor et Drawdown
4. La génération et la validité de l'export CSV prospectif
"""

import pytest
import os
import csv
from src.research.prospective_ob_csa_evaluator import (
    parse_all_ob_trades,
    analyze_7_age_brackets,
    compare_5_combinations,
    export_prospective_signals_csv,
    wilson_score_interval,
)


@pytest.fixture
def ob_trades():
    return parse_all_ob_trades()


def test_ob_dataset_integrity(ob_trades):
    """Vérifie que la population globale contient 122 trades OB (67 Bearish, 55 Bullish)."""
    assert len(ob_trades) == 122
    bullish = [t for t in ob_trades if t["action"] == "BUY"]
    bearish = [t for t in ob_trades if t["action"] == "SELL"]

    assert len(bullish) == 55
    assert len(bearish) == 67


def test_wilson_confidence_interval():
    """Vérifie que le calcul d'intervalle de Wilson est valide même sur de petits effectifs."""
    # 0 victoires sur 2 trades
    low, high = wilson_score_interval(0, 2)
    assert low == 0.0
    assert high > 50.0  # Incertitude forte

    # 10 victoires sur 20 trades (50%)
    low, high = wilson_score_interval(10, 20)
    assert low < 50.0 < high


def test_granular_7_age_brackets(ob_trades):
    """Vérifie que la tranche 0-5 bougies concentre le meilleur Profit Factor."""
    brackets = analyze_7_age_brackets(ob_trades)

    assert "0-5 bougies" in brackets
    assert "6-10 bougies" in brackets
    assert "11-15 bougies" in brackets
    assert "> 40 bougies" in brackets

    ultra_fresh = brackets["0-5 bougies"]
    assert ultra_fresh["n"] == 17
    assert ultra_fresh["profit_factor"] > 3.0
    assert ultra_fresh["expectancy_r"] > 0.50


def test_combination_c_csa_outperforms_baseline(ob_trades):
    """Vérifie que la Combinaison C (OB + CSA) surperforme la Baseline A sur le PF et le Drawdown."""
    combs = compare_5_combinations(ob_trades)

    base = combs["A. OB Seul (Baseline)"]
    csa = combs["C. OB + CSA (Anti-contre-courant)"]

    # Le filtre CSA dynamique améliore le PF et réduit le Drawdown
    assert csa["profit_factor"] > base["profit_factor"]
    assert csa["max_dd"] < base["max_dd"]
    assert csa["net_pnl"] > base["net_pnl"]


def test_export_prospective_signals_csv(ob_trades):
    """Vérifie la création correcte de l'export CSV prospectif."""
    test_csv = os.path.join("data", "test_prospective_signals.csv")
    try:
        export_prospective_signals_csv(ob_trades, filename=test_csv)
        assert os.path.exists(test_csv)
        with open(test_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 122
            assert "ticket" in rows[0]
            assert "csa_diff" in rows[0]
            assert "ob_age_candles" in rows[0]
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
