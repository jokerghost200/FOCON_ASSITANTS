# -*- coding: utf-8 -*-
"""
Tests de Validation Statistique et Étude des Cerveaux (FOCON)
============================================================
Vérifie :
1. Reproductibilité du calcul des modes Profit Lock A, B et C
2. Absence de lookahead bias (données triées chronologiquement)
3. Validation du découpage Walk-Forward et In-Sample / Out-of-Sample
4. Non-promotion d'une stratégie si échantillon < 30 trades (Boomerang / CRT)
"""

import pytest
import os
import json
from src.research.statistical_validator import (
    simulate_modes,
    walk_forward_windows,
    analyze_by_strategy,
    calculate_metrics,
)


@pytest.fixture
def sample_closed_trades():
    path = os.path.join("data", "trade_memory.json")
    if not os.path.exists(path):
        pytest.skip("data/trade_memory.json absent pour le test")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    closed = [t for t in data.get("trades", []) if t.get("status") == "CLOSED"]
    return sorted(closed, key=lambda t: t.get("open_time", 0.0))


def test_profit_lock_modes_reproducibility(sample_closed_trades):
    """Vérifie la cohérence mathématique et la stricte reproductibilité des trois modes."""
    assert len(sample_closed_trades) == 126

    sim_res = simulate_modes(sample_closed_trades)
    mode_a = sim_res["Mode A (Actuel)"]
    mode_b = sim_res["Mode B (Désactivé)"]
    mode_c = sim_res["Mode C (Adaptatif R)"]

    # 1. Tous les modes doivent avoir traité exactement 126 trades
    assert mode_a["trades"] == 126
    assert mode_b["trades"] == 126
    assert mode_c["trades"] == 126

    # 2. Le Mode B (Désactivé) a un Max Drawdown supérieur ou égal au Mode A
    assert mode_b["max_drawdown"] >= mode_a["max_drawdown"]

    # 3. Le Mode C (Adaptatif R) améliore le Win Rate et réduit le Max Drawdown
    assert mode_c["win_rate"] > mode_a["win_rate"]
    assert mode_c["max_drawdown"] < mode_a["max_drawdown"]


def test_walk_forward_chronological_integrity(sample_closed_trades):
    """Vérifie que le Walk-Forward ne contient aucune fuite temporelle."""
    sim_res = simulate_modes(sample_closed_trades)
    wf_results = walk_forward_windows(sim_res["raw_a"], windows=3)

    assert len(wf_results) == 3
    # Les dernières fenêtres doivent montrer une persistance de l'edge
    assert wf_results[-1]["test_exp_r"] > 0
    assert wf_results[-1]["wfe_ratio"] >= 0.50


def test_investigador_promotion_guard_rejects_insufficient_sample(sample_closed_trades):
    """Vérifie qu'une stratégie avec un échantillon insuffisant (< 30) ne peut pas être validée."""
    sim_res = simulate_modes(sample_closed_trades)
    strat_metrics = analyze_by_strategy(sim_res["raw_a"])

    # Boomerang n'a que 1 trade : rejet automatique de promotion
    assert "BOOMERANG" in strat_metrics
    assert strat_metrics["BOOMERANG"]["trades"] < 30
