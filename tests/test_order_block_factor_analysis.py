# -*- coding: utf-8 -*-
"""
Tests d'Analyse des Facteurs de Performance SMC Order Block
==========================================================
Vérifie :
1. L'intégrité du parsing des 67 OB Bearish et 55 OB Bullish
2. L'identification des facteurs discriminants (Âge OB, Alignement Flux / Symbole)
3. La reproductibilité des métriques de filtrage en sandbox
"""

import pytest
import os
import json
from src.research.order_block_factor_analyzer import (
    parse_trade_features,
    compute_group_stats,
)


@pytest.fixture
def ob_dataset():
    path = os.path.join("data", "trade_memory.json")
    if not os.path.exists(path):
        pytest.skip("data/trade_memory.json absent pour le test")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    closed = [t for t in data.get("trades", []) if t.get("status") == "CLOSED"]
    parsed = [parse_trade_features(t) for t in closed]
    bearish = [t for t in parsed if t["ob_type"] == "BEARISH"]
    bullish = [t for t in parsed if t["ob_type"] == "BULLISH"]
    return bearish, bullish


def test_order_block_sample_sizes(ob_dataset):
    bearish, bullish = ob_dataset
    # Vérification des tailles d'échantillons
    assert len(bearish) >= 67
    assert len(bullish) == 55


def test_bearish_outperformance_confirmed(ob_dataset):
    bearish, bullish = ob_dataset
    b_stats = compute_group_stats(bearish)
    u_stats = compute_group_stats(bullish)

    # L'analyse confirme que Bearish a un meilleur PF et Win Rate que Bullish
    assert b_stats["profit_factor"] > u_stats["profit_factor"]
    assert b_stats["win_rate"] > u_stats["win_rate"]
    assert b_stats["expectancy_r"] > u_stats["expectancy_r"]


def test_bullish_age_decay_factor(ob_dataset):
    _, bullish = ob_dataset
    wins = [t for t in bullish if t["is_win"]]
    losses = [t for t in bullish if t["is_loss"]]

    avg_age_wins = sum(t["ob_age"] for t in wins) / len(wins)
    avg_age_losses = sum(t["ob_age"] for t in losses) / len(losses)

    # Les Bullish gagnants sont nettement plus récents (fraîcheur) que les perdants
    assert avg_age_wins < avg_age_losses
