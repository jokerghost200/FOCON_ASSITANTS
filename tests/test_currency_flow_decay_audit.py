# -*- coding: utf-8 -*-
"""
Tests de Validation de l'Audit Currency Flow & Decay Factor (FOCON)
==================================================================
Vérifie :
1. L'exactitude arithmétique de la Variante 1 (19 trades exclus, conservation de GBPNZD)
2. La décomposition rigoureuse des tranches d'âge du Decay Factor (0-10, 11-25, 26-40, >40)
3. La génération et la validité du fichier d'export CSV d'audit
4. L'absence de régression sur les métriques globales des 55 OB Bullish
"""

import pytest
import os
import csv
from src.research.audit_currency_flow_decay import (
    parse_all_bullish_ob_trades,
    audit_variant_1_exclusions,
    analyze_decay_brackets,
    sweep_age_cutoffs,
    export_trades_csv,
)


@pytest.fixture
def bullish_trades():
    return parse_all_bullish_ob_trades()


def test_audit_bullish_sample_integrity(bullish_trades):
    """Vérifie que la population des OB Bullish fermés contient exactement 55 trades."""
    assert len(bullish_trades) == 55
    total_wins = sum(1 for t in bullish_trades if t["is_win"])
    total_pnl = sum(t["pnl_usd"] for t in bullish_trades)

    assert total_wins == 18
    assert round(total_pnl, 2) == 62.78


def test_variant_1_exclusion_arithmetics(bullish_trades):
    """
    Vérifie l'arithmétique exacte de la Variante 1 :
    - 19 trades exclus (12 AUDCAD + 6 AUDNZD + 1 NZDCAD)
    - 0 gagnant dans les exclus (tous perdants ou neutralisés)
    - 36 trades conservés, préservant l'intégralité des 18 gagnants
    - GBPNZD fait partie des conservés (2 trades perdants, non exclus par v1)
    """
    res = audit_variant_1_exclusions(bullish_trades)

    assert res["excluded_count"] == 19
    assert res["excluded_wins"] == 0
    assert res["excluded_losses"] == 15
    assert round(res["excluded_pnl"], 2) == -30.80

    assert res["kept_count"] == 36
    assert res["kept_wins"] == 18
    assert round(res["kept_pnl"], 2) == 93.58

    # Vérification du cas GBPNZD
    assert res["gbpnzd_count"] == 2
    assert res["gbpnzd_losses"] == 2
    assert round(res["gbpnzd_pnl"], 2) == -9.27
    assert "GBPNZD" not in res["excluded_symbols_v1"]


def test_decay_brackets_concentration(bullish_trades):
    """
    Vérifie que la tranche '0-10 bougies' concentre l'essentiel de l'alpha positif
    et que les tranches supérieures ont une espérance dégradée.
    """
    brackets = analyze_decay_brackets(bullish_trades)

    fresh = brackets["0-10 (Ultra-frais)"]
    assert fresh["count"] == 32
    assert fresh["win_rate"] > 35.0
    assert fresh["profit_factor"] >= 2.0
    assert fresh["expectancy_r"] > 0.30
    assert fresh["net_pnl"] > 60.0

    # Les tranches intermédiaires et anciennes affichent une espérance négative
    assert brackets["11-25 (Frais standard)"]["expectancy_r"] < 0
    assert brackets["26-40 (Vieillissant)"]["expectancy_r"] < 0
    assert brackets[">40 (Ancien / Dégradé)"]["expectancy_r"] < 0


def test_export_trades_csv(bullish_trades):
    """Vérifie la création et la structure de l'export CSV d'audit."""
    test_csv = os.path.join("data", "test_export_audit_trades.csv")
    try:
        export_trades_csv(bullish_trades, filename=test_csv)
        assert os.path.exists(test_csv)
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 55
            first_row = reader[0]
            assert "ticket" in first_row
            assert "symbol" in first_row
            assert "ob_age" in first_row
            assert "v1_excluded_knife" in first_row
            assert "v2_retained_fresh25" in first_row
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
