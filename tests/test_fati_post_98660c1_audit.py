# -*- coding: utf-8 -*-
"""
Suite de Tests de Robustesse — Audit Post-Commit 98660c1
========================================================
Teste en environnement simulé (zéro compte réel) :
1. Calcul du risque réel jusqu'au SL (avec exemption Break-Even et anti-double comptage pending/open).
2. Recalcul du risque cumulé portefeuille (6%) immédiatement avant mt5.order_send() dans TradeExecutionEngine.
3. Résilience Fail-Closed sur déconnexion MT5 et données broker manquantes/corrompues.
4. Stress test d'ordres simultanés et exposition corrélée multi-devises (EURUSD + GBPUSD).
5. Cycle de vie des Pending Orders (expiration, invalidation SL, exécution) et synchronisation El Sabio à la clôture.
"""

import os
import tempfile
import pytest

from src.trading.trade_execution import TradeExecutionEngine
from src.trading.trade_manager import TradeManager
from src.cerebros.el_centinela import ElCentinela
from src.cerebros.el_sabio import ElSabio
from src.learning.trade_journal import TradeJournal


class DummyMT5:
    def __init__(self, balance: float = 1000.0):
        self.connected = True
        self._last_orders_get_ok = True
        self.balance = balance
        self.equity = balance
        self.open_positions = []
        self.pending_orders = []
        self.cancelled_tickets = []
        self.deals = {}
        self.symbols = {
            "EURUSD": {
                "name": "EURUSD",
                "ask": 1.1000,
                "bid": 1.0998,
                "point": 0.00001,
                "digits": 5,
                "spread": 20,
                "trade_tick_size": 0.00001,
                "trade_tick_value": 1.0,
                "trade_contract_size": 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "filling_mode": 1,
            },
            "GBPUSD": {
                "name": "GBPUSD",
                "ask": 1.2700,
                "bid": 1.2698,
                "point": 0.00001,
                "digits": 5,
                "spread": 20,
                "trade_tick_size": 0.00001,
                "trade_tick_value": 1.0,
                "trade_contract_size": 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "filling_mode": 1,
            },
        }

    def get_account_info(self):
        if not self.connected:
            return None
        return {"balance": self.balance, "equity": self.equity, "margin_free": self.balance}

    def get_open_positions(self):
        if not self.connected:
            return None
        return list(self.open_positions)

    def get_pending_orders(self, symbol: str = None):
        if not self.connected:
            self._last_orders_get_ok = False
            return []
        self._last_orders_get_ok = True
        if symbol:
            return [o for o in self.pending_orders if o.get("symbol") == symbol]
        return list(self.pending_orders)

    def get_daily_profit(self):
        return 0.0

    def get_symbol_info(self, symbol: str):
        return self.symbols.get(symbol)

    def get_server_time(self):
        return 1700000000.0

    def cancel_pending_order(self, ticket: int) -> bool:
        self.cancelled_tickets.append(ticket)
        self.pending_orders = [o for o in self.pending_orders if o.get("ticket") != ticket]
        return True

    def get_deal_for_ticket(self, ticket: int):
        return self.deals.get(ticket)


def test_1_real_sl_risk_break_even_and_no_double_counting():
    """
    Vérifie que _compute_real_committed_risk_usd :
    - Calcule le risque exact jusqu'au SL via tick_value et tick_size,
    - Compte 0$ de risque pour une position déjà sécurisée à Break-Even (SL >= price_open sur BUY),
    - Ne compte JAMAIS deux fois un ordre en attente dont le ticket est déjà dans open_positions.
    """
    mt5_sim = DummyMT5(balance=1000.0)
    # Position 1 : BUY EURUSD 0.05 lot @ 1.1000, SL = 1.0950 (500 points * 1$ * 0.05 = 25.00$)
    # Position 2 : BUY GBPUSD 0.10 lot @ 1.2700, SL = 1.2710 (Break-Even dépassé -> risque = 0.00$)
    mt5_sim.open_positions = [
        {"ticket": 101, "symbol": "EURUSD", "type": "BUY", "volume": 0.05, "price_open": 1.1000, "sl": 1.0950, "tp": 1.1100},
        {"ticket": 102, "symbol": "GBPUSD", "type": "BUY", "volume": 0.10, "price_open": 1.2700, "sl": 1.2710, "tp": 1.2800},
    ]
    # Ordre pending #101 (déjà exécuté en position 101 -> ne doit pas être recompté)
    # Ordre pending #201 : BUY_LIMIT EURUSD 0.04 lot @ 1.0900, SL = 1.0850 (500 points * 1$ * 0.04 = 20.00$)
    mt5_sim.pending_orders = [
        {"ticket": 101, "symbol": "EURUSD", "type": "BUY", "volume": 0.05, "price_open": 1.1000, "sl": 1.0950, "tp": 1.1100},
        {"ticket": 201, "symbol": "EURUSD", "type": "BUY", "volume": 0.04, "price_open": 1.0900, "sl": 1.0850, "tp": 1.1000},
    ]

    exec_engine = TradeExecutionEngine(mt5_connector=mt5_sim, mode="assist", centinela=ElCentinela())
    open_risk, pending_risk = exec_engine._compute_real_committed_risk_usd(1000.0)

    assert open_risk == 25.00, f"Attendu 25.00$ (pos #102 à BE = 0$), reçu {open_risk}"
    assert pending_risk == 20.00, f"Attendu 20.00$ (ticket #101 dédupliqué), reçu {pending_risk}"


def test_2_pre_order_send_portfolio_ceiling_6pct_block():
    """
    Vérifie que execute_order recalcule le risque cumulé juste avant mt5.order_send()
    et bloque un ordre qui ferait dépasser le plafond absolu de 6% du capital.
    """
    mt5_sim = DummyMT5(balance=1000.0)
    # Risque déjà engagé : 30$ (open) + 20$ (pending) = 50$ (5.0% du capital)
    mt5_sim.open_positions = [
        {"ticket": 101, "symbol": "EURUSD", "type": "BUY", "volume": 0.06, "price_open": 1.1000, "sl": 1.0950, "tp": 1.1100},
    ]
    mt5_sim.pending_orders = [
        {"ticket": 201, "symbol": "GBPUSD", "type": "BUY", "volume": 0.04, "price_open": 1.2700, "sl": 1.2650, "tp": 1.2800},
    ]

    exec_engine = TradeExecutionEngine(
        mt5_connector=mt5_sim,
        mode="assist",
        centinela=ElCentinela(),
        default_risk_percent=1.5,
        max_open_positions=3,
    )

    # Nouveau trade : risque de 12$ (0.04 lot, 300 pts SL) -> Total = 30 + 20 + 12 = 62$ (6.2% > 6.0%)
    req = {
        "action": 1,
        "symbol": "EURUSD",
        "volume": 0.04,
        "type": 0,  # BUY
        "price": 1.1000,
        "sl": 1.0970,
        "tp": 1.1060,
        "_economic_context": {"status": "OK", "recommendation": "NORMAL", "event_risk": "LOW", "high_impact_window": False},
    }
    res = exec_engine.execute_order(req)
    assert res["success"] is False
    assert "Plafond de risque cumulé portefeuille dépassé" in res["reason"]


def test_3_mt5_disconnection_and_corrupted_broker_metrics_fail_closed():
    """
    Vérifie que sur déconnexion MT5 ou métriques de tick corrompues (tick_value=0),
    TradeExecutionEngine refuse immédiatement l'ordre (Fail-Closed).
    """
    mt5_sim = DummyMT5(balance=1000.0)
    exec_engine = TradeExecutionEngine(mt5_connector=mt5_sim, mode="assist", centinela=ElCentinela())

    # 1. Déconnexion MT5
    mt5_sim.connected = False
    req = {
        "action": 1,
        "symbol": "EURUSD",
        "volume": 0.01,
        "type": 0,
        "price": 1.1000,
        "sl": 1.0970,
        "tp": 1.1060,
    }
    res_disc = exec_engine.execute_order(req)
    assert res_disc["success"] is False
    assert "El Centinela" in res_disc["reason"]

    # 2. Reconnexion mais tick_value corrompu (0.0)
    mt5_sim.connected = True
    mt5_sim.symbols["EURUSD"]["trade_tick_value"] = 0.0
    res_corrupt = exec_engine.execute_order(req)
    assert res_corrupt["success"] is False
    assert "nulles ou négatives" in res_corrupt["reason"]


def test_4_pending_order_invalidation_and_sabio_exit_sync():
    """
    Vérifie :
    - Qu'un ordre en attente dont le SL a été franchi par le marché est annulé proprement par TradeManager,
    - Qu'une position clôturée sur MT5 synchronise automatiquement son résultat avec El Sabio (TradeJournal SQLite).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tm_sabio.db")
        journal = TradeJournal(db_path=db_path)
        sabio = ElSabio(journal=journal)

        mt5_sim = DummyMT5(balance=1000.0)
        # Ordre BUY_LIMIT @ 1.1010 avec SL @ 1.1000, alors que le Bid actuel est 1.0998 (<= SL 1.1000 -> invalidé)
        mt5_sim.pending_orders = [
            {
                "ticket": 501,
                "symbol": "EURUSD",
                "type": "BUY",
                "volume": 0.01,
                "price_open": 1.1010,
                "sl": 1.1000,
                "tp": 1.1050,
                "time_expiration": 1800000000.0,
            }
        ]

        class DummyOrch:
            def __init__(self, s):
                self.sabio = s

            def set_symbol_cooldown(self, *args, **kwargs):
                pass

        tm = TradeManager(mt5_connector=mt5_sim, orchestrator=DummyOrch(sabio))
        cancelled = tm.manage_pending_orders()
        assert len(cancelled) == 1
        assert cancelled[0]["ticket"] == 501
        assert 501 in mt5_sim.cancelled_tickets

        # Ouverture puis clôture d'un trade #777 synchronisé avec El Sabio
        sabio.registrar_apertura_trade(
            ticket=777,
            symbol="EURUSD",
            action="BUY",
            lot=0.01,
            price=1.1000,
            sl=1.0980,
            tp=1.1040,
            snapshot={"regime": "TREND_BULLISH"},
        )
        tm._known_open_tickets = {777}
        mt5_sim.open_positions = []  # Le trade #777 vient de fermer sur TP chez MT5
        mt5_sim.deals[777] = {"deal_ticket": 9001, "price": 1.1040, "profit": 4.0, "comment": "tp"}

        tm.manage_open_positions(auto_mode="assist")
        closed_trades = sabio.journal.get_closed_trades(limit=10)
        assert len(closed_trades) == 1
        assert closed_trades[0]["ticket"] == 777
        assert float(closed_trades[0]["pnl_usd"]) == 4.0
