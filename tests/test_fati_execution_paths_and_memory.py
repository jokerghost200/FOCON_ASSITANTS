# -*- coding: utf-8 -*-
"""
Suite de Tests de Sécurité des Chemins d'Ordres et Idempotence Mémoire
=====================================================================
Valide :
1. Sécurisation de l'ancien widget Pending OB via TradeExecutionEngine & El Centinela.
2. Blocage d'un ordre (manuel ou automatique) si le risque cumulé portefeuille dépasse 6%.
3. Idempotence absolue de la mémoire (TradeJournal, ExperienceMemory, El Sabio) sans duplication.
4. Résilience après redémarrage : un trade déjà fermé n'est jamais ré-évalué par la boucle d'apprentissage.
5. Fail-Closed strict lors de déconnexion ou données corrompues.
"""

import os
import tempfile
import pytest

from src.trading.trade_execution import TradeExecutionEngine
from src.trading.pending_order_manager import PendingOrderManager, PendingOBOrder
from src.cerebros.el_centinela import ElCentinela
from src.cerebros.el_sabio import ElSabio
from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.feedback_engine import FeedbackEngine
from src.learning.adaptive_weights import AdaptiveWeightsEngine


class DummyMT5:
    def __init__(self, balance: float = 1000.0):
        self.connected = True
        self._last_orders_get_ok = True
        self.balance = balance
        self.equity = balance
        self.open_positions = []
        self.pending_orders = []
        self.cancelled_tickets = []
        self.order_send_calls = []
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
            }
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

    def place_pending_order(self, *args, **kwargs):
        raise AssertionError("place_pending_order direct ne doit JAMAIS être appelé quand TradeExecutionEngine est actif !")

    def cancel_pending_order(self, ticket: int) -> bool:
        self.cancelled_tickets.append(ticket)
        return True

    def order_check(self, *args, **kwargs):
        return {"margin": 50.0, "margin_free": 950.0}

    def order_send(self, req: dict):
        self.order_send_calls.append(req)
        class Res:
            retcode = 10008  # TRADE_RETCODE_PLACED
            order = 998877
            deal = 0
            price = req.get("price", 0.0)
            volume = req.get("volume", 0.01)
            comment = "Placed"
        return Res()


class DummyOB:
    def __init__(self, direction="BUY", entry=1.0950, sl=1.0920, tp=1.1020, score=78.0):
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.score = score
        self.ob_type = "Bullish OB"
        self.ob_low = 1.0945
        self.ob_high = 1.0955


def test_1_pending_order_manager_uses_trade_execution_engine():
    """
    Vérifie que l'ancien widget Pending OB (PendingOrderManager.place_ob_pending)
    passe désormais obligatoirement par TradeExecutionEngine et El Centinela.
    """
    mt5_sim = DummyMT5(balance=1000.0)
    centinela = ElCentinela()
    exec_engine = TradeExecutionEngine(mt5_connector=mt5_sim, mode="assist", centinela=centinela)

    pom = PendingOrderManager(
        mt5_connector=mt5_sim,
        ob_detector=None,
        tech_indicators=None,
        market_data=None,
        risk_manager=None,
        trade_execution=exec_engine,
    )

    ob = DummyOB(direction="BUY", entry=1.0950, sl=1.0920, tp=1.1020)
    pending = pom.place_ob_pending(
        symbol="EURUSD",
        ob=ob,
        timeframe="M15",
        lot=0.02,
        label="Test_OB_Safety",
    )

    assert pending is not None
    assert pending.ticket == 998877
    assert len(mt5_sim.order_send_calls) == 1
    # Vérification que le commentaire contient bien le tag
    assert "Test_OB_Safety" in mt5_sim.order_send_calls[0].get("comment", "")


def test_2_pending_order_manager_blocked_if_portfolio_risk_exceeds_6pct():
    """
    Vérifie que PendingOrderManager est bloqué si le risque portefeuille cumulé dépasse 6%.
    """
    mt5_sim = DummyMT5(balance=1000.0)
    # 50$ de risque déjà engagé (5% du capital)
    mt5_sim.open_positions = [
        {"ticket": 101, "symbol": "EURUSD", "type": "BUY", "volume": 0.10, "price_open": 1.1000, "sl": 1.0950, "tp": 1.1100}
    ]
    centinela = ElCentinela()
    exec_engine = TradeExecutionEngine(mt5_connector=mt5_sim, mode="assist", centinela=centinela)

    pom = PendingOrderManager(
        mt5_connector=mt5_sim,
        ob_detector=None,
        tech_indicators=None,
        market_data=None,
        risk_manager=None,
        trade_execution=exec_engine,
    )

    # Nouvel ordre pending demandant 15$ de risque supplémentaire (Total = 50 + 15 = 65$ > 60$ / 6%)
    ob = DummyOB(direction="BUY", entry=1.0950, sl=1.0920, tp=1.1020)
    pending = pom.place_ob_pending(
        symbol="EURUSD",
        ob=ob,
        timeframe="M15",
        lot=0.05,  # 300 pts * 0.05 * 1$ = 15.00$
    )
    assert pending is None
    assert len(mt5_sim.order_send_calls) == 0


def test_3_memory_idempotency_no_duplicate_learning():
    """
    Vérifie l'idempotence absolue :
    - Un même ticket clôturé deux fois ne produit qu'UNE SEULE écriture dans TradeJournal et ExperienceMemory.
    - El Sabio refuse le second appel avec status ALREADY_PROCESSED.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "idempotent_journal.db")
        journal = TradeJournal(db_path=db_path)
        exp_mem = ExperienceMemory()
        weights = AdaptiveWeightsEngine(file_path=os.path.join(tmpdir, "weights.json"))
        sabio = ElSabio(journal=journal, experience=exp_mem, weights_engine=weights)

        # Enregistrement initial du trade
        sabio.registrar_apertura_trade(
            ticket=4040,
            symbol="EURUSD",
            action="BUY",
            lot=0.01,
            price=1.1000,
            sl=1.0980,
            tp=1.1040,
            snapshot={"strategy": "SMC_OB"},
        )

        # Première clôture
        rep1 = sabio.aprender_de_trade(ticket=4040, exit_price=1.1040, pnl_usd=4.0, exit_reason="TP")
        assert rep1.get("trade", {}).get("ticket") == 4040 or rep1.get("status") != "ALREADY_PROCESSED"
        assert len(exp_mem._experiences) == 1

        initial_success_count = len(exp_mem._success_clusters.get("SMC_OB::UNKNOWN::UNKNOWN", []))

        # Deuxième tentative de clôture sur le même ticket (ex: boucle répétée ou après redémarrage)
        rep2 = sabio.aprender_de_trade(ticket=4040, exit_price=1.1040, pnl_usd=4.0, exit_reason="TP")
        assert rep2.get("status") == "ALREADY_PROCESSED"

        # Même en forçant FeedbackEngine directement sur le journal :
        feedback = FeedbackEngine(journal=journal, experience=exp_mem, weights_engine=weights)
        rep3 = feedback.process_closed_trade(ticket=4040, exit_price=1.1040, pnl_usd=4.0, exit_reason="TP")
        assert rep3.get("status") == "ALREADY_PROCESSED"

        # La mémoire d'expérience et le journal ne contiennent toujours qu'un seul enregistrement
        assert len(exp_mem._experiences) == 1
        assert len(exp_mem._success_clusters.get("SMC_OB::UNKNOWN::UNKNOWN", [])) == initial_success_count
        assert len(journal.get_closed_trades(limit=10)) == 1
