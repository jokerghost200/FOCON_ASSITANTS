"""
Tests unitaires pour RiskManager (calcul de lot, logique pure sans MT5).
Lancer avec :  python -m pytest tests/ -v  (ou python tests/test_risk_manager.py)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.risk = RiskManager(default_risk_percent=0.1,
                                max_daily_loss_percent=5.0,
                                global_max_lot=1.0)

    def test_zero_risk_uses_default(self):
        # capital 10000 €, risque 0 -> fallback 0.1%
        lot, raw, risk_amt, loss_per_lot = self.risk.calculate_lot_size(
            capital=10000.0, risk_percent=0.0, entry_price=1.1000,
            sl_price=1.0900, contract_size=100000.0, tick_size=0.00001,
            tick_value=1.0, min_lot=0.01, max_lot=100.0, lot_step=0.01
        )
        # Risque = 10 € ; perte par lot = 1000 ticks * 1$ = 1000 €/lot
        self.assertAlmostEqual(raw, 0.01, places=2)
        self.assertAlmostEqual(risk_amt, 10.0, places=2)

    def test_lot_respects_risk_amount(self):
        lot, raw, risk_amt, loss_per_lot = self.risk.calculate_lot_size(
            capital=10000.0, risk_percent=1.0, entry_price=1.1000,
            sl_price=1.0900, contract_size=100000.0, tick_size=0.00001,
            tick_value=1.0, min_lot=0.01, max_lot=100.0, lot_step=0.01
        )
        # Risque = 100 €, perte/lot = 1000 $. => lot = 0.10
        self.assertAlmostEqual(lot, 0.10, places=2)
        self.assertLessEqual(lot * loss_per_lot, risk_amt * 1.05)

    def test_global_max_lot_enforced(self):
        lot, raw, risk_amt, loss_per_lot = self.risk.calculate_lot_size(
            capital=1000000.0, risk_percent=1.0, entry_price=1.1000,
            sl_price=1.0900, contract_size=100000.0, tick_size=0.00001,
            tick_value=1.0, min_lot=0.01, max_lot=100.0, lot_step=0.01
        )
        # Risque = 10000 €, perte/lot = 1000 € => lot brut = 10, mais cap global à 1.0
        self.assertLessEqual(lot, 1.0)
        self.assertGreaterEqual(lot, 0.01)

    def test_zero_sl_distance_returns_zero_lot_and_fails_closed(self):
        lot, raw, risk_amt, loss_per_lot = self.risk.calculate_lot_size(
            capital=10000.0, risk_percent=1.0, entry_price=1.1000,
            sl_price=1.1000, contract_size=100000.0, tick_size=0.00001,
            tick_value=1.0, min_lot=0.01, max_lot=100.0, lot_step=0.01
        )
        self.assertEqual(lot, 0.0)
        self.assertEqual(loss_per_lot, 0.0)

    def test_zero_tick_value_fails_closed(self):
        # tick_value = 0 -> loss_per_lot <= 0 -> doit renvoyer 0.0 pour bloquer l'ordre
        result = self.risk.calculate_lot_size(
            capital=10000.0, risk_percent=1.0, entry_price=1.1000,
            sl_price=1.0900, contract_size=100000.0, tick_size=0.00001,
            tick_value=0.0, min_lot=0.01, max_lot=100.0, lot_step=0.01
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[3], 0.0)

    def test_fixed_lot_exceeding_capital_is_rejected(self):
        # Fixed lot 1.2 sur compte de 500 $ avec SL de 50 pips (perte = 600 $ > capital disponible de 500 $)
        risk_mgr = RiskManager(fixed_lot_size=1.2)
        account = {"balance": 500.0, "margin_free": 500.0}
        symbol = {
            "name": "EURUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 100000.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
            "trade_stops_level": 0, "point": 0.00001, "spread": 10,
        }
        proposal = {
            "action": "BUY",
            "entry": 1.1000,
            "sl": 1.0950, # 50 pips -> 500 ticks * 1$ * 1.2 = 600 $
            "tp": 1.1150,
            "confidence": 80.0
        }
        res = risk_mgr.validate_proposal(proposal, account, symbol)
        self.assertFalse(res["valid"])
        self.assertIn("Lot fixe", res["reason"])

    def test_validate_proposal_missing_sl(self):
        account = {"balance": 10000.0, "margin_free": 10000.0}
        symbol = {"name": "EURUSD", "volume_max": 100.0, "volume_min": 0.01,
                  "volume_step": 0.01, "trade_contract_size": 100000.0,
                  "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
                  "trade_stops_level": 0, "point": 0.00001}
        res = self.risk.validate_proposal({"action": "BUY", "entry": 1.10},
                                          account, symbol)
        self.assertFalse(res["valid"])

    def test_adaptive_spread_buffer_adapts_sl_and_reduces_lot(self):
        """Vérifie que le Spread Buffer élargit le SL et réduit le lot quand spread/SL > seuil."""
        risk_mgr = RiskManager(
            default_risk_percent=1.0,
            spread_buffer_enabled=True,
            max_spread_sl_ratio=0.25,
            spread_buffer_max_cap=0.40,
        )
        account = {"balance": 10000.0, "margin_free": 10000.0}
        # Spread = 0.04 (4000 points à 0.00001), SL distance = 0.10 -> ratio = 40%
        symbol = {
            "name": "ADAUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 1.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 0.00001,
            "trade_stops_level": 0, "point": 0.00001, "spread": 4000,
        }
        proposal = {
            "action": "SELL",
            "entry": 0.3000,
            "sl": 0.4000,  # SL distance = 0.10. Spread = 0.04 -> ratio = 40% (<= cap 40%)
            "tp": 0.1000,  # R:R = 2.0
            "confidence": 80.0,
        }
        res = risk_mgr.validate_proposal(proposal, account, symbol)
        self.assertTrue(res["valid"])
        # SL doit avoir été élargi pour absorber le spread (sl > 0.4000 pour un SELL)
        self.assertGreater(res["proposal"]["sl"], 0.4000)
        # TP doit être recalibré à la baisse pour préserver le R:R
        self.assertLess(res["proposal"]["tp"], 0.1000)
        # Le lot doit être positif et valide
        self.assertGreater(res["proposal"]["lot"], 0)

    def test_spread_buffer_disabled_rejects_high_spread(self):
        """Vérifie que si le spread buffer est désactivé, le trade est rejeté."""
        risk_mgr = RiskManager(
            default_risk_percent=1.0,
            spread_buffer_enabled=False,
            max_spread_sl_ratio=0.25,
            spread_buffer_max_cap=0.40,
        )
        account = {"balance": 10000.0, "margin_free": 10000.0}
        symbol = {
            "name": "ADAUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 1.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 0.00001,
            "trade_stops_level": 0, "point": 0.00001, "spread": 4000,
        }
        proposal = {
            "action": "SELL",
            "entry": 0.3000,
            "sl": 0.4000,  # ratio = 40% > 25%
            "tp": 0.1000,
            "confidence": 80.0,
        }
        res = risk_mgr.validate_proposal(proposal, account, symbol)
        self.assertFalse(res["valid"])
        self.assertIn("Spread trop élevé", res["reason"])

    def test_spread_exceeds_max_cap_rejects(self):
        """Vérifie qu'un spread toxique au-delà du plafond absolu (ex: 50% > 40%) est toujours rejeté."""
        risk_mgr = RiskManager(
            default_risk_percent=1.0,
            spread_buffer_enabled=True,
            max_spread_sl_ratio=0.25,
            spread_buffer_max_cap=0.40,
        )
        account = {"balance": 10000.0, "margin_free": 10000.0}
        # Spread = 0.05, SL distance = 0.10 -> ratio = 50% > 40% cap
        symbol = {
            "name": "ADAUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 1.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 0.00001,
            "trade_stops_level": 0, "point": 0.00001, "spread": 5000,
        }
        proposal = {
            "action": "SELL",
            "entry": 0.3000,
            "sl": 0.4000,
            "tp": 0.1000,
            "confidence": 80.0,
        }
        res = risk_mgr.validate_proposal(proposal, account, symbol)
        self.assertFalse(res["valid"])
        self.assertIn("Spread excessif vs SL", res["reason"])

    def test_get_account_tier_boundaries(self):
        """Vérifie la détection automatique des 4 paliers de compte (Micro, Standard, Growth, Institutional)."""
        self.assertEqual(RiskManager.get_account_tier(50.0)["tier_id"], "MICRO")
        self.assertEqual(RiskManager.get_account_tier(249.99)["tier_id"], "MICRO")
        self.assertEqual(RiskManager.get_account_tier(250.0)["tier_id"], "STANDARD")
        self.assertEqual(RiskManager.get_account_tier(2499.0)["tier_id"], "STANDARD")
        self.assertEqual(RiskManager.get_account_tier(2500.0)["tier_id"], "GROWTH")
        self.assertEqual(RiskManager.get_account_tier(9999.0)["tier_id"], "GROWTH")
        self.assertEqual(RiskManager.get_account_tier(10000.0)["tier_id"], "INSTITUTIONAL")
        self.assertEqual(RiskManager.get_account_tier(50000.0)["tier_id"], "INSTITUTIONAL")


    def test_apply_risk_profile_presets(self):
        """Vérifie que les profils prédéfinis ajustent correctement le risque et l'objectif journalier."""
        rm = RiskManager()
        rm.apply_risk_profile("PRUDENT")
        self.assertAlmostEqual(rm.default_risk_percent, 0.25)
        self.assertAlmostEqual(rm.max_daily_loss_percent, 1.0)
        self.assertAlmostEqual(rm.min_confidence, 75.0)

        rm.apply_risk_profile("CONSERVATIVE")
        self.assertAlmostEqual(rm.default_risk_percent, 0.5)
        self.assertAlmostEqual(rm.max_daily_loss_percent, 2.5)

        rm.apply_risk_profile("BALANCED")
        self.assertAlmostEqual(rm.default_risk_percent, 1.0)
        self.assertAlmostEqual(rm.max_daily_loss_percent, 4.0)

        rm.apply_risk_profile("AGGRESSIVE")
        self.assertAlmostEqual(rm.default_risk_percent, 1.5)
        self.assertAlmostEqual(rm.max_daily_loss_percent, 5.0)

    def test_intransigent_risk_rejects_when_min_lot_exceeds_budget(self):
        """Priorité 1 de l'audit : Si le lot minimum imposé par le courtier dépasse le budget de perte alloué,
        le trade DOIT être formellement rejeté. Ne jamais autoriser un dépassement de budget de perte."""
        rm = RiskManager(
            default_risk_percent=0.25,  # 0.25% de 1000$ = 2.50$ max autorisés
            auto_tier_enabled=False     # Standard / Sans dérogation micro
        )
        account = {"balance": 1000.0, "margin_free": 1000.0}
        # EURUSD : 1 pip = 10$ par lot standard, 0.10$ par microlot 0.01
        # Avec un SL de 50 pips, perte minimale au lot min (0.01) = 5.00$
        # Or le budget autorisé à 0.25% n'est que de 2.50$.
        symbol = {
            "name": "EURUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 100000.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
            "trade_stops_level": 0, "point": 0.00001, "spread": 10,
        }
        proposal = {
            "action": "BUY",
            "entry": 1.1000,
            "sl": 1.0950,  # 50 pips de SL -> perte de 5.00$ > budget de 2.50$
            "tp": 1.1150,
            "confidence": 85.0
        }
        res = rm.validate_proposal(proposal, account, symbol)
        self.assertFalse(res["valid"])
        self.assertIn("Risque Intransigeant", res["reason"])


    def test_get_effective_max_open_positions(self):
        """Vérifie le bridage dynamique des positions simultanées selon le capital."""
        rm = RiskManager(auto_tier_enabled=True)
        # Micro account (< $250) -> max 1 trade même si la base autorise 5
        self.assertEqual(rm.get_effective_max_open_positions(100.0, base_max_pos=5), 1)
        # Standard account ($1000) -> max 2 trades
        self.assertEqual(rm.get_effective_max_open_positions(1000.0, base_max_pos=5), 2)
        # Institutional ($50000) -> min(4, 5) = 4
        self.assertEqual(rm.get_effective_max_open_positions(50000.0, base_max_pos=5), 4)

        # Si auto_tier désactivé -> respecte strictement la base
        rm_disabled = RiskManager(auto_tier_enabled=False)
        self.assertEqual(rm_disabled.get_effective_max_open_positions(100.0, base_max_pos=5), 5)

    def test_micro_account_min_lot_risk_cap_protection(self):
        """Vérifie que la protection micro-compte bloque un trade si le lot min (0.01) dépasse le risque max toléré (ex: 4%)."""
        rm = RiskManager(
            default_risk_percent=1.0,
            auto_tier_enabled=True,
            max_micro_risk_cap_percent=4.0
        )
        # Capital de 50 $. Risque visé 1% = 0.50 $.
        # Avec un SL de 100 pips sur EURUSD, lot min 0.01 = perte de 10 $ = 20 % du compte !
        account = {"balance": 50.0, "margin_free": 50.0}
        symbol = {
            "name": "EURUSD", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 100000.0,
            "trade_tick_size": 0.00001, "trade_tick_value": 1.0,
            "trade_stops_level": 0, "point": 0.00001, "spread": 10,
        }
        proposal_toxic = {
            "action": "BUY",
            "entry": 1.1000,
            "sl": 1.0900,  # 100 pips -> perte 10.00 $ = 20.0% du compte (dépasse 4.0%)
            "tp": 1.1200,
            "confidence": 85.0
        }
        res_toxic = rm.validate_proposal(proposal_toxic, account, symbol)
        self.assertFalse(res_toxic["valid"])
        self.assertIn("lot minimum broker", res_toxic["reason"])


        # SL très court (15 pips) : perte 0.01 = 1.50 $ = 3.0 % du compte (<= 4.0 %)
        proposal_safe = {
            "action": "BUY",
            "entry": 1.1000,
            "sl": 1.0985,  # 15 pips -> perte 1.50 $ = 3.0% du compte (autorisé)
            "tp": 1.1030,
            "confidence": 85.0
        }
        res_safe = rm.validate_proposal(proposal_safe, account, symbol)
        self.assertTrue(res_safe["valid"])

    def test_fixed_lot_when_auto_tier_disabled(self):
        """Vérifie que le lot fixe n'est pas bridé arbitrairement à 0.02 quand auto_tier est désactivé."""
        from unittest.mock import patch
        rm = RiskManager(
            auto_tier_enabled=False,
            fixed_lot_size=0.10
        )
        account = {"balance": 60.0, "margin_free": 55.0}
        symbol = {
            "name": "USDJPY", "volume_max": 100.0, "volume_min": 0.01,
            "volume_step": 0.01, "trade_contract_size": 100000.0,
            "trade_tick_size": 0.001, "trade_tick_value": 0.65,
            "trade_stops_level": 0, "point": 0.001, "spread": 2,
        }
        proposal = {
            "action": "BUY",
            "entry": 158.00,
            "sl": 157.80,
            "tp": 158.50,
            "confidence": 85.0
        }
        # 1 lot requires ~200 USD margin; for 0.10 lot, requires ~20 USD margin.
        # With 55 USD free margin and 75% ratio (41.25 USD limit), 0.10 lot fits easily!
        with patch("MetaTrader5.order_calc_margin", return_value=200.0):
            res = rm.validate_proposal(proposal, account, symbol)
            self.assertTrue(res["valid"])
            self.assertEqual(res["proposal"]["lot"], 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)