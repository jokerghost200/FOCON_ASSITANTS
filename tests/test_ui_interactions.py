import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import yaml
from PySide6.QtWidgets import QApplication, QMessageBox, QFrame, QPushButton, QDoubleSpinBox
from PySide6.QtCore import Qt

from src.ui.app import MainWindow, ConfigDialog, ScannerThread, ChatThread
from src.core.crt_detector import CRTSetup


class TestUIInteractions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.mock_mt5 = MagicMock()
        self.mock_mt5.get_symbol_info.return_value = {
            'ask': 1.0970, 'bid': 1.0968, 'digits': 5,
            'point': 0.00001, 'trade_tick_value': 1.0, 'trade_tick_size': 0.00001,
            'trade_stops_level': 10, 'spread': 2.0
        }
        self.mock_mt5.get_open_positions.return_value = [
            {
                'ticket': 1001, 'symbol': 'EURUSD', 'type': 0, 'volume': 0.10,
                'price_open': 1.0950, 'sl': 1.0920, 'tp': 1.1000,
                'price_current': 1.0965, 'profit': 15.0
            }
        ]
        self.mock_mt5.get_account_info.return_value = {
            'balance': 2000.0, 'equity': 2015.0, 'margin': 100.0, 'margin_free': 1900.0
        }
        self.mock_mt5.get_visible_symbols.return_value = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.mock_mt5.is_market_open.return_value = True

        self.orchestrator = MagicMock()
        self.orchestrator.mt5 = self.mock_mt5
        self.orchestrator.allowed_trade_direction = 'BOTH'
        self.orchestrator.config = self.config
        self.orchestrator.scan_markets.return_value = []

        self.trade_execution = MagicMock()
        self.trade_execution.mt5 = self.mock_mt5
        self.trade_execution.min_reward_risk = 1.5

        self.chat_engine = MagicMock()
        self.chat_engine.get_response.return_value = "Réponse IA de test."
        self.trade_manager = MagicMock()

        self.win = MainWindow(
            orchestrator=self.orchestrator,
            trade_execution=self.trade_execution,
            chat_engine=self.chat_engine,
            config=self.config,
            trade_manager=self.trade_manager
        )

    def tearDown(self):
        self.win.close()

    def test_01_window_components_initialized(self):
        """Vérifie que tous les éléments de l'interface sont initialisés et conformes."""
        self.assertIsNotNone(self.win.analyze_btn)
        self.assertIsNotNone(self.win.autopilot_btn)
        self.assertIsNotNone(self.win.memory_btn)
        self.assertIsNotNone(self.win.backtest_btn)
        self.assertIsNotNone(self.win.settings_btn)
        self.assertIsNotNone(self.win.sniper_toggle_btn)
        self.assertIsNotNone(self.win.quick_tp_spin)
        self.assertIsNotNone(self.win.quick_giveback_spin)
        self.assertIsNotNone(self.win.quick_daily_spin)
        self.assertIsNotNone(self.win.quick_direction_combo)
        self.assertIsNotNone(self.win.quick_maxpos_spin)
        self.assertIsNotNone(self.win.quick_mode_combo)
        self.assertEqual(self.win.tabs.count(), 3)
        self.assertIn("Scanner", self.win.tabs.tabText(0))
        self.assertIn("Pendings", self.win.tabs.tabText(1))
        self.assertIn("Positions", self.win.tabs.tabText(2))

    def test_02_scanner_toggle(self):
        """Teste l'activation et la désactivation du bouton Scanner."""
        # Toggle ON via le bouton UI
        self.win.analyze_btn.setChecked(True)
        self.assertTrue(self.win._scanner_active)
        self.assertIn("Scan en cours", self.win.analyze_btn.text())
        self.assertIsNotNone(self.win.scanner_timer)

        # Toggle OFF via le bouton UI
        self.win.analyze_btn.setChecked(False)
        self.assertFalse(self.win._scanner_active)
        self.assertEqual(self.win.analyze_btn.text(), "🔍 LANCER LE SCANNER")

    def test_03_autopilot_toggle(self):
        """Teste l'activation et la désactivation du bouton Autopilot."""
        # Toggle ON via le bouton UI
        self.win.autopilot_btn.setChecked(True)
        self.assertTrue(self.win.autopilot_btn.isChecked())
        self.assertIn("AUTOPILOT : ON", self.win.autopilot_btn.text())
        self.assertIsNotNone(getattr(self.win, 'autopilot_timer', None))

        # Toggle OFF via le bouton UI
        self.win.autopilot_btn.setChecked(False)
        self.assertFalse(self.win.autopilot_btn.isChecked())
        self.assertIn("AUTOPILOT : OFF", self.win.autopilot_btn.text())

    def test_04_sniper_mode_toggle(self):
        """Teste le bouton Sniper Mode (bascule BALANCED <-> SNIPER)."""
        with patch.object(self.win, '_save_risk_param_to_config'):
            # Activer Sniper
            self.win.sniper_toggle_btn.setChecked(True)
            self.win.on_toggle_sniper_mode(True)
            self.assertEqual(self.win.config['risk_management']['risk_profile'], 'SNIPER')
            self.assertIn("SNIPER : ACTIF", self.win.sniper_toggle_btn.text())

            # Désactiver Sniper
            self.win.sniper_toggle_btn.setChecked(False)
            self.win.on_toggle_sniper_mode(False)
            self.assertEqual(self.win.config['risk_management']['risk_profile'], 'BALANCED')
            self.assertIn("SNIPER : OFF", self.win.sniper_toggle_btn.text())

    def test_05_quick_profit_and_risk_controls(self):
        """Teste les réglages rapides de TP, Giveback, Daily, Direction, MaxPos et Mode."""
        with patch.object(self.win, '_save_risk_param_to_config'):
            # 1. TP
            self.win.quick_tp_spin.setValue(8.50)
            self.assertEqual(self.win.trade_manager.take_profit_target_per_trade_usd, 8.50)

            # 2. Giveback
            self.win.quick_giveback_spin.setValue(1.50)
            self.assertEqual(self.win.trade_manager.giveback_usd_small, 1.50)

            # 3. Daily
            self.win.quick_daily_spin.setValue(120.0)
            self.assertEqual(self.win.trade_manager.daily_profit_target_usd, 120.0)

            # 4. Max positions
            self.win.quick_maxpos_spin.setValue(5)
            self.assertEqual(self.win.config['risk_management']['max_open_positions'], 5)

            # 5. Direction combo
            self.win.quick_direction_combo.setCurrentIndex(1)  # BUY_ONLY
            self.assertEqual(self.orchestrator.allowed_trade_direction, 'BUY_ONLY')

            self.win.quick_direction_combo.setCurrentIndex(2)  # SELL_ONLY
            self.assertEqual(self.orchestrator.allowed_trade_direction, 'SELL_ONLY')

            self.win.quick_direction_combo.setCurrentIndex(0)  # BOTH
            self.assertEqual(self.orchestrator.allowed_trade_direction, 'BOTH')

            # 6. Mode combo
            self.win.quick_mode_combo.setCurrentIndex(1)  # auto
            self.assertEqual(self.win.config['trading']['automation_mode'], 'auto')

            self.win.quick_mode_combo.setCurrentIndex(0)  # assist
            self.assertEqual(self.win.config['trading']['automation_mode'], 'assist')

    def test_06_tp_preset_buttons(self):
        """Teste tous les boutons de preset TP rapide (4$, 5$, 10$, 12$, 15$, Trailing)."""
        presets = [4.0, 5.0, 10.0, 12.0, 15.0, 0.0]
        for val in presets:
            btn = self.win.tp_preset_buttons.get(val)
            self.assertIsNotNone(btn)
            self.win.set_quick_tp_target(val)
            self.assertEqual(self.win.quick_tp_spin.value(), val)

    def test_07_signal_card_pending_crt_display(self):
        """Vérifie l'affichage précis d'une opportunité PENDING_CRT avec Order Flow."""
        signal = {
            "status": "PENDING_CRT",
            "symbol": "EURUSD",
            "timeframe": "M15",
            "decision": {
                "action": "BUY",
                "order_type": "BUY_LIMIT",
                "is_pending": True,
                "confidence": 88,
                "entry": 1.0960,
                "sl": 1.0940,
                "tp": 1.1000,
                "lot": 0.05,
                "expiration_bars": 4,
                "order_flow_confirmed": True,
                "order_flow_score": 85.0,
                "reason": "CRT Sweep C1 Low + MSS + FVG"
            }
        }
        self.win.add_signal_widget(signal)
        # Vérifie qu'un widget a été ajouté dans le scroll_layout
        self.assertGreaterEqual(self.win.scroll_layout.count(), 1)
        added_frame = self.win.scroll_layout.itemAt(self.win.scroll_layout.count() - 1).widget()
        self.assertIsInstance(added_frame, QFrame)

        # Vérifie les boutons du widget
        btns = added_frame.findChildren(QPushButton)
        self.assertGreaterEqual(len(btns), 2)
        take_btn = next((b for b in btns if "BUY_LIMIT" in b.text() or "PLACER" in b.text()), None)
        self.assertIsNotNone(take_btn)
        self.assertIn("BUY_LIMIT", take_btn.text())

    def test_08_signal_card_market_crt_display(self):
        """Vérifie l'affichage d'un signal MARKET_CRT direct."""
        signal = {
            "status": "MARKET_CRT",
            "symbol": "GBPUSD",
            "timeframe": "H1",
            "decision": {
                "action": "SELL",
                "order_type": "MARKET",
                "is_pending": False,
                "confidence": 80,
                "entry": 1.2750,
                "sl": 1.2790,
                "tp": 1.2670,
                "lot": 0.03,
                "reason": "CRT Bearish Reversal Direct Market"
            }
        }
        self.win.add_signal_widget(signal)
        added_frame = self.win.scroll_layout.itemAt(self.win.scroll_layout.count() - 1).widget()
        btns = added_frame.findChildren(QPushButton)
        take_btn = next((b for b in btns if "PRENDRE POSITION" in b.text()), None)
        self.assertIsNotNone(take_btn)

    def test_09_take_trade_pending_limit_execution(self):
        """Teste le flux complet de prise de trade sur ordre limite PENDING_CRT."""
        signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "order_type": "BUY_LIMIT",
                "is_pending": True,
                "entry": 1.0960,
                "sl": 1.0940,
                "tp": 1.1000,
                "lot": 0.05,
                "confidence": 85,
            }
        }
        mock_req = {"action": 5, "symbol": "EURUSD", "volume": 0.05, "price": 1.0960}
        self.trade_execution.prepare_order_request.return_value = mock_req
        self.trade_execution.check_order.return_value = {"valid": True}
        self.trade_execution.execute_order.return_value = {
            "success": True, "deal_id": 0, "ticket": 778899, "volume": 0.05, "price": 1.0960
        }

        frame = QFrame()
        btn = QPushButton()

        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as mock_q:
            with patch.object(QMessageBox, 'information') as mock_info:
                self.win.on_take_trade(signal, frame, btn, manual_lot=0.10)
                # Vérifie que la question a été posée
                mock_q.assert_called_once()
                # Vérifie que l'ordre a été exécuté
                self.trade_execution.execute_order.assert_called_once_with(mock_req)
                # Vérifie que le ticket MT5 apparaît dans la confirmation de succès
                mock_info.assert_called_once()
                args, _ = mock_info.call_args
                self.assertIn("778899", args[2])
                self.assertIn("Ordre Limite", args[2])

    def test_10_take_trade_check_rejected(self):
        """Teste le rejet d'un ordre par MT5 (marge insuffisante ou stops invalides)."""
        signal = {"symbol": "EURUSD", "decision": {"action": "BUY", "lot": 10.0}}
        self.trade_execution.prepare_order_request.return_value = {"action": 1}
        self.trade_execution.check_order.return_value = {
            "valid": False, "reason": "No money"
        }

        frame = QFrame()
        btn = QPushButton()
        with patch.object(QMessageBox, 'warning') as mock_warn:
            self.win.on_take_trade(signal, frame, btn)
            mock_warn.assert_called_once()
            self.assertTrue(btn.isEnabled())

    def test_11_refresh_positions_tab(self):
        """Teste le rafraîchissement du tableau des positions ouvertes."""
        self.win.refresh_positions()
        table = self.win.positions_table
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 0).text(), "1001")
        self.assertEqual(table.item(0, 1).text(), "EURUSD")
        self.assertEqual(table.item(0, 3).text(), "0.1")
        self.assertIn("15.00", table.item(0, 8).text())

    def test_12_pendings_tab_interactions(self):
        """Teste le rafraîchissement et le nettoyage des ordres en attente."""
        fake_pending = SimpleNamespace(
            ticket=554433,
            symbol="EURUSD",
            direction="BUY",
            label="Zone Test",
            entry_price=1.0960,
            status="ACTIVE",
            created_at=1000.0
        )
        mock_pm = MagicMock()
        mock_pm.get_all_pendings.return_value = [fake_pending]
        with patch.object(self.win, '_get_pending_manager', return_value=mock_pm):
            self.win.refresh_pendings_tab()
            self.assertEqual(self.win.pendings_table.rowCount(), 1)
            self.assertEqual(self.win.pendings_table.item(0, 0).text(), "554433")
            self.assertEqual(self.win.pendings_table.item(0, 1).text(), "EURUSD")
            self.assertEqual(self.win.pendings_table.item(0, 3).text(), "Zone Test")

            # Nettoyer terminés
            self.win.clean_closed_pendings()
            mock_pm.clear_inactive.assert_called_once()

    def test_13_chat_interaction(self):
        """Teste l'envoi d'un message dans le chat assistant."""
        self.win.chat_input.setText("Quelle est l'exposition actuelle ?")
        with patch('src.ui.app.ChatThread.start'):
            self.win.on_send_chat()
            self.assertEqual(self.win.chat_input.text(), "")
            self.assertIn("Quelle est l'exposition actuelle ?", self.win.chat_display.toPlainText())

    def test_14_voice_buttons(self):
        """Teste les boutons micro et muet de l'assistant vocal."""
        if hasattr(self.win, 'voice_manager') and self.win.voice_manager:
            # Voice output toggle
            self.win.on_toggle_voice_output()
            self.assertIn("Voix", self.win.voice_btn.text())

    def test_15_config_dialog_crt_tab_saving(self):
        """Vérifie que ConfigDialog présente l'onglet CRT & Order Flow et sauvegarde les paramètres."""
        dialog = ConfigDialog(self.config, parent=self.win)
        self.assertIsNotNone(dialog.crt_enabled_cb)
        self.assertIsNotNone(dialog.crt_order_mode_combo)
        self.assertIsNotNone(dialog.crt_min_rr_spin)
        self.assertIsNotNone(dialog.crt_expiration_spin)
        self.assertIsNotNone(dialog.of_enabled_cb)
        self.assertIsNotNone(dialog.of_require_conf_cb)
        self.assertIsNotNone(dialog.of_min_score_spin)

        # Modifier les valeurs
        dialog.crt_enabled_cb.setChecked(True)
        dialog.crt_order_mode_combo.setCurrentIndex(dialog.crt_order_mode_combo.findData("both"))
        dialog.crt_min_rr_spin.setValue(2.0)
        dialog.crt_expiration_spin.setValue(6)
        dialog.of_enabled_cb.setChecked(True)
        dialog.of_require_conf_cb.setChecked(True)
        dialog.of_min_score_spin.setValue(75.0)

        with patch('builtins.open', unittest.mock.mock_open()):
            with patch.object(QMessageBox, 'information'):
                dialog.save_config()

        self.assertEqual(self.config['crt']['order_mode'], 'both')
        self.assertEqual(self.config['crt']['min_rr'], 2.0)
        self.assertEqual(self.config['crt']['expiration_bars'], 6)
        self.assertTrue(self.config['order_flow']['require_confirmation'])
        self.assertEqual(self.config['order_flow']['min_confluence_score'], 75.0)


if __name__ == '__main__':
    unittest.main()
