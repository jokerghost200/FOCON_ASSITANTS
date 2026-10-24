"""
Backtest Dialog (Interface Graphique de Simulation MT5)
=======================================================
Fenêtre interactive permettant de tester le robot sur les bougies réelles MT5
avec n'importe quel capital (du micro-compte $20-$100 aux grands capitaux),
d'analyser les positions prises, les gains/pertes et la conclusion diagnostique du bot.
"""

import logging
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QFrame,
    QMessageBox, QSplitter, QAbstractItemView, QWidget
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

from src.trading.backtester import BacktestEngine
from src.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestWorkerThread(QThread):
    progress = Signal(float, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        engine: BacktestEngine,
        symbol: str,
        timeframe: str,
        num_bars: int,
        initial_capital: float,
        risk_profile: str,
        risk_percent: float,
        direction_filter: str,
        strategy_filter: str = "ALL",
        fixed_lot_size: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars
        self.initial_capital = initial_capital
        self.risk_profile = risk_profile
        self.risk_percent = risk_percent
        self.direction_filter = direction_filter
        self.strategy_filter = strategy_filter
        self.fixed_lot_size = fixed_lot_size
        self._is_stopped = False

    def run(self):
        try:
            def _cb(pct, msg):
                if not self._is_stopped:
                    self.progress.emit(pct, msg)

            res = self.engine.run_backtest(
                symbol=self.symbol,
                timeframe=self.timeframe,
                num_bars=self.num_bars,
                initial_capital=self.initial_capital,
                risk_profile=self.risk_profile,
                risk_percent=self.risk_percent,
                direction_filter=self.direction_filter,
                strategy_filter=self.strategy_filter,
                fixed_lot_size=self.fixed_lot_size,
                progress_cb=_cb,
                stop_event=self,
            )
            if "error" in res:
                self.failed.emit(res["error"])
            else:
                self.finished.emit(res)
        except Exception as e:
            logger.exception("Erreur durant l'exécution du backtest")
            self.failed.emit(str(e))

    def stop(self):
        self._is_stopped = True

    def is_set(self):
        return self._is_stopped


class BacktestDialog(QDialog):
    def __init__(self, mt5_connector, risk_manager=None, weights_engine=None, parent=None):
        super().__init__(parent)
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager or RiskManager()
        self.weights_engine = weights_engine
        self.engine = BacktestEngine(
            mt5_connector=self.mt5,
            risk_manager=self.risk_manager,
            weights_engine=self.weights_engine,
        )
        self.worker = None

        self.setWindowTitle("🔬 LABORATOIRE DE BACKTEST MT5 — SIMULATION VIRTUELLE MULTI-CAPITAL")
        self.resize(1180, 750)
        self.setMinimumSize(950, 620)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # ── 1. En-tête Moderne ────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a237e, stop:1 #004d40);
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #3949ab;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(12, 6, 12, 6)

        title_lbl = QLabel("🔬 BACKTEST SUR BOUGIES RÉELLES MT5")
        title_lbl.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title_lbl.setStyleSheet("color: #ffffff;")

        subtitle_lbl = QLabel(
            "Testez virtuellement les stratégies institutionnelles (Order Blocks SMC, Boomerang, Anti-FOMO, Trailing)\n"
            "avec n'importe quel capital : analysez les positions gagnées/perdues et la conclusion diagnostique du robot."
        )
        subtitle_lbl.setStyleSheet("color: #b2dfdb; font-size: 11px;")

        h_vbox = QVBoxLayout()
        h_vbox.addWidget(title_lbl)
        h_vbox.addWidget(subtitle_lbl)
        h_layout.addLayout(h_vbox)
        h_layout.addStretch()

        self.live_tier_badge = QLabel("🟢 TIER: STANDARD")
        self.live_tier_badge.setStyleSheet("""
            QLabel {
                background-color: #102027;
                color: #80cbc4;
                font-weight: bold;
                padding: 8px 14px;
                border-radius: 6px;
                border: 1px solid #00897b;
                font-size: 12px;
            }
        """)
        h_layout.addWidget(self.live_tier_badge)
        main_layout.addWidget(header_frame)

        # ── 2. Splitter Paramètres (Gauche) / Résultats (Droite) ──────────────
        splitter = QSplitter(Qt.Horizontal)

        # --- Panneau Gauche : Configuration ---
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 10, 0)
        config_layout.setSpacing(10)

        params_frame = QFrame()
        params_frame.setStyleSheet("""
            QFrame {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel {
                color: #d1d4dc;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #131722;
                color: #ffffff;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #2962ff;
            }
        """)
        params_grid = QGridLayout(params_frame)
        params_grid.setSpacing(10)

        # Symbole
        params_grid.addWidget(QLabel("Actif / Marché :"), 0, 0)
        self.symbol_combo = QComboBox()
        self.populate_symbols()
        params_grid.addWidget(self.symbol_combo, 0, 1)

        # Timeframe
        params_grid.addWidget(QLabel("Timeframe :"), 1, 0)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["M5", "M15", "H1", "H4", "D1"])
        self.tf_combo.setCurrentText("M15")
        params_grid.addWidget(self.tf_combo, 1, 1)

        # Nombre de bougies
        params_grid.addWidget(QLabel("Bougies réelles :"), 2, 0)
        self.bars_combo = QComboBox()
        self.bars_combo.addItem("300 bougies (~3 jours M15)", 300)
        self.bars_combo.addItem("600 bougies (~1 semaine M15)", 600)
        self.bars_combo.addItem("1200 bougies (~2 semaines M15)", 1200)
        self.bars_combo.addItem("2500 bougies (~1 mois M15)", 2500)
        self.bars_combo.setCurrentIndex(1)
        params_grid.addWidget(self.bars_combo, 2, 1)

        # Capital Initial
        params_grid.addWidget(QLabel("Capital Initial ($) :"), 3, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(10.0, 1000000.0)
        self.capital_spin.setValue(100.0)
        self.capital_spin.setSingleStep(50.0)
        self.capital_spin.setDecimals(2)
        self.capital_spin.setSuffix(" $")
        self.capital_spin.valueChanged.connect(self.on_capital_changed)
        params_grid.addWidget(self.capital_spin, 3, 1)

        # Boutons rapides capital
        quick_cap_layout = QHBoxLayout()
        for cap_val in [50, 100, 250, 1000, 5000]:
            btn = QPushButton(f"{cap_val}$")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2e39;
                    color: #80cbc4;
                    border: 1px solid #363c4e;
                    border-radius: 3px;
                    padding: 4px 2px;
                    font-size: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #00897b;
                    color: white;
                }
            """)
            btn.clicked.connect(lambda _, v=cap_val: self.capital_spin.setValue(v))
            quick_cap_layout.addWidget(btn)
        params_grid.addLayout(quick_cap_layout, 4, 0, 1, 2)

        # Profil de risque
        params_grid.addWidget(QLabel("Profil de Risque :"), 5, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("⚖️ Équilibré (1.0% / trade)", "BALANCED")
        self.profile_combo.addItem("🛡️ Prudent (0.5% / trade)", "CONSERVATIVE")
        self.profile_combo.addItem("🚀 Dynamique (2.0% / trade)", "AGGRESSIVE")
        self.profile_combo.addItem("⚙️ Personnalisé", "CUSTOM")
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        params_grid.addWidget(self.profile_combo, 5, 1)

        # Risque % par trade
        params_grid.addWidget(QLabel("Risque / Trade (%) :"), 6, 0)
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setSingleStep(0.1)
        self.risk_spin.setDecimals(1)
        self.risk_spin.setSuffix(" %")
        params_grid.addWidget(self.risk_spin, 6, 1)

        # Sens du marché
        params_grid.addWidget(QLabel("Sens autorisé :"), 7, 0)
        self.dir_combo = QComboBox()
        self.dir_combo.addItem("🔄 Tous les sens (Achat + Vente)", "BOTH")
        self.dir_combo.addItem("📈 Hausse seule (BUY_ONLY)", "BUY_ONLY")
        self.dir_combo.addItem("📉 Baisse seule (SELL_ONLY)", "SELL_ONLY")
        params_grid.addWidget(self.dir_combo, 7, 1)

        # Stratégie à tester
        params_grid.addWidget(QLabel("Stratégie :"), 8, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("🔥 Toutes (Synergie complète)", "ALL")
        self.strategy_combo.addItem("🎯 Candle Range Theory (CRT Limits)", "CRT")
        self.strategy_combo.addItem("🧱 Order Blocks SMC & FVG", "ORDER_BLOCKS")
        self.strategy_combo.addItem("🪃 Boomerang (Rebond EMA + ZigZag)", "BOOMERANG")
        params_grid.addWidget(self.strategy_combo, 8, 1)

        # Volume Fixe optionnel
        params_grid.addWidget(QLabel("Volume Fixe (0 = AutoPilot) :"), 9, 0)
        self.fixed_lot_spin = QDoubleSpinBox()
        self.fixed_lot_spin.setRange(0.0, 100.0)
        self.fixed_lot_spin.setSingleStep(0.01)
        self.fixed_lot_spin.setDecimals(2)
        init_fl = float(getattr(self.risk_manager, "fixed_lot_size", 0.0) or 0.0)
        self.fixed_lot_spin.setValue(init_fl)
        self.fixed_lot_spin.setToolTip("Volume fixe (ex: 0.2 ou 1.0). Si 0.0, le robot calcule le volume dynamiquement selon le % de risque.")
        params_grid.addWidget(self.fixed_lot_spin, 9, 1)

        config_layout.addWidget(params_frame)

        # Bouton d'action Démarrer
        self.btn_run = QPushButton("🚀 LANCER LE BACKTEST VIRTUEL")
        self.btn_run.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_run.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00897b, stop:1 #00b0ff);
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00bfa5, stop:1 #40c4ff);
            }
            QPushButton:disabled {
                background-color: #455a64;
                color: #90a4ae;
            }
        """)
        self.btn_run.clicked.connect(self.on_start_backtest)
        config_layout.addWidget(self.btn_run)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #37474f;
                border-radius: 4px;
                text-align: center;
                background-color: #102027;
                color: white;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #00b0ff;
                border-radius: 3px;
            }
        """)
        config_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Prêt à simuler.")
        self.status_lbl.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        config_layout.addWidget(self.status_lbl)
        config_layout.addStretch()

        splitter.addWidget(config_widget)

        # --- Panneau Droit : Résultats & Tableaux ---
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(10, 0, 0, 0)
        results_layout.setSpacing(8)

        # KPI Cards (Statistiques clés)
        kpi_frame = QFrame()
        kpi_frame.setStyleSheet("""
            QFrame {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        kpi_layout = QHBoxLayout(kpi_frame)
        kpi_layout.setSpacing(8)

        self.kpi_profit = self._create_kpi_card("SOLDE FINAL", "100.00 $", "Profit: +0.00 $ (0.0%)", "#29b6f6")
        self.kpi_winrate = self._create_kpi_card("TAUX DE RÉUSSITE", "0.0 %", "0 Gagnés / 0 Perdus", "#ffb74d")
        self.kpi_pf = self._create_kpi_card("PROFIT FACTOR", "1.00", "Qualité R/R", "#ba68c8")
        self.kpi_dd = self._create_kpi_card("MAX DRAWDOWN", "0.0 %", "-0.00 $", "#ef5350")
        self.kpi_safety = self._create_kpi_card("SÉCURITÉ MICRO", "0 Filtrés", "Risque 0.01 bloqué", "#81c784")

        kpi_layout.addWidget(self.kpi_profit)
        kpi_layout.addWidget(self.kpi_winrate)
        kpi_layout.addWidget(self.kpi_pf)
        kpi_layout.addWidget(self.kpi_dd)
        kpi_layout.addWidget(self.kpi_safety)
        results_layout.addWidget(kpi_frame)

        # Tabs : Conclusion / Journal des Trades / Courbe
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2a2e39;
                background-color: #131722;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #1e222d;
                color: #90a4ae;
                padding: 8px 16px;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2a2e39;
                color: #00e676;
                border-bottom: 2px solid #00e676;
            }
        """)

        # Tab 1: Conclusion du Bot
        tab_conclusion = QWidget()
        lay_conc = QVBoxLayout(tab_conclusion)
        self.conclusion_text = QTextEdit()
        self.conclusion_text.setReadOnly(True)
        self.conclusion_text.setFont(QFont("Consolas", 10))
        self.conclusion_text.setStyleSheet("""
            QTextEdit {
                background-color: #131722;
                color: #eceff1;
                border: none;
                padding: 10px;
                line-height: 1.4;
            }
        """)
        self.conclusion_text.setPlainText("Lancez une simulation pour obtenir la conclusion diagnostique approfondie du robot...")
        lay_conc.addWidget(self.conclusion_text)
        self.tabs.addTab(tab_conclusion, "🧠 Conclusion & Diagnostic du Bot")

        # Tab 2: Journal des Positions Virtuelles
        tab_trades = QWidget()
        lay_trades = QVBoxLayout(tab_trades)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(12)
        self.trades_table.setHorizontalHeaderLabels([
            "Ticket", "Stratégie", "Sens", "Lot", "Entrée", "SL", "TP", "Sortie", "PnL ($)", "Gain (%)", "Raison Sortie", "Date/Heure"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(10, QHeaderView.Stretch)
        self.trades_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: #131722;
                color: #d1d4dc;
                gridline-color: #2a2e39;
                border: none;
            }
            QHeaderView::section {
                background-color: #1e222d;
                color: #90a4ae;
                padding: 4px;
                border: 1px solid #2a2e39;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        lay_trades.addWidget(self.trades_table)
        self.tabs.addTab(tab_trades, "📋 Positions Virtuelles")

        # Tab 3: Évolution du Capital
        tab_equity = QWidget()
        lay_eq = QVBoxLayout(tab_equity)
        self.equity_text = QTextEdit()
        self.equity_text.setReadOnly(True)
        self.equity_text.setFont(QFont("Consolas", 10))
        self.equity_text.setStyleSheet("background-color: #131722; color: #80d8ff; border: none; padding: 10px;")
        lay_eq.addWidget(self.equity_text)
        self.tabs.addTab(tab_equity, "📈 Évolution du Capital")

        results_layout.addWidget(self.tabs)
        splitter.addWidget(results_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        # Initialisation du badge
        self.on_capital_changed(self.capital_spin.value())

    def _create_kpi_card(self, title: str, main_val: str, sub_val: str, color_hex: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #131722;
                border: 1px solid #2a2e39;
                border-radius: 6px;
                padding: 6px 10px;
            }
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #78909c; font-size: 10px; font-weight: bold;")
        lbl_m = QLabel(main_val)
        lbl_m.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_m.setStyleSheet(f"color: {color_hex};")
        lbl_s = QLabel(sub_val)
        lbl_s.setStyleSheet("color: #b0bec5; font-size: 10px;")

        vbox.addWidget(lbl_t)
        vbox.addWidget(lbl_m)
        vbox.addWidget(lbl_s)
        card.lbl_m = lbl_m
        card.lbl_s = lbl_s
        return card

    def populate_symbols(self):
        self.symbol_combo.clear()
        visible = []
        if self.mt5 and self.mt5.connected:
            try:
                visible = self.mt5.get_visible_symbols()
            except Exception:
                pass
        if not visible:
            visible = ["BTCUSD", "ETHUSD", "SOLUSD", "EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.symbol_combo.addItems(visible)
        # Préférer BTCUSD ou EURUSD par défaut
        idx = self.symbol_combo.findText("BTCUSD")
        if idx >= 0:
            self.symbol_combo.setCurrentIndex(idx)
        else:
            idx2 = self.symbol_combo.findText("EURUSD")
            if idx2 >= 0:
                self.symbol_combo.setCurrentIndex(idx2)

    def on_capital_changed(self, val: float):
        tier = RiskManager.get_account_tier(val)
        t_id = tier.get("tier_id", "STANDARD")
        t_name = tier.get("name", "Standard")
        if t_id == "MICRO":
            self.live_tier_badge.setText(f"🟢 TIER: {t_name.upper()} ({val:.0f} $)")
            self.live_tier_badge.setStyleSheet("color: #00e676; border: 1px solid #00e676; background: #002e18; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        elif t_id == "STANDARD":
            self.live_tier_badge.setText(f"🔵 TIER: {t_name.upper()} ({val:.0f} $)")
            self.live_tier_badge.setStyleSheet("color: #29b6f6; border: 1px solid #0288d1; background: #011d28; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        elif t_id == "GROWTH":
            self.live_tier_badge.setText(f"🟣 TIER: {t_name.upper()} ({val:.0f} $)")
            self.live_tier_badge.setStyleSheet("color: #ba68c8; border: 1px solid #7b1fa2; background: #230831; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        else:
            self.live_tier_badge.setText(f"🟡 TIER: {t_name.upper()} ({val:.0f} $)")
            self.live_tier_badge.setStyleSheet("color: #ffd54f; border: 1px solid #f57f17; background: #2f2500; padding: 6px 12px; border-radius: 4px; font-weight: bold;")

    def on_profile_changed(self):
        prof = self.profile_combo.currentData()
        if prof == "CONSERVATIVE":
            self.risk_spin.setValue(0.5)
        elif prof == "BALANCED":
            self.risk_spin.setValue(1.0)
        elif prof == "AGGRESSIVE":
            self.risk_spin.setValue(2.0)

    def on_start_backtest(self):
        symbol = self.symbol_combo.currentText().strip()
        timeframe = self.tf_combo.currentText().strip()
        num_bars = int(self.bars_combo.currentData())
        capital = float(self.capital_spin.value())
        prof = str(self.profile_combo.currentData())
        risk_pct = float(self.risk_spin.value())
        direction = str(self.dir_combo.currentData())

        if not symbol:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner un symbole valide.")
            return

        strategy = str(self.strategy_combo.currentData() or "ALL")
        fixed_lot = float(self.fixed_lot_spin.value())

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(5)
        self.status_lbl.setText("Démarrage de la simulation...")

        self.worker = BacktestWorkerThread(
            engine=self.engine,
            symbol=symbol,
            timeframe=timeframe,
            num_bars=num_bars,
            initial_capital=capital,
            risk_profile=prof,
            risk_percent=risk_pct,
            direction_filter=direction,
            strategy_filter=strategy,
            fixed_lot_size=fixed_lot,
            parent=self,
        )
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def on_worker_progress(self, pct: float, msg: str):
        self.progress_bar.setValue(int(pct * 100))
        self.status_lbl.setText(msg)

    def on_worker_failed(self, err_msg: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Échec du backtest.")
        QMessageBox.critical(self, "Erreur de simulation", f"Le backtest n'a pas pu se terminer :\n\n{err_msg}")

    def on_worker_finished(self, res: dict):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText(f"Simulation terminée ({res.get('total_bars', 0)} bougies).")

        # Mise à jour des KPI cards
        final_bal = res.get("final_balance", 0.0)
        init_cap = res.get("initial_capital", 100.0)
        net_profit = res.get("net_profit_usd", 0.0)
        net_ret = res.get("net_return_pct", 0.0)
        p_color = "#00e676" if net_profit >= 0 else "#ff5252"

        self.kpi_profit.lbl_m.setText(f"{final_bal:.2f} $")
        self.kpi_profit.lbl_m.setStyleSheet(f"color: {p_color};")
        self.kpi_profit.lbl_s.setText(f"Net: {net_profit:+.2f} $ ({net_ret:+.1f}%)")

        win_rate = res.get("win_rate_pct", 0.0)
        wins = res.get("wins_count", 0)
        losses = res.get("losses_count", 0)
        self.kpi_winrate.lbl_m.setText(f"{win_rate:.1f} %")
        self.kpi_winrate.lbl_s.setText(f"{wins} Gagnés / {losses} Perdus")

        pf = res.get("profit_factor", 1.0)
        self.kpi_pf.lbl_m.setText(f"{pf:.2f}")

        dd_pct = res.get("max_drawdown_pct", 0.0)
        dd_usd = res.get("max_drawdown_usd", 0.0)
        self.kpi_dd.lbl_m.setText(f"-{dd_pct:.1f} %")
        self.kpi_dd.lbl_s.setText(f"-{dd_usd:.2f} $")

        blocked_micro = res.get("blocked_by_micro_protection", 0)
        blocked_regime = res.get("blocked_by_regime", 0)
        total_blocked = blocked_micro + blocked_regime
        self.kpi_safety.lbl_m.setText(f"{total_blocked} Filtrés")
        self.kpi_safety.lbl_s.setText(f"{blocked_micro} micro / {blocked_regime} illiquides")

        # Mise à jour de l'onglet Conclusion
        self.conclusion_text.setPlainText(res.get("conclusion", ""))

        # Mise à jour de la table des trades
        trades = res.get("trades", [])
        self.trades_table.setRowCount(len(trades))
        for r, t in enumerate(trades):
            pnl = float(t.get("profit_usd", 0.0) or 0.0)
            col_pnl = QColor("#00e676") if pnl > 0 else (QColor("#ff5252") if pnl < 0 else QColor("#bdbdbd"))

            self.trades_table.setItem(r, 0, QTableWidgetItem(str(t.get("ticket", ""))))

            strat_tag = str(t.get("strategy", "FUSION"))
            strat_item = QTableWidgetItem(strat_tag)
            strat_item.setTextAlignment(Qt.AlignCenter)
            if "CRT" in strat_tag:
                strat_item.setForeground(QColor("#ffb74d"))
            elif "ORDER" in strat_tag:
                strat_item.setForeground(QColor("#4fc3f7"))
            elif "BOOM" in strat_tag:
                strat_item.setForeground(QColor("#ba68c8"))
            else:
                strat_item.setForeground(QColor("#80cbc4"))
            self.trades_table.setItem(r, 1, strat_item)

            act_item = QTableWidgetItem(str(t.get("action", "")))
            act_item.setForeground(QColor("#00e676") if t.get("action") == "BUY" else QColor("#ff5252"))
            act_item.setTextAlignment(Qt.AlignCenter)
            self.trades_table.setItem(r, 2, act_item)

            self.trades_table.setItem(r, 3, QTableWidgetItem(f"{t.get('lot', 0.01):.2f}"))
            self.trades_table.setItem(r, 4, QTableWidgetItem(f"{t.get('entry', 0.0):.5f}"))
            self.trades_table.setItem(r, 5, QTableWidgetItem(f"{t.get('sl', 0.0):.5f}"))
            self.trades_table.setItem(r, 6, QTableWidgetItem(f"{t.get('tp', 0.0):.5f}"))
            self.trades_table.setItem(r, 7, QTableWidgetItem(f"{t.get('exit_price', 0.0):.5f}"))

            pnl_item = QTableWidgetItem(f"{pnl:+.2f} $")
            pnl_item.setForeground(col_pnl)
            pnl_item.setTextAlignment(Qt.AlignCenter)
            self.trades_table.setItem(r, 8, pnl_item)

            pct_item = QTableWidgetItem(f"{t.get('profit_pct', 0.0):+.1f}%")
            pct_item.setForeground(col_pnl)
            pct_item.setTextAlignment(Qt.AlignCenter)
            self.trades_table.setItem(r, 9, pct_item)

            reason_item = QTableWidgetItem(str(t.get("exit_reason", "")))
            if t.get("exit_reason") == "TAKE_PROFIT":
                reason_item.setForeground(QColor("#00e676"))
            elif t.get("exit_reason") == "BREAK_EVEN":
                reason_item.setForeground(QColor("#80d8ff"))
            elif t.get("exit_reason") == "STOP_LOSS":
                reason_item.setForeground(QColor("#ff5252"))
            self.trades_table.setItem(r, 10, reason_item)

            self.trades_table.setItem(r, 11, QTableWidgetItem(str(t.get("close_time", ""))))

        # Mise à jour de la courbe d'évolution
        eq_curve = res.get("equity_curve", [])
        txt_eq = f"=== HISTORIQUE DE L'ÉVOLUTION DU CAPITAL ({len(trades)} positions) ===\n\n"
        txt_eq += f"Capital de départ : {init_cap:.2f} $\n"
        for i, pt in enumerate(eq_curve):
            b = pt.get("balance", 0.0)
            txt_eq += f"• Jalon #{i+1} [{pt.get('time')}]: Solde = {b:.2f} $ ({((b - init_cap)/init_cap*100):+.2f}%)\n"
        txt_eq += f"\nSolde d'arrivée : {final_bal:.2f} $ (Net: {net_profit:+.2f} $)\n"
        self.equity_text.setPlainText(txt_eq)

        # Basculer vers l'onglet Conclusion
        self.tabs.setCurrentIndex(0)
