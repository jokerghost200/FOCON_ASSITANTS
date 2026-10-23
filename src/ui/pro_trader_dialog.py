"""
Pro Trader Strategy Copy Dialog (Interface Graphique FOCON Strategy Copy)
========================================================================
Permet de sélectionner, configurer et activer les stratégies des 4 traders professionnels :
- Corentin Trading (Order Flow & Inducement · EURUSD)
- Benjamin Mauger (Scalping d'Ouverture · US30/NASDAQ)
- Romain Bailleul (Price Action & Fibonacci · Indices)
- Nada FX (BMC & Retest 50-61.8% · Gold/NASDAQ)

Modes disponibles :
1. Simulation (Signaux détectés et journalisés, aucun ordre MT5)
2. Copie Démo (Exécution automatique sur compte démo MT5 avec gestion du risque)
3. Personnalisée (Ajustement fin des paramètres et des règles)
"""

import logging
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea,
    QWidget, QGroupBox, QMessageBox, QTabWidget, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from src.strategies.strategy_profile_manager import StrategyProfileManager

logger = logging.getLogger(__name__)


class ProTraderCopyDialog(QDialog):
    """Fenêtre interactive de sélection et de configuration des 4 traders professionnels."""

    profile_updated = Signal(str, str)  # (active_profile_id or "", mode)

    def __init__(self, profile_manager: StrategyProfileManager, parent=None):
        super().__init__(parent)
        self.manager = profile_manager
        self.setWindowTitle("👑 FOCON — Copie de Stratégies Professionnelles (4 Traders Pros)")
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(1000, max(720, int(avail.width() * 0.85)))
            target_h = min(820, max(520, int(avail.height() * 0.85)))
            self.resize(target_w, target_h)
        else:
            self.resize(880, 680)
        self.setMinimumSize(660, 480)

        # Style sombre & haute visibilité
        self.setStyleSheet("""
            QDialog {
                background-color: #0b0e17;
                color: #f1f5f9;
            }
            QTabWidget::pane {
                background-color: #0f1422;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 6px;
            }
            QTabBar::tab {
                background-color: #161d2d;
                color: #94a3b8;
                padding: 8px 18px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #1e293b;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #ffd54f;
                border-bottom: 2px solid #ffd54f;
            }
            QTabBar::tab:hover {
                background-color: #24324a;
                color: #ffffff;
            }
            QScrollArea {
                background-color: #0f1422;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #0f1422;
            }
            QGroupBox {
                background-color: #131929;
                border: 1px solid #232f48;
                border-radius: 8px;
                margin-top: 14px;
                padding: 16px 14px 12px 14px;
                font-weight: bold;
                font-size: 13px;
                color: #60a5fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 10px;
                color: #93c5fd;
                background-color: #1a233a;
                border: 1px solid #283754;
                border-radius: 4px;
            }
            QLabel {
                color: #f1f5f9;
                font-size: 12px;
                background: transparent;
            }
            QRadioButton {
                color: #f1f5f9;
                font-size: 13px;
                font-weight: 600;
                spacing: 8px;
                background: transparent;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #64748b;
                background-color: #090d16;
            }
            QRadioButton::indicator:checked {
                background-color: #f59e0b;
                border: 3px solid #ffffff;
            }
            QRadioButton::indicator:hover {
                border-color: #93c5fd;
            }
            QCheckBox {
                color: #f1f5f9;
                font-size: 12px;
                font-weight: 500;
                spacing: 8px;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #64748b;
                background-color: #090d16;
            }
            QCheckBox::indicator:checked {
                background-color: #10b981;
                border: 2px solid #34d399;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #090d16;
                color: #ffd54f;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #60a5fa;
                background-color: #0d1320;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 18px;
                background-color: #1a233a;
                border: 1px solid #283754;
            }
            QTableWidget {
                background-color: #0d121e;
                color: #f1f5f9;
                border: 1px solid #1e293b;
                border-radius: 6px;
                gridline-color: #1e293b;
            }
            QTableWidget::item {
                background-color: #0d121e;
                color: #f1f5f9;
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #1e293b;
                color: #ffd54f;
            }
            QHeaderView::section {
                background-color: #161e30;
                color: #93c5fd;
                font-weight: bold;
                border: 1px solid #1e293b;
                padding: 6px 10px;
                font-size: 12px;
            }
            QScrollBar:vertical {
                background: #0f1422;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #283754;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3b82f6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self.selected_trader_id = self.manager.get_active_profile_id() or "nada"
        self.selected_mode = self.manager.get_active_mode() or "simulation"

        self.init_ui()
        self.load_current_state()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ── 1. HEADER BANNER ──
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a2035, stop:0.5 #202b48, stop:1 #1a2035);
                border: 1px solid #ffb300;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(12, 6, 12, 6)

        title_vbox = QVBoxLayout()
        title_lbl = QLabel("👑 FOCON — Strategy Copy (4 Traders Professionnels)")
        title_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        title_lbl.setStyleSheet("color: #ffd54f; letter-spacing: 0.5px;")

        sub_lbl = QLabel(
            "Reproduisez automatiquement les règles codées des 4 traders pros du challenge "
            "en mode Démo MT5 ou Simulation."
        )
        sub_lbl.setFont(QFont("Arial", 10))
        sub_lbl.setStyleSheet("color: #b0bec5;")
        title_vbox.addWidget(title_lbl)
        title_vbox.addWidget(sub_lbl)
        h_layout.addLayout(title_vbox)

        h_layout.addStretch()

        self.status_badge = QLabel("INACTIF")
        self.status_badge.setFont(QFont("Arial", 11, QFont.Bold))
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(180)
        self.status_badge.setMinimumHeight(44)
        self._update_status_badge()
        h_layout.addWidget(self.status_badge)

        main_layout.addWidget(header_frame)

        # ── 2. ONGLETS : CONFIGURATION & STATISTIQUES ──
        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 10, QFont.Bold))

        # --- Onglet 1 : Sélection & Paramétrage ---
        tab_config = QWidget()
        tab_cfg_lay = QVBoxLayout(tab_config)
        tab_cfg_lay.setSpacing(10)

        # A. Sélection du Trader (4 Cartes Riches)
        trader_grp = QGroupBox("1. Sélectionner le Trader Professionnel")
        trader_lay = QGridLayout(trader_grp)
        trader_lay.setSpacing(10)

        self.trader_btn_group = QButtonGroup(self)
        self.trader_cards = {}

        _TRADERS_DEF = [
            (
                "corentin",
                "🔵 Corentin Trading",
                "Order Flow · Imbalances M15",
                "EURUSD (Futures & CFD)",
                "• Analyse liquidité & imbalances M15\n• Fenêtre précise (16h00 ±30m)\n• Confirmation LTF M5/M3\n• Filtre strict anti-range",
                "#60a5fa"
            ),
            (
                "benjamin",
                "🟣 Benjamin Mauger",
                "Scalping d'Ouverture · Clusters",
                "US30, NASDAQ, Gold",
                "• Exploitation de la volatilité d'ouverture\n• Grappes de 2 à 3 ordres limites étagés\n• TP courts & contrôle strict du spread\n• Risque dégressif par trade",
                "#c084fc"
            ),
            (
                "romain",
                "🟢 Romain Bailleul",
                "Price Action · Stats · Fibonacci",
                "NASDAQ, S&P 500, Dow Jones",
                "• Pullback sur prix d'ouverture (OPR)\n• Repli Fibonacci 61.8% Golden Pocket\n• Filtre d'annonces & panique VIX\n• SL ATR, Break Even 1R & trailing",
                "#4ade80"
            ),
            (
                "nada",
                "🟡 Nada FX",
                "BMC · Structure & Fibonacci 50-61.8%",
                "Gold (XAUUSD), NASDAQ",
                "• Prise de liquidité (Sweep) session Londres\n• Cassure de structure H1 (BOS)\n• Entrée sur retest Fibo 50% - 61.8%\n• Stop Loss sur invalidation du sweep",
                "#fbbf24"
            ),
        ]

        row, col = 0, 0
        for pid, name, style, markets, details, accent_col in _TRADERS_DEF:
            card = QFrame()
            card.setObjectName(f"card_{pid}")
            card.setStyleSheet(f"""
                QFrame#card_{pid} {{
                    background-color: #171d2b;
                    border: 1px solid #2d3748;
                    border-radius: 8px;
                    padding: 8px;
                }}
                QFrame#card_{pid}:hover {{
                    border: 1px solid {accent_col};
                    background-color: #1d2436;
                }}
            """)
            c_lay = QVBoxLayout(card)
            c_lay.setSpacing(4)

            radio = QRadioButton(name)
            radio.setFont(QFont("Arial", 11, QFont.Bold))
            radio.setStyleSheet(f"color: #ffffff; font-weight: bold;")
            self.trader_btn_group.addButton(radio)
            self.trader_cards[pid] = (radio, card)
            c_lay.addWidget(radio)

            style_lbl = QLabel(f"<b>Style :</b> {style}")
            style_lbl.setStyleSheet(f"color: {accent_col}; font-size: 11px;")
            c_lay.addWidget(style_lbl)

            mkt_lbl = QLabel(f"<b>Actifs :</b> {markets}")
            mkt_lbl.setStyleSheet("color: #90caf9; font-size: 11px;")
            c_lay.addWidget(mkt_lbl)

            det_lbl = QLabel(details)
            det_lbl.setStyleSheet("color: #a0aec0; font-size: 11px; margin-top: 4px;")
            det_lbl.setWordWrap(True)
            c_lay.addWidget(det_lbl)

            trader_lay.addWidget(card, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        self.trader_btn_group.buttonClicked.connect(self.on_trader_selected)
        tab_cfg_lay.addWidget(trader_grp)

        # B. Sélection du Mode d'Opération (3 Modes)
        mode_grp = QGroupBox("2. Choisir le Mode d'Opération")
        mode_lay = QHBoxLayout(mode_grp)
        mode_lay.setSpacing(12)

        self.mode_btn_group = QButtonGroup(self)

        self.mode_sim_radio = QRadioButton("🧪 Mode 1 — Simulation (Sans ordre MT5)")
        self.mode_sim_radio.setToolTip("Détecte les signaux, les journalise et teste l'exactitude des règles sans placer d'ordres réels.")
        self.mode_btn_group.addButton(self.mode_sim_radio)
        mode_lay.addWidget(self.mode_sim_radio)

        self.mode_copy_radio = QRadioButton("🚀 Mode 2 — Copie de Stratégie (Démo MT5)")
        self.mode_copy_radio.setToolTip("Exécute fidèlement les règles sur votre compte démo MT5 avec validation du RiskManager.")
        self.mode_btn_group.addButton(self.mode_copy_radio)
        mode_lay.addWidget(self.mode_copy_radio)

        self.mode_custom_radio = QRadioButton("⚙️ Mode 3 — Stratégie Personnalisée")
        self.mode_custom_radio.setToolTip("Permet d'ajuster individuellement les paramètres ci-dessous.")
        self.mode_btn_group.addButton(self.mode_custom_radio)
        mode_lay.addWidget(self.mode_custom_radio)

        self.mode_btn_group.buttonClicked.connect(self.on_mode_changed)
        tab_cfg_lay.addWidget(mode_grp)

        # C. Paramètres Détaillés Ajustables
        self.params_grp = QGroupBox("3. Paramètres de la Stratégie Sélectionnée")
        self.params_lay = QGridLayout(self.params_grp)
        self.params_lay.setSpacing(8)
        self.build_parameters_panel()
        tab_cfg_lay.addWidget(self.params_grp)

        # --- Onglet 1 : Sélection & Paramétrage (Déroulant / Réactif) ---
        tab_config.setStyleSheet("background-color: #0f1422; color: #f1f5f9;")
        scroll_config = QScrollArea()
        scroll_config.setWidgetResizable(True)
        scroll_config.setFrameShape(QFrame.NoFrame)
        scroll_config.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_config.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_config.setStyleSheet("background-color: #0f1422; border: none;")
        scroll_config.viewport().setStyleSheet("background-color: #0f1422;")
        scroll_config.setWidget(tab_config)
        tabs.addTab(scroll_config, "⚙️ Configuration de la Stratégie")

        # --- Onglet 2 : Journal des Performances & Comparatif (Déroulant / Réactif) ---
        tab_stats = QWidget()
        tab_stats.setStyleSheet("background-color: #0f1422; color: #f1f5f9;")
        tab_stats_lay = QVBoxLayout(tab_stats)
        tab_stats_lay.setSpacing(8)

        stats_hint = QLabel("📊 Suivi en temps réel des signaux générés et des trades clôturés par profil :")
        stats_hint.setFont(QFont("Arial", 10, QFont.Bold))
        stats_hint.setStyleSheet("color: #93c5fd;")
        tab_stats_lay.addWidget(stats_hint)

        self.stats_table = QTableWidget(4, 7)
        self.stats_table.setHorizontalHeaderLabels([
            "Trader Pro", "Style", "Statut", "Signaux", "Trades Démo", "Winrate", "PnL Total ($)"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tab_stats_lay.addWidget(self.stats_table)

        self.refresh_stats_btn = QPushButton("🔄 Rafraîchir les Statistiques")
        self.refresh_stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: #ffffff;
            }
        """)
        self.refresh_stats_btn.clicked.connect(self.refresh_stats_table)
        tab_stats_lay.addWidget(self.refresh_stats_btn)

        scroll_stats = QScrollArea()
        scroll_stats.setWidgetResizable(True)
        scroll_stats.setFrameShape(QFrame.NoFrame)
        scroll_stats.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_stats.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_stats.setStyleSheet("background-color: #0f1422; border: none;")
        scroll_stats.viewport().setStyleSheet("background-color: #0f1422;")
        scroll_stats.setWidget(tab_stats)
        tabs.addTab(scroll_stats, "📈 Performance & Comparatif")

        main_layout.addWidget(tabs)

        # ── 3. ACTIONS DE BAS DE PAGE ──
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(10)

        self.activate_btn = QPushButton("🚀 ACTIVER CETTE STRATÉGIE")
        self.activate_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.activate_btn.setFixedHeight(38)
        self.activate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2e7d32, stop:1 #43a047);
                color: white;
                border-radius: 6px;
                padding: 6px 16px;
                border: 1px solid #81c784;
            }
            QPushButton:hover {
                background: #4caf50;
            }
        """)
        self.activate_btn.clicked.connect(self.on_activate_clicked)
        btn_bar.addWidget(self.activate_btn)

        self.deactivate_btn = QPushButton("🛑 DÉSACTIVER LA COPIE")
        self.deactivate_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.deactivate_btn.setFixedHeight(38)
        self.deactivate_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                border-radius: 6px;
                padding: 6px 16px;
                border: 1px solid #ef5350;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)
        self.deactivate_btn.clicked.connect(self.on_deactivate_clicked)
        btn_bar.addWidget(self.deactivate_btn)

        btn_bar.addStretch()

        self.close_btn = QPushButton("Fermer")
        self.close_btn.setFixedHeight(38)
        self.close_btn.setStyleSheet("background-color: #374151; color: white; border-radius: 6px; padding: 6px 16px;")
        self.close_btn.clicked.connect(self.accept)
        btn_bar.addWidget(self.close_btn)

        main_layout.addLayout(btn_bar)

    def build_parameters_panel(self):
        """Construit dynamiquement les champs d'ajustement selon le profil."""
        # Nettoyer l'ancien layout
        while self.params_lay.count():
            item = self.params_lay.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        pid = self.selected_trader_id
        strat = self.manager.strategies.get(pid)
        cfg = strat.config if strat else {}

        row = 0
        self.param_widgets = {}

        self.params_lay.setColumnStretch(0, 0)
        self.params_lay.setColumnStretch(1, 0)
        self.params_lay.setColumnStretch(2, 1)

        if pid == "corentin":
            # Paramètres Corentin
            self.params_lay.addWidget(QLabel("Ratio Range / ATR (Filtre Range) :"), row, 0)
            sp_range = QDoubleSpinBox()
            sp_range.setRange(0.2, 3.0); sp_range.setSingleStep(0.1); sp_range.setValue(float(cfg.get("range_atr_ratio", 0.8)))
            self.params_lay.addWidget(sp_range, row, 1)
            self.param_widgets["range_atr_ratio"] = sp_range

            row += 1
            self.params_lay.addWidget(QLabel("Ratio R:R minimum :"), row, 0)
            sp_rr = QDoubleSpinBox()
            sp_rr.setRange(1.0, 5.0); sp_rr.setSingleStep(0.2); sp_rr.setValue(float(cfg.get("min_rr", 2.0)))
            self.params_lay.addWidget(sp_rr, row, 1)
            self.param_widgets["min_rr"] = sp_rr

            row += 1
            cb_flow = QCheckBox("Activer le filtre de flux d'ordres (Order Flow)")
            cb_flow.setChecked(bool(cfg.get("use_flow", True)))
            self.params_lay.addWidget(cb_flow, row, 0, 1, 2)
            self.param_widgets["use_flow"] = cb_flow

        elif pid == "benjamin":
            # Paramètres Benjamin
            self.params_lay.addWidget(QLabel("Fenêtre d'Ouverture (minutes) :"), row, 0)
            sp_win = QSpinBox()
            sp_win.setRange(2, 60); sp_win.setValue(int(cfg.get("open_window_min", 15)))
            self.params_lay.addWidget(sp_win, row, 1)
            self.param_widgets["open_window_min"] = sp_win

            row += 1
            self.params_lay.addWidget(QLabel("Spread Maximum toléré (pts) :"), row, 0)
            sp_spr = QDoubleSpinBox()
            sp_spr.setRange(0.5, 20.0); sp_spr.setSingleStep(0.5); sp_spr.setValue(float(cfg.get("max_spread", 2.5)))
            self.params_lay.addWidget(sp_spr, row, 1)
            self.param_widgets["max_spread"] = sp_spr

            row += 1
            self.params_lay.addWidget(QLabel("Nombre d'ordres par grappe (Cluster) :"), row, 0)
            sp_n = QSpinBox()
            sp_n.setRange(1, 5); sp_n.setValue(int(cfg.get("cluster_n", 3)))
            self.params_lay.addWidget(sp_n, row, 1)
            self.param_widgets["cluster_n"] = sp_n

            row += 1
            self.params_lay.addWidget(QLabel("Espacement entre ordres (pts) :"), row, 0)
            sp_step = QDoubleSpinBox()
            sp_step.setRange(1.0, 50.0); sp_step.setValue(float(cfg.get("step_pts", 5.0)))
            self.params_lay.addWidget(sp_step, row, 1)
            self.param_widgets["step_pts"] = sp_step

            row += 1
            self.params_lay.addWidget(QLabel("Take Profit Scalp (pts) :"), row, 0)
            sp_tp = QDoubleSpinBox()
            sp_tp.setRange(1.0, 100.0); sp_tp.setValue(float(cfg.get("tp_pts", 8.0)))
            self.params_lay.addWidget(sp_tp, row, 1)
            self.param_widgets["tp_pts"] = sp_tp

        elif pid == "romain":
            # Paramètres Romain
            self.params_lay.addWidget(QLabel("Multiplicateur Stop Loss ATR :"), row, 0)
            sp_atr = QDoubleSpinBox()
            sp_atr.setRange(0.5, 4.0); sp_atr.setSingleStep(0.1); sp_atr.setValue(float(cfg.get("atr_mult", 1.5)))
            self.params_lay.addWidget(sp_atr, row, 1)
            self.param_widgets["atr_mult"] = sp_atr

            row += 1
            self.params_lay.addWidget(QLabel("Seuil Break Even (en multiple de R) :"), row, 0)
            sp_be = QDoubleSpinBox()
            sp_be.setRange(0.5, 3.0); sp_be.setSingleStep(0.2); sp_be.setValue(float(cfg.get("be_r", 1.0)))
            self.params_lay.addWidget(sp_be, row, 1)
            self.param_widgets["be_r"] = sp_be

            row += 1
            self.params_lay.addWidget(QLabel("Filtre Annonces Économiques (minutes) :"), row, 0)
            sp_news = QSpinBox()
            sp_news.setRange(5, 120); sp_news.setValue(int(cfg.get("news_minutes", 30)))
            self.params_lay.addWidget(sp_news, row, 1)
            self.param_widgets["news_minutes"] = sp_news

            row += 1
            self.params_lay.addWidget(QLabel("Niveau Fibonacci Golden Pocket :"), row, 0)
            sp_fib = QDoubleSpinBox()
            sp_fib.setRange(0.50, 0.786); sp_fib.setDecimals(3); sp_fib.setValue(float(cfg.get("fib_level", 0.618)))
            self.params_lay.addWidget(sp_fib, row, 1)
            self.param_widgets["fib_level"] = sp_fib

        elif pid == "nada":
            # Paramètres Nada FX
            self.params_lay.addWidget(QLabel("Ratio R:R minimum garanti :"), row, 0)
            sp_rr = QDoubleSpinBox()
            sp_rr.setRange(1.5, 5.0); sp_rr.setSingleStep(0.2); sp_rr.setValue(float(cfg.get("min_rr", 2.0)))
            self.params_lay.addWidget(sp_rr, row, 1)
            self.param_widgets["min_rr"] = sp_rr

            row += 1
            self.params_lay.addWidget(QLabel("Zone Fibonacci Retest Min :"), row, 0)
            sp_fmin = QDoubleSpinBox()
            sp_fmin.setRange(0.382, 0.618); sp_fmin.setDecimals(3); sp_fmin.setValue(float(cfg.get("fib_zone_min", 0.50)))
            self.params_lay.addWidget(sp_fmin, row, 1)
            self.param_widgets["fib_zone_min"] = sp_fmin

            row += 1
            self.params_lay.addWidget(QLabel("Zone Fibonacci Retest Max :"), row, 0)
            sp_fmax = QDoubleSpinBox()
            sp_fmax.setRange(0.50, 0.786); sp_fmax.setDecimals(3); sp_fmax.setValue(float(cfg.get("fib_zone_max", 0.618)))
            self.params_lay.addWidget(sp_fmax, row, 1)
            self.param_widgets["fib_zone_max"] = sp_fmax

        # Activer ou désactiver les widgets selon le mode choisi
        is_custom = (self.selected_mode == "custom")
        for w in self.param_widgets.values():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setFixedWidth(130)
            w.setEnabled(is_custom)

    def load_current_state(self):
        """Met à jour les contrôles d'après l'état actuel de StrategyProfileManager."""
        active_pid = self.manager.get_active_profile_id()
        active_mode = self.manager.get_active_mode()

        if active_pid in self.trader_cards:
            self.trader_cards[active_pid][0].setChecked(True)
            self.selected_trader_id = active_pid
        elif "nada" in self.trader_cards:
            self.trader_cards["nada"][0].setChecked(True)
            self.selected_trader_id = "nada"

        if active_mode == "copy":
            self.mode_copy_radio.setChecked(True)
        elif active_mode == "custom":
            self.mode_custom_radio.setChecked(True)
        else:
            self.mode_sim_radio.setChecked(True)
        self.selected_mode = active_mode

        self.build_parameters_panel()
        self._update_status_badge()
        self.refresh_stats_table()

    def on_trader_selected(self, button):
        """Gère le clic sur une carte de trader."""
        for pid, (radio, card) in self.trader_cards.items():
            if radio == button:
                self.selected_trader_id = pid
                break
        self.build_parameters_panel()

    def on_mode_changed(self, button):
        """Gère le basculement entre les modes Simulation, Copie Démo et Personnalisé."""
        if self.mode_copy_radio.isChecked():
            self.selected_mode = "copy"
        elif self.mode_custom_radio.isChecked():
            self.selected_mode = "custom"
        else:
            self.selected_mode = "simulation"

        is_custom = (self.selected_mode == "custom")
        for w in self.param_widgets.values():
            w.setEnabled(is_custom)

    def on_activate_clicked(self):
        """Active la stratégie sélectionnée."""
        custom_cfg = {}
        if self.selected_mode == "custom":
            for k, w in self.param_widgets.items():
                if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    custom_cfg[k] = w.value()
                elif isinstance(w, QCheckBox):
                    custom_cfg[k] = w.isChecked()

        self.manager.set_active_profile(
            profile_id=self.selected_trader_id,
            mode=self.selected_mode,
            custom_cfg=custom_cfg if custom_cfg else None,
        )

        self._update_status_badge()
        self.refresh_stats_table()
        self.profile_updated.emit(self.selected_trader_id, self.selected_mode)

        strat_name = self.manager.strategies[self.selected_trader_id].name
        mode_desc = "SIMULATION (0 risque, signaux journalisés)" if self.selected_mode == "simulation" else (
            "COPIE DÉMO (Exécution automatique MT5)" if self.selected_mode == "copy" else "PERSONNALISÉE"
        )
        QMessageBox.information(
            self,
            "👑 Stratégie Pro Activée",
            f"La stratégie de <b>{strat_name}</b> est maintenant active en <b>{mode_desc}</b>.\n\n"
            f"Le bot FOCON appliquera ses règles d'entrée, de confirmation et de sortie."
        )

    def on_deactivate_clicked(self):
        """Désactive toute copie de trader pro pour revenir au fonctionnement standard."""
        self.manager.set_active_profile(None)
        self._update_status_badge()
        self.refresh_stats_table()
        self.profile_updated.emit("", "simulation")

        QMessageBox.information(
            self,
            "🛑 Copie Désactivée",
            "La copie des traders professionnels a été désactivée.\n"
            "Le bot FOCON revient à son mode d'analyse multi-stratégies standard (CRT + SMC)."
        )

    def _update_status_badge(self):
        """Met à jour le badge d'état visuel en haut à droite."""
        active_pid = self.manager.get_active_profile_id()
        mode = self.manager.get_active_mode().upper()

        if active_pid:
            strat_name = self.manager.strategies[active_pid].name.split()[0].upper()
            if mode == "SIMULATION":
                self.status_badge.setText(f"🧪 {strat_name}\n[SIMU]")
                self.status_badge.setStyleSheet("""
                    background-color: #0d47a1; color: #90caf9;
                    border: 1px solid #42a5f5; border-radius: 6px; font-weight: bold;
                """)
            elif mode == "COPY":
                self.status_badge.setText(f"👑 {strat_name}\n[DÉMO]")
                self.status_badge.setStyleSheet("""
                    background-color: #1b5e20; color: #a5d6a7;
                    border: 1px solid #4caf50; border-radius: 6px; font-weight: bold;
                """)
            else:
                self.status_badge.setText(f"⚙️ {strat_name}\n[CUSTOM]")
                self.status_badge.setStyleSheet("""
                    background-color: #e65100; color: #ffcc80;
                    border: 1px solid #ff9800; border-radius: 6px; font-weight: bold;
                """)
        else:
            self.status_badge.setText("INACTIF\n(Mode Standard)")
            self.status_badge.setStyleSheet("""
                background-color: #263238; color: #90a4ae;
                border: 1px solid #546e7a; border-radius: 6px; font-weight: bold;
            """)

    def refresh_stats_table(self):
        """Remplit le tableau comparatif des performances de chaque trader."""
        profiles = self.manager.get_all_profiles_info()
        for row, p in enumerate(profiles):
            name_item = QTableWidgetItem(p["name"])
            name_item.setFont(QFont("Arial", 10, QFont.Bold))
            name_item.setForeground(QColor("#f1f5f9"))

            style_item = QTableWidgetItem(p["style"])
            style_item.setForeground(QColor("#94a3b8"))

            statut = "ACTIF (" + (p["active_mode"].upper() if p["active_mode"] else "") + ")" if p["is_active"] else "Inactif"
            statut_item = QTableWidgetItem(statut)
            statut_item.setTextAlignment(Qt.AlignCenter)
            if p["is_active"]:
                statut_item.setForeground(QColor("#10b981"))
                statut_item.setFont(QFont("Arial", 10, QFont.Bold))
            else:
                statut_item.setForeground(QColor("#64748b"))

            st = p["stats"]
            sig_item = QTableWidgetItem(str(st["signals_count"]))
            sig_item.setForeground(QColor("#e2e8f0"))
            sig_item.setTextAlignment(Qt.AlignCenter)

            trades_item = QTableWidgetItem(str(st["trades"]))
            trades_item.setForeground(QColor("#e2e8f0"))
            trades_item.setTextAlignment(Qt.AlignCenter)

            wr_item = QTableWidgetItem(f"{st['win_rate_pct']:.1f} %")
            wr_item.setForeground(QColor("#60a5fa"))
            wr_item.setFont(QFont("Arial", 10, QFont.Bold))
            wr_item.setTextAlignment(Qt.AlignCenter)

            pnl = st["total_pnl"]
            pnl_item = QTableWidgetItem(f"{pnl:+.2f} $")
            pnl_item.setFont(QFont("Arial", 10, QFont.Bold))
            pnl_item.setTextAlignment(Qt.AlignCenter)
            if pnl > 0:
                pnl_item.setForeground(QColor("#10b981"))
            elif pnl < 0:
                pnl_item.setForeground(QColor("#ef4444"))
            else:
                pnl_item.setForeground(QColor("#94a3b8"))

            self.stats_table.setItem(row, 0, name_item)
            self.stats_table.setItem(row, 1, style_item)
            self.stats_table.setItem(row, 2, statut_item)
            self.stats_table.setItem(row, 3, sig_item)
            self.stats_table.setItem(row, 4, trades_item)
            self.stats_table.setItem(row, 5, wr_item)
            self.stats_table.setItem(row, 6, pnl_item)
