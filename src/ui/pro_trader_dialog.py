"""
Pro Trader Strategy Copy Dialog (Interface Graphique FOCON Strategy Copy)
========================================================================
Permet de sélectionner, configurer et activer les stratégies des 5 traders professionnels :
- Corentin Trading (Order Flow & Inducement · EURUSD)
- Benjamin Mauger (Scalping d'Ouverture · US30/NASDAQ)
- Romain Bailleul (Price Action & Fibonacci · Indices)
- Nada FX (BMC & Retest 50-61.8% · Gold/NASDAQ)
- George Soros (Global Macro & Réflexivité · Forex & Indices)

Modes disponibles :
1. Simulation (Signaux détectés et journalisés, aucun ordre MT5)
2. Copie Démo (Exécution automatique sur compte démo MT5 avec gestion du risque)
3. Personnalisée (Ajustement fin des paramètres et des règles)
"""

import os
import logging
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea,
    QWidget, QGroupBox, QMessageBox, QTabWidget, QAbstractItemView, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QCursor, QPixmap, QPainter, QPen

from src.strategies.strategy_profile_manager import StrategyProfileManager

logger = logging.getLogger(__name__)


def _ensure_radio_assets():
    """Génère des icônes radio vectorielles parfaitement circulaires et antialiasées."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    off_path = os.path.join(assets_dir, "radio_off.png")
    on_path = os.path.join(assets_dir, "radio_on.png")

    if not (os.path.exists(off_path) and os.path.exists(on_path)):
        for is_on, path, col in [(False, off_path, "#64748b"), (True, on_path, "#f59e0b")]:
            pix = QPixmap(24, 24)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(col), 2)
            p.setPen(pen)
            p.setBrush(QColor("#0b111e"))
            p.drawEllipse(2, 2, 20, 20)
            if is_on:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(col))
                p.drawEllipse(6, 6, 12, 12)
            p.end()
            pix.save(path)

    return off_path.replace("\\", "/"), on_path.replace("\\", "/")


class ClickableCard(QFrame):
    """Carte cliquable stylée avec curseur pointeur et signal de clic."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ProTraderCopyDialog(QDialog):
    """Fenêtre interactive, responsive et haute fidélité pour la copie des Traders Pros."""

    profile_updated = Signal(str, str)  # (active_profile_id or "", mode)

    def __init__(self, profile_manager: StrategyProfileManager, parent=None):
        super().__init__(parent)
        self.manager = profile_manager
        self.setWindowTitle("👑 FOCON — Copie de Stratégies Professionnelles (5 Traders Pros)")

        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(1080, max(820, int(avail.width() * 0.85)))
            target_h = min(880, max(600, int(avail.height() * 0.88)))
            self.resize(target_w, target_h)
        else:
            self.resize(980, 780)
        self.setMinimumSize(780, 560)

        off_icon_path, on_icon_path = _ensure_radio_assets()

        # Style global moderne, responsive et exempt de bugs de sous-contrôles
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #0b0e17;
                color: #f1f5f9;
            }}
            QTabWidget::pane {{
                background-color: #0f1422;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 4px;
            }}
            QTabBar::tab {{
                background-color: #141b2c;
                color: #94a3b8;
                padding: 10px 22px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #1e293b;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: #1e293b;
                color: #ffd54f;
                border-bottom: 2px solid #ffd54f;
            }}
            QTabBar::tab:hover {{
                background-color: #24324a;
                color: #ffffff;
            }}
            QScrollArea {{
                background-color: #0f1422;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: #0f1422;
            }}
            QGroupBox {{
                background-color: #111726;
                border: 1px solid #1f2a3f;
                border-radius: 8px;
                margin-top: 14px;
                padding: 14px 12px 12px 12px;
                font-weight: bold;
                font-size: 13px;
                color: #60a5fa;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 10px;
                color: #93c5fd;
                background-color: #1a233a;
                border: 1px solid #283754;
                border-radius: 4px;
            }}
            QLabel {{
                color: #f1f5f9;
                font-size: 12px;
                background: transparent;
            }}

            /* Radio Buttons avec images vectorielles parfaitement circulaires */
            QRadioButton {{
                color: #f1f5f9;
                font-size: 13px;
                font-weight: 600;
                spacing: 8px;
                background: transparent;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
            QRadioButton::indicator:unchecked {{
                image: url("{off_icon_path}");
            }}
            QRadioButton::indicator:checked {{
                image: url("{on_icon_path}");
            }}

            /* CheckBoxes */
            QCheckBox {{
                color: #f1f5f9;
                font-size: 12px;
                font-weight: 500;
                spacing: 8px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                min-width: 18px;
                max-width: 18px;
                min-height: 18px;
                max-height: 18px;
                border-radius: 4px;
                border: 2px solid #64748b;
                background-color: #090d16;
            }}
            QCheckBox::indicator:checked {{
                background-color: #10b981;
                border: 2px solid #34d399;
            }}

            /* SpinBoxes & DoubleSpinBoxes avec subcontrols parfaitement ancrés à droite */
            QSpinBox, QDoubleSpinBox {{
                background-color: #0b111e;
                color: #ffd54f;
                border: 1px solid #2d3b55;
                border-radius: 6px;
                padding: 4px 8px;
                padding-right: 26px;
                font-weight: bold;
                font-size: 13px;
                min-height: 24px;
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border: 1px solid #3b82f6;
                background-color: #0e1626;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid #60a5fa;
                background-color: #0e1626;
            }}
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: #0d121c;
                color: #93c5fd;
                border: 1px solid #1a2334;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                height: 13px;
                border-left: 1px solid #2d3b55;
                border-bottom: 1px solid #1e293b;
                background-color: #162032;
                border-top-right-radius: 5px;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
                background-color: #2563eb;
            }}
            QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled {{
                background-color: #0d121c;
                border-left: 1px solid #1a2334;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid #93c5fd;
            }}
            QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
                border-bottom: 4px solid #475569;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                height: 13px;
                border-left: 1px solid #2d3b55;
                background-color: #162032;
                border-bottom-right-radius: 5px;
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #2563eb;
            }}
            QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {{
                background-color: #0d121c;
                border-left: 1px solid #1a2334;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #93c5fd;
            }}
            QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
                border-top: 4px solid #475569;
            }}

            /* Table des Statistiques */
            QTableWidget {{
                background-color: #0d121e;
                color: #f1f5f9;
                border: 1px solid #1e293b;
                border-radius: 6px;
                gridline-color: #1e293b;
            }}
            QTableWidget::item {{
                background-color: #0d121e;
                color: #f1f5f9;
                padding: 8px;
            }}
            QTableWidget::item:selected {{
                background-color: #1e293b;
                color: #ffd54f;
            }}
            QHeaderView::section {{
                background-color: #161e30;
                color: #93c5fd;
                font-weight: bold;
                border: 1px solid #1e293b;
                padding: 6px 10px;
                font-size: 12px;
            }}
            QScrollBar:vertical {{
                background: #0f1422;
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #283754;
                min-height: 24px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #3b82f6;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        self.selected_trader_id = self.manager.get_active_profile_id() or "nada"
        self.selected_mode = self.manager.get_active_mode() or "simulation"

        self.trader_cards = {}   # pid -> (radio, card_frame, accent_col)
        self.mode_cards = {}     # mode_id -> (radio, card_frame, accent_col)
        self.param_widgets = {}  # key -> widget
        self.params_container = None

        self.init_ui()
        self.load_current_state()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # ── 1. HEADER BANNER ──
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a2035, stop:0.5 #202b48, stop:1 #1a2035);
                border: 1px solid #ffb300;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(12, 6, 12, 6)

        title_vbox = QVBoxLayout()
        title_lbl = QLabel("👑 FOCON — Strategy Copy (5 Traders Professionnels)")
        title_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        title_lbl.setStyleSheet("color: #ffd54f; letter-spacing: 0.5px;")

        sub_lbl = QLabel(
            "Reproduisez automatiquement les règles codées des 5 traders pros du challenge "
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

        # --- Onglet 1 : Sélection & Paramétrage (Déroulant & Responsive) ---
        tab_config = QWidget()
        tab_config.setStyleSheet("background-color: #0f1422; color: #f1f5f9;")
        tab_cfg_lay = QVBoxLayout(tab_config)
        tab_cfg_lay.setSpacing(12)
        tab_cfg_lay.setContentsMargins(8, 8, 8, 8)

        # A. Sélection du Trader (5 Cartes Équilibrées : Ligne 1 = 3 cartes, Ligne 2 = 2 cartes)
        trader_grp = QGroupBox("1. Sélectionner le Trader Professionnel")
        trader_lay = QGridLayout(trader_grp)
        trader_lay.setSpacing(8)

        self.trader_btn_group = QButtonGroup(self)

        _TRADERS_DEF = [
            (
                "corentin",
                "🔵 Corentin Trading",
                "Order Flow · Inducement · CRT",
                "EURUSD, GBPUSD",
                "• Chasse de liquidité (Inducement / FVG)\n• Confirmation au carnet (Order Flow / Tape) · R:R 2:1+",
                "#60a5fa"
            ),
            (
                "benjamin",
                "🟣 Benjamin Mauger",
                "Scalping Impulsif · 15h30 US Open",
                "Dow Jones (US30), NASDAQ",
                "• Grappe d'ordres échelonnés (3 ordres)\n• Explosion de volatilité 15h30 · TP rapide (8-10 pts)",
                "#c084fc"
            ),
            (
                "romain",
                "🟢 Romain Bailleul",
                "Price Action · Stats · Fibonacci",
                "NASDAQ, S&P 500, Dow Jones",
                "• Pullback sur prix d'ouverture (OPR)\n• Repli Fibonacci 61.8% Golden Pocket · Filtre d'annonces",
                "#4ade80"
            ),
            (
                "nada",
                "🟡 Nada FX",
                "BMC · Structure & Fibonacci 50-61.8%",
                "Gold (XAUUSD), NASDAQ",
                "• Prise de liquidité (Sweep) session Londres · BOS H1\n• Retest Fibo 50% - 61.8% · Stop sur invalidation",
                "#fbbf24"
            ),
            (
                "soros",
                "🔴 George Soros",
                "Global Macro · Réflexivité & Asymétrie",
                "GBPUSD, EURUSD, USDJPY, Gold, US500, US30",
                "• Détection des bulles réflexives & climax ATR/RSI vs EMA200\n• Rupture de structure (MSS/BOS) · R:R asymétrique (2.5+)",
                "#f87171"
            ),
        ]

        # Disposition harmonieuse : 3 cartes en ligne 0, 2 cartes en ligne 1
        grid_positions = [
            (0, 0, 1, 2),  # Corentin (cols 0-1)
            (0, 2, 1, 2),  # Benjamin (cols 2-3)
            (0, 4, 1, 2),  # Romain   (cols 4-5)
            (1, 0, 1, 3),  # Nada     (cols 0-2)
            (1, 3, 1, 3),  # Soros    (cols 3-5)
        ]

        for (pid, name, style, markets, details, accent_col), (r, c, rspan, cspan) in zip(_TRADERS_DEF, grid_positions):
            card = ClickableCard()
            card.setObjectName(f"card_{pid}")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(10, 8, 10, 8)
            c_lay.setSpacing(3)

            radio = QRadioButton(name)
            radio.setFont(QFont("Arial", 11, QFont.Bold))
            radio.setStyleSheet("color: #ffffff; font-weight: bold;")
            self.trader_btn_group.addButton(radio)
            self.trader_cards[pid] = (radio, card, accent_col)
            c_lay.addWidget(radio)

            style_lbl = QLabel(f"<b>Style :</b> {style}")
            style_lbl.setStyleSheet(f"color: {accent_col}; font-size: 11px;")
            c_lay.addWidget(style_lbl)

            mkt_lbl = QLabel(f"<b>Actifs :</b> {markets}")
            mkt_lbl.setStyleSheet("color: #90caf9; font-size: 10px;")
            c_lay.addWidget(mkt_lbl)

            det_lbl = QLabel(details)
            det_lbl.setStyleSheet("color: #a0aec0; font-size: 10px; margin-top: 2px;")
            det_lbl.setWordWrap(True)
            c_lay.addWidget(det_lbl)

            card.clicked.connect(lambda p=pid: self.on_trader_card_clicked(p))
            trader_lay.addWidget(card, r, c, rspan, cspan)

        self.trader_btn_group.buttonClicked.connect(self.on_trader_selected)
        tab_cfg_lay.addWidget(trader_grp)

        # B. Sélection du Mode d'Opération (3 Cartes Responsives & Cliquables)
        mode_grp = QGroupBox("2. Choisir le Mode d'Opération")
        mode_lay = QGridLayout(mode_grp)
        mode_lay.setSpacing(10)

        self.mode_btn_group = QButtonGroup(self)

        _MODES_DEF = [
            (
                "simulation",
                "🧪",
                "Mode 1 — Simulation",
                "0 Risque · Signaux & Journalisation sans ordre MT5",
                "#3b82f6"
            ),
            (
                "copy",
                "🚀",
                "Mode 2 — Copie de Stratégie",
                "Démo MT5 · Exécution fidèle aux règles du trader pro",
                "#10b981"
            ),
            (
                "custom",
                "⚙️",
                "Mode 3 — Personnalisée",
                "Déverrouillage · Ajustement libre de chaque paramètre",
                "#f59e0b"
            ),
        ]

        for m_col, (m_id, m_icon, m_title, m_desc, m_accent) in enumerate(_MODES_DEF):
            m_card = ClickableCard()
            m_card.setObjectName(f"mode_card_{m_id}")
            m_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            m_card_lay = QVBoxLayout(m_card)
            m_card_lay.setContentsMargins(12, 8, 12, 8)
            m_card_lay.setSpacing(4)

            m_radio = QRadioButton(f"{m_icon}  {m_title}")
            m_radio.setFont(QFont("Arial", 11, QFont.Bold))
            m_radio.setStyleSheet("color: #ffffff; font-weight: bold;")
            self.mode_btn_group.addButton(m_radio)
            self.mode_cards[m_id] = (m_radio, m_card, m_accent)
            m_card_lay.addWidget(m_radio)

            m_desc_lbl = QLabel(m_desc)
            m_desc_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
            m_desc_lbl.setWordWrap(True)
            m_card_lay.addWidget(m_desc_lbl)

            m_card.clicked.connect(lambda mid=m_id: self.on_mode_card_clicked(mid))
            mode_lay.addWidget(m_card, 0, m_col)

        self.mode_sim_radio = self.mode_cards["simulation"][0]
        self.mode_copy_radio = self.mode_cards["copy"][0]
        self.mode_custom_radio = self.mode_cards["custom"][0]

        self.mode_btn_group.buttonClicked.connect(self.on_mode_radio_clicked)
        tab_cfg_lay.addWidget(mode_grp)

        # C. Paramètres Détaillés Ajustables
        self.params_grp = QGroupBox("3. Paramètres de la Stratégie Sélectionnée")
        self.params_vbox = QVBoxLayout(self.params_grp)
        self.params_vbox.setSpacing(8)
        self.params_vbox.setContentsMargins(10, 10, 10, 10)
        tab_cfg_lay.addWidget(self.params_grp)

        # Déroulant fluide sans scrollbar horizontale
        scroll_config = QScrollArea()
        scroll_config.setWidgetResizable(True)
        scroll_config.setFrameShape(QFrame.NoFrame)
        scroll_config.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_config.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_config.setStyleSheet("background-color: #0f1422; border: none;")
        scroll_config.viewport().setStyleSheet("background-color: #0f1422;")
        scroll_config.setWidget(tab_config)
        tabs.addTab(scroll_config, "⚙️ Configuration de la Stratégie")

        # --- Onglet 2 : Journal des Performances & Comparatif ---
        tab_stats = QWidget()
        tab_stats.setStyleSheet("background-color: #0f1422; color: #f1f5f9;")
        tab_stats_lay = QVBoxLayout(tab_stats)
        tab_stats_lay.setSpacing(8)

        stats_hint = QLabel("📊 Suivi en temps réel des signaux générés et des trades clôturés par profil :")
        stats_hint.setFont(QFont("Arial", 10, QFont.Bold))
        stats_hint.setStyleSheet("color: #93c5fd;")
        tab_stats_lay.addWidget(stats_hint)

        self.stats_table = QTableWidget(5, 7)
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
            }
        """)
        self.refresh_stats_btn.clicked.connect(self.refresh_stats_table)
        tab_stats_lay.addWidget(self.refresh_stats_btn)

        tabs.addTab(tab_stats, "📊 Performances & Comparatif")
        main_layout.addWidget(tabs)

        # ── 3. BOUTONS D'ACTION EN BAS ──
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

    def _create_param_row(self, title: str, subtitle: str, widget: QWidget) -> QFrame:
        """Crée une ligne de paramètre sous forme de carte élégante et responsive."""
        row_card = QFrame()
        row_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_card.setStyleSheet("""
            QFrame {
                background-color: #131a29;
                border: 1px solid #233148;
                border-radius: 6px;
            }
            QFrame:hover {
                border: 1px solid #3b82f6;
                background-color: #172134;
            }
        """)
        r_lay = QHBoxLayout(row_card)
        r_lay.setContentsMargins(12, 6, 12, 6)
        r_lay.setSpacing(12)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        t_lbl.setStyleSheet("color: #f1f5f9; background: transparent;")
        text_box.addWidget(t_lbl)

        if subtitle:
            s_lbl = QLabel(subtitle)
            s_lbl.setFont(QFont("Arial", 9))
            s_lbl.setStyleSheet("color: #8da2c0; background: transparent;")
            s_lbl.setWordWrap(True)
            text_box.addWidget(s_lbl)

        r_lay.addLayout(text_box, 1)

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setFixedWidth(130)
        r_lay.addWidget(widget, 0, Qt.AlignRight | Qt.AlignVCenter)
        return row_card

    def build_parameters_panel(self):
        """Construit dynamiquement les champs d'ajustement avec nettoyage immédiat et récursif."""
        if hasattr(self, 'params_container') and self.params_container:
            self.params_vbox.removeWidget(self.params_container)
            self.params_container.setParent(None)
            self.params_container.deleteLater()
            self.params_container = None

        self.params_container = QWidget()
        cont_lay = QVBoxLayout(self.params_container)
        cont_lay.setContentsMargins(0, 0, 0, 0)
        cont_lay.setSpacing(8)

        pid = self.selected_trader_id
        strat = self.manager.strategies.get(pid)
        cfg = strat.config if strat else {}
        is_custom = (self.selected_mode == "custom")
        strat_name = strat.name if strat else pid.upper()

        # ── 1. BANNIÈRE D'INFORMATION DE VERROUILLAGE / PERSONNALISATION ──
        banner = QFrame()
        b_lay = QHBoxLayout(banner)
        b_lay.setContentsMargins(12, 8, 12, 8)
        b_lay.setSpacing(10)

        b_icon = QLabel("🔒" if not is_custom else "🔓")
        b_icon.setFont(QFont("Arial", 13))
        b_lay.addWidget(b_icon)

        if not is_custom:
            mode_lbl = "Simulation" if self.selected_mode == "simulation" else "Copie Démo"
            b_text = QLabel(
                f"<b>Paramètres officiels verrouillés ({mode_lbl}) :</b> "
                f"Reproduction stricte des règles de <b>{strat_name}</b>. "
                f"Pour déverrouiller et ajuster ces valeurs selon votre convenance, sélectionnez le <b>Mode 3 — Personnalisée</b> ci-dessus."
            )
            banner.setStyleSheet("""
                QFrame {
                    background-color: #0c182b;
                    border: 1px solid #1d4ed8;
                    border-radius: 6px;
                }
            """)
            b_text.setStyleSheet("color: #93c5fd; font-size: 11px;")
        else:
            b_text = QLabel(
                f"<b>Mode Personnalisé Actif :</b> Vous avez le plein contrôle sur les paramètres de <b>{strat_name}</b>. "
                f"Ajustez librement chaque seuil ou multiplicateur ci-dessous."
            )
            banner.setStyleSheet("""
                QFrame {
                    background-color: #261604;
                    border: 1px solid #d97706;
                    border-radius: 6px;
                }
            """)
            b_text.setStyleSheet("color: #fde047; font-size: 11px;")

        b_text.setWordWrap(True)
        b_lay.addWidget(b_text, 1)
        cont_lay.addWidget(banner)

        # ── 2. GRILLE RESPONSIVE DES PARAMÈTRES ──
        params_grid = QGridLayout()
        params_grid.setSpacing(8)

        self.param_widgets = {}
        rows_data = []

        if pid == "corentin":
            sp_range = QDoubleSpinBox()
            sp_range.setRange(0.2, 3.0); sp_range.setSingleStep(0.1); sp_range.setValue(float(cfg.get("range_atr_ratio", 0.8)))
            sp_range.setSuffix(" × ATR")
            self.param_widgets["range_atr_ratio"] = sp_range
            rows_data.append(("Filtre Range / ATR", "Ratio maximum de compression pour valider l'inducement", sp_range))

            sp_rr = QDoubleSpinBox()
            sp_rr.setRange(1.0, 5.0); sp_rr.setSingleStep(0.2); sp_rr.setValue(float(cfg.get("min_rr", 2.0)))
            sp_rr.setSuffix(" R")
            self.param_widgets["min_rr"] = sp_rr
            rows_data.append(("Ratio R:R Minimum", "Objectif de profit minimal garanti par rapport au risque", sp_rr))

            cb_flow = QCheckBox("Exiger confirmation du carnet d'ordres")
            cb_flow.setChecked(bool(cfg.get("use_flow", True)))
            self.param_widgets["use_flow"] = cb_flow
            rows_data.append(("Filtre Flux d'Ordres (Order Flow)", "Valide l'absorption ou le déséquilibre au carnet", cb_flow))

        elif pid == "benjamin":
            sp_win = QSpinBox()
            sp_win.setRange(2, 60); sp_win.setValue(int(cfg.get("open_window_min", 15)))
            sp_win.setSuffix(" min")
            self.param_widgets["open_window_min"] = sp_win
            rows_data.append(("Fenêtre d'Ouverture US", "Période active après 15h30 pour capturer l'impulsion", sp_win))

            sp_spr = QDoubleSpinBox()
            sp_spr.setRange(0.5, 20.0); sp_spr.setSingleStep(0.5); sp_spr.setValue(float(cfg.get("max_spread", 2.5)))
            sp_spr.setSuffix(" pts")
            self.param_widgets["max_spread"] = sp_spr
            rows_data.append(("Spread Maximum Toléré", "Bloque l'exécution si le spread dépasse ce seuil", sp_spr))

            sp_n = QSpinBox()
            sp_n.setRange(1, 5); sp_n.setValue(int(cfg.get("cluster_n", 3)))
            sp_n.setSuffix(" ordres")
            self.param_widgets["cluster_n"] = sp_n
            rows_data.append(("Grappe d'Ordres (Cluster)", "Nombre de positions échelonnées à l'entrée", sp_n))

            sp_step = QDoubleSpinBox()
            sp_step.setRange(1.0, 50.0); sp_step.setValue(float(cfg.get("step_pts", 5.0)))
            sp_step.setSuffix(" pts")
            self.param_widgets["step_pts"] = sp_step
            rows_data.append(("Espacement de Grappe", "Écart en points entre chaque palier d'entrée", sp_step))

            sp_tp = QDoubleSpinBox()
            sp_tp.setRange(1.0, 100.0); sp_tp.setValue(float(cfg.get("tp_pts", 8.0)))
            sp_tp.setSuffix(" pts")
            self.param_widgets["tp_pts"] = sp_tp
            rows_data.append(("Take Profit Scalp", "Objectif de sortie ultra-rapide pour sécuriser les gains", sp_tp))

        elif pid == "romain":
            sp_atr = QDoubleSpinBox()
            sp_atr.setRange(0.5, 4.0); sp_atr.setSingleStep(0.1); sp_atr.setValue(float(cfg.get("atr_mult", 1.5)))
            sp_atr.setSuffix(" × ATR")
            self.param_widgets["atr_mult"] = sp_atr
            rows_data.append(("Multiplicateur Stop Loss ATR", "Distance de sécurité basée sur la volatilité", sp_atr))

            sp_be = QDoubleSpinBox()
            sp_be.setRange(0.5, 3.0); sp_be.setSingleStep(0.2); sp_be.setValue(float(cfg.get("be_r", 1.0)))
            sp_be.setSuffix(" R")
            self.param_widgets["be_r"] = sp_be
            rows_data.append(("Seuil Break Even", "Déplace le Stop Loss au point d'entrée dès ce gain atteint", sp_be))

            sp_news = QSpinBox()
            sp_news.setRange(5, 120); sp_news.setValue(int(cfg.get("news_minutes", 30)))
            sp_news.setSuffix(" min")
            self.param_widgets["news_minutes"] = sp_news
            rows_data.append(("Filtre Annonces Économiques", "Gel des trades avant et après les publications à fort impact", sp_news))

            sp_fib = QDoubleSpinBox()
            sp_fib.setRange(0.50, 0.786); sp_fib.setDecimals(3); sp_fib.setValue(float(cfg.get("fib_level", 0.618)))
            sp_fib.setSingleStep(0.005)
            self.param_widgets["fib_level"] = sp_fib
            rows_data.append(("Niveau Golden Pocket", "Seuil de repli Fibonacci (0.618 = 61.8%)", sp_fib))

        elif pid == "nada":
            sp_rr = QDoubleSpinBox()
            sp_rr.setRange(1.5, 5.0); sp_rr.setSingleStep(0.2); sp_rr.setValue(float(cfg.get("min_rr", 2.0)))
            sp_rr.setSuffix(" R")
            self.param_widgets["min_rr"] = sp_rr
            rows_data.append(("Ratio R:R Minimum Garanti", "Objectif de rentabilité minimal exigé pour l'ordre", sp_rr))

            sp_fmin = QDoubleSpinBox()
            sp_fmin.setRange(0.382, 0.618); sp_fmin.setDecimals(3); sp_fmin.setValue(float(cfg.get("fib_zone_min", 0.50)))
            sp_fmin.setSingleStep(0.005)
            self.param_widgets["fib_zone_min"] = sp_fmin
            rows_data.append(("Zone Retest Fibo Minimum", "Seuil d'entrée inférieur sur repli (0.500 = 50.0%)", sp_fmin))

            sp_fmax = QDoubleSpinBox()
            sp_fmax.setRange(0.50, 0.786); sp_fmax.setDecimals(3); sp_fmax.setValue(float(cfg.get("fib_zone_max", 0.618)))
            sp_fmax.setSingleStep(0.005)
            self.param_widgets["fib_zone_max"] = sp_fmax
            rows_data.append(("Zone Retest Fibo Maximum", "Seuil d'entrée supérieur sur repli (0.618 = 61.8%)", sp_fmax))

        elif pid == "soros":
            sp_dev = QDoubleSpinBox()
            sp_dev.setRange(1.0, 5.0); sp_dev.setSingleStep(0.2); sp_dev.setValue(float(cfg.get("deviation_atr_mult", 2.0)))
            sp_dev.setSuffix(" × ATR")
            self.param_widgets["deviation_atr_mult"] = sp_dev
            rows_data.append(("Distorsion Réflexive Min", "Écart minimal requis par rapport à l'EMA 200 institutionnelle", sp_dev))

            sp_rr = QDoubleSpinBox()
            sp_rr.setRange(2.0, 6.0); sp_rr.setSingleStep(0.25); sp_rr.setValue(float(cfg.get("min_rr", 2.5)))
            sp_rr.setSuffix(" R")
            self.param_widgets["min_rr"] = sp_rr
            rows_data.append(("Ratio R:R Asymétrique Minimum", "Exigence stricte de rentabilité asymétrique (au moins 2.5:1)", sp_rr))

            sp_rob = QDoubleSpinBox()
            sp_rob.setRange(60.0, 90.0); sp_rob.setSingleStep(1.0); sp_rob.setValue(float(cfg.get("rsi_overbought", 68.0)))
            sp_rob.setSuffix(" pts")
            self.param_widgets["rsi_overbought"] = sp_rob
            rows_data.append(("RSI Seuil Sur-Achat (Bulle)", "Niveau d'épuisement acheteur annonçant l'éclatement", sp_rob))

            sp_ros = QDoubleSpinBox()
            sp_ros.setRange(10.0, 40.0); sp_ros.setSingleStep(1.0); sp_ros.setValue(float(cfg.get("rsi_oversold", 32.0)))
            sp_ros.setSuffix(" pts")
            self.param_widgets["rsi_oversold"] = sp_ros
            rows_data.append(("RSI Seuil Sur-Vente (Panique)", "Niveau d'épuisement vendeur annonçant le rebond réflexif", sp_ros))

        # Disposition en 2 colonnes équilibrées
        p_row, p_col = 0, 0
        for idx, (title, sub, w) in enumerate(rows_data):
            w.setEnabled(is_custom)
            card_row = self._create_param_row(title, sub, w)
            if idx == len(rows_data) - 1 and len(rows_data) % 2 != 0:
                params_grid.addWidget(card_row, p_row, 0, 1, 2)
            else:
                params_grid.addWidget(card_row, p_row, p_col)
                p_col += 1
                if p_col > 1:
                    p_col = 0
                    p_row += 1

        cont_lay.addLayout(params_grid)
        self.params_vbox.addWidget(self.params_container)

    def _update_card_styles(self):
        """Met à jour l'apparence des cartes de traders et des modes selon la sélection active."""
        # 1. Cartes des traders
        for pid, (radio, card, accent_col) in self.trader_cards.items():
            is_sel = (pid == self.selected_trader_id)
            if is_sel:
                card.setStyleSheet(f"""
                    QFrame#card_{pid} {{
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 #1b263d, stop:1 #131b2c);
                        border: 2px solid {accent_col};
                        border-radius: 8px;
                    }}
                """)
            else:
                card.setStyleSheet(f"""
                    QFrame#card_{pid} {{
                        background-color: #121826;
                        border: 1px solid #232d42;
                        border-radius: 8px;
                    }}
                    QFrame#card_{pid}:hover {{
                        border: 1px solid {accent_col};
                        background-color: #162033;
                    }}
                """)

        # 2. Cartes des modes
        for mode_key, (radio, card, accent_col) in self.mode_cards.items():
            is_sel = (mode_key == self.selected_mode)
            if is_sel:
                card.setStyleSheet(f"""
                    QFrame#mode_card_{mode_key} {{
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 #1b2742, stop:1 #121a2d);
                        border: 2px solid {accent_col};
                        border-radius: 8px;
                    }}
                """)
            else:
                card.setStyleSheet(f"""
                    QFrame#mode_card_{mode_key} {{
                        background-color: #121826;
                        border: 1px solid #232d42;
                        border-radius: 8px;
                    }}
                    QFrame#mode_card_{mode_key}:hover {{
                        border: 1px solid {accent_col};
                        background-color: #162033;
                    }}
                """)

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

        self.selected_mode = active_mode if active_mode in ("copy", "custom") else "simulation"
        if self.selected_mode in self.mode_cards:
            self.mode_cards[self.selected_mode][0].setChecked(True)

        self._update_card_styles()
        self.build_parameters_panel()
        self._update_status_badge()
        self.refresh_stats_table()

    def on_trader_card_clicked(self, pid: str):
        """Sélectionne un trader via clic sur sa carte entière."""
        if pid in self.trader_cards:
            self.trader_cards[pid][0].setChecked(True)
            self.selected_trader_id = pid
            self._update_card_styles()
            self.build_parameters_panel()

    def on_trader_selected(self, button):
        """Gère le clic sur le radio bouton d'une carte de trader."""
        for pid, (radio, card, accent_col) in self.trader_cards.items():
            if radio == button:
                self.selected_trader_id = pid
                break
        self._update_card_styles()
        self.build_parameters_panel()

    def on_mode_card_clicked(self, mode_id: str):
        """Sélectionne un mode d'opération via clic sur sa carte entière."""
        if mode_id in self.mode_cards:
            self.mode_cards[mode_id][0].setChecked(True)
            self.selected_mode = mode_id
            self._update_card_styles()
            self.build_parameters_panel()

    def on_mode_radio_clicked(self, button):
        """Gère le clic direct sur les radio boutons des modes."""
        for m_id, (radio, card, accent_col) in self.mode_cards.items():
            if radio == button:
                self.selected_mode = m_id
                break
        self._update_card_styles()
        self.build_parameters_panel()

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

        if active_pid and active_pid in self.manager.strategies:
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
        self.stats_table.setRowCount(len(profiles))
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
