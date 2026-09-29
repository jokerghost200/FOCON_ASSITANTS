"""
Boîte de dialogue interactive pour les propositions d'ajustement du Conseiller IA.
================================================================================
Affiche les changements proposés par l'IA sous forme de comparaison claire (actuel -> proposé),
avec la justification et les boutons de confirmation sécurisée.
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QFrame,
    QAbstractItemView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


class ConfigProposalDialog(QDialog):
    def __init__(
        self,
        reply_text: str,
        changes: List[Dict[str, Any]],
        voice_manager=None,
        parent=None
    ):
        super().__init__(parent)
        self.reply_text = reply_text
        self.changes = changes
        self.voice_manager = voice_manager
        self.applied = False

        self.setWindowTitle("🧠 Conseiller IA — Proposition d'Ajustement des Réglages")
        self.setMinimumSize(700, 520)
        self.resize(780, 580)
        self.setStyleSheet("""
            QDialog {
                background-color: #131722;
                color: #d1d4dc;
            }
            QLabel {
                color: #d1d4dc;
            }
            QFrame#card {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 8px;
            }
        """)

        self._init_ui()

        # Lecture vocale automatique si le son n'est pas coupé
        if self.voice_manager and not self.voice_manager.is_muted and self.reply_text:
            self.voice_manager.speak(self.reply_text)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ── 1. En-tête avec titre & badge sécurité ──
        header_layout = QHBoxLayout()
        title_lbl = QLabel("🧠 Recommandations Stratégiques du Bot")
        title_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        title_lbl.setStyleSheet("color: #2962ff;")
        header_layout.addWidget(title_lbl)

        header_layout.addStretch()

        if self.voice_manager:
            self.speak_btn = QPushButton("🔊 Réécouter")
            self.speak_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2e39;
                    color: white;
                    border: 1px solid #363c4e;
                    border-radius: 4px;
                    padding: 5px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #363c4e;
                }
            """)
            self.speak_btn.clicked.connect(self._on_replay_speech)
            header_layout.addWidget(self.speak_btn)

        layout.addLayout(header_layout)

        # ── 2. Explication de l'IA (Card) ──
        exp_card = QFrame()
        exp_card.setObjectName("card")
        exp_layout = QVBoxLayout(exp_card)
        exp_layout.setContentsMargins(14, 12, 14, 12)

        exp_title = QLabel("💡 Analyse & Diagnostic :")
        exp_title.setFont(QFont("Arial", 11, QFont.Bold))
        exp_title.setStyleSheet("color: #ff9800;")
        exp_layout.addWidget(exp_title)

        self.exp_display = QTextEdit()
        self.exp_display.setReadOnly(True)
        self.exp_display.setFont(QFont("Arial", 10))
        self.exp_display.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #e0e0e0;
                line-height: 1.4;
            }
        """)
        self.exp_display.setPlainText(self.reply_text or "L'IA propose les ajustements ci-dessous pour optimiser vos performances.")
        self.exp_display.setMaximumHeight(90)
        exp_layout.addWidget(self.exp_display)
        layout.addWidget(exp_card)

        # ── 3. Tableau comparatif des modifications ──
        table_lbl = QLabel(f"📋 Modifications proposées ({len(self.changes)}) :")
        table_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(table_lbl)

        self.table = QTableWidget(len(self.changes), 4)
        self.table.setHorizontalHeaderLabels(["Paramètre", "Valeur Actuelle", "Valeur Proposée", "Justification"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e222d;
                color: #d1d4dc;
                gridline-color: #2a2e39;
                border: 1px solid #2a2e39;
                border-radius: 6px;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #2a2e39;
                color: #9ca3af;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
        """)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        for row, ch in enumerate(self.changes):
            label = ch.get("label", f"{ch.get('section')}.{ch.get('key')}")
            curr_val = str(ch.get("current", "N/A"))
            new_val = str(ch.get("value", ""))
            reason = ch.get("reason", "")

            # Formater l'affichage des valeurs
            item_lbl = QTableWidgetItem(f"⚙️ {label}")
            item_lbl.setFont(QFont("Arial", 10, QFont.Bold))

            item_curr = QTableWidgetItem(curr_val)
            item_curr.setTextAlignment(Qt.AlignCenter)
            item_curr.setForeground(QColor("#9ca3af"))

            item_new = QTableWidgetItem(f"➡️ {new_val}")
            item_new.setTextAlignment(Qt.AlignCenter)
            item_new.setFont(QFont("Arial", 10, QFont.Bold))
            item_new.setForeground(QColor("#00e676"))  # Vert vif pour la nouvelle valeur

            item_reason = QTableWidgetItem(reason)
            item_reason.setForeground(QColor("#cfd8dc"))

            self.table.setItem(row, 0, item_lbl)
            self.table.setItem(row, 1, item_curr)
            self.table.setItem(row, 2, item_new)
            self.table.setItem(row, 3, item_reason)

        layout.addWidget(self.table)

        # ── 4. Avertissement de sécurité ──
        safety_lbl = QLabel("🛡️ Sécurité : Aucun paramètre n'est modifié tant que vous ne cliquez pas sur 'Valider et Appliquer'.")
        safety_lbl.setFont(QFont("Arial", 9, QFont.StyleItalic))
        safety_lbl.setStyleSheet("color: #787b86;")
        layout.addWidget(safety_lbl)

        # ── 5. Boutons d'Action ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.cancel_btn = QPushButton("❌ Conserver les réglages actuels")
        self.cancel_btn.setFont(QFont("Arial", 10))
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #d1d4dc;
                border: 1px solid #363c4e;
                border-radius: 6px;
                padding: 10px 18px;
            }
            QPushButton:hover {
                background-color: #363c4e;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)

        self.apply_btn = QPushButton("✅ Valider et Appliquer les Réglages")
        self.apply_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #00c853;
                color: #000000;
                border: none;
                border-radius: 6px;
                padding: 11px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #69f0ae;
            }
            QPushButton:pressed {
                background-color: #00b248;
            }
        """)
        self.apply_btn.clicked.connect(self._on_apply)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)

    def _on_replay_speech(self):
        if self.voice_manager and self.reply_text:
            self.voice_manager.speak(self.reply_text)

    def _on_apply(self):
        self.applied = True
        self.accept()

    def closeEvent(self, event):
        # Arrêter la synthèse vocale si l'utilisateur ferme la boîte
        if self.voice_manager:
            self.voice_manager.stop_speaking()
        event.accept()
