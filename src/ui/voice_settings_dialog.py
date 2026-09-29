"""
Boîte de dialogue de configuration des paramètres vocaux et de l'Auto-Adaptation IA.
=====================================================================================
Permet de personnaliser :
- Le choix de la voix (Microsoft Hortense, Julie, Paul, etc.)
- La vitesse de parole (Rate)
- La hauteur de voix / tonalité (Pitch)
- Le volume sonore (Volume)
- L'activation de l'Auto-Adaptation intelligente du trade en cas de danger
- Les annonces vocales en direct (prises de position, break-even, TP)
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QFrame, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


class VoiceSettingsDialog(QDialog):
    def __init__(self, voice_manager, config: dict, on_save_callback=None, parent=None):
        super().__init__(parent)
        self.voice_manager = voice_manager
        self.config = config or {}
        self.on_save_callback = on_save_callback

        self.setWindowTitle("⚙️ Paramètres Vocaux & Intelligence Adaptative")
        self.setMinimumSize(620, 560)
        self.resize(650, 580)
        self.setStyleSheet("""
            QDialog {
                background-color: #131722;
                color: #d1d4dc;
            }
            QLabel {
                color: #d1d4dc;
            }
            QGroupBox {
                border: 1px solid #2a2e39;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: bold;
                color: #2962ff;
                background-color: #1e222d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QComboBox {
                background-color: #2a2e39;
                color: white;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 22px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #363c4e;
                height: 6px;
                background: #2a2e39;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2962ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #2962ff;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)

        self._init_ui()
        self._load_current_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ── Titre ──
        header = QHBoxLayout()
        title_lbl = QLabel("🎙️ Personnalisation de la Voix & Cerveau Vocal")
        title_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        title_lbl.setStyleSheet("color: #2962ff;")
        header.addWidget(title_lbl)
        header.addStretch()
        layout.addLayout(header)

        # ── 1. Section Moteur Vocal ──
        tts_group = QGroupBox("🗣️ Synthèse Vocale & Tonalité")
        tts_group.setFont(QFont("Arial", 11, QFont.Bold))
        tts_layout = QVBoxLayout(tts_group)
        tts_layout.setSpacing(12)

        # Choix de la voix
        voice_row = QHBoxLayout()
        v_lbl = QLabel("Voix du Robot :")
        v_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        v_lbl.setFixedWidth(130)
        self.voice_combo = QComboBox()
        self.voice_combo.setFont(QFont("Arial", 10))
        voice_row.addWidget(v_lbl)
        voice_row.addWidget(self.voice_combo, stretch=1)
        tts_layout.addLayout(voice_row)

        # Vitesse (Rate)
        rate_row = QHBoxLayout()
        r_lbl = QLabel("Vitesse de parole :")
        r_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        r_lbl.setFixedWidth(130)
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(-100, 100)
        self.rate_slider.setValue(0)
        self.rate_val_lbl = QLabel("Normale (1.0x)")
        self.rate_val_lbl.setFixedWidth(110)
        self.rate_val_lbl.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.rate_slider.valueChanged.connect(self._on_rate_slider_changed)
        rate_row.addWidget(r_lbl)
        rate_row.addWidget(self.rate_slider, stretch=1)
        rate_row.addWidget(self.rate_val_lbl)
        tts_layout.addLayout(rate_row)

        # Hauteur / Tonalité (Pitch)
        pitch_row = QHBoxLayout()
        p_lbl = QLabel("Hauteur de ton :")
        p_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        p_lbl.setFixedWidth(130)
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-100, 100)
        self.pitch_slider.setValue(0)
        self.pitch_val_lbl = QLabel("Standard")
        self.pitch_val_lbl.setFixedWidth(110)
        self.pitch_val_lbl.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.pitch_slider.valueChanged.connect(self._on_pitch_slider_changed)
        pitch_row.addWidget(p_lbl)
        pitch_row.addWidget(self.pitch_slider, stretch=1)
        pitch_row.addWidget(self.pitch_val_lbl)
        tts_layout.addLayout(pitch_row)

        # Volume sonore
        vol_row = QHBoxLayout()
        vol_lbl = QLabel("Volume sonore :")
        vol_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        vol_lbl.setFixedWidth(130)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(100)
        self.vol_val_lbl = QLabel("100 %")
        self.vol_val_lbl.setFixedWidth(110)
        self.vol_val_lbl.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.vol_slider.valueChanged.connect(self._on_vol_slider_changed)
        vol_row.addWidget(vol_lbl)
        vol_row.addWidget(self.vol_slider, stretch=1)
        vol_row.addWidget(self.vol_val_lbl)
        tts_layout.addLayout(vol_row)

        # Bouton tester la voix
        test_row = QHBoxLayout()
        test_row.addStretch()
        self.test_btn = QPushButton("🔊 Tester la voix")
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #e0e0e0;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #363c4e;
                color: white;
            }
        """)
        self.test_btn.clicked.connect(self._on_test_voice)
        test_row.addWidget(self.test_btn)
        tts_layout.addLayout(test_row)

        layout.addWidget(tts_group)

        # ── 2. Section Intelligence & Auto-Adaptation ──
        intel_group = QGroupBox("🧠 Intelligence & Adaptation au Marché")
        intel_group.setFont(QFont("Arial", 11, QFont.Bold))
        intel_layout = QVBoxLayout(intel_group)
        intel_layout.setSpacing(10)

        # Checkbox Auto-Adaptation
        self.auto_adapt_cb = QCheckBox("🤖 Auto-Adaptation Proactive aux Conditions de Marché")
        self.auto_adapt_cb.setFont(QFont("Arial", 10, QFont.Bold))
        self.auto_adapt_cb.setStyleSheet("color: #00e676;")
        auto_adapt_desc = QLabel(
            "Permet au robot d'adapter automatiquement ses réglages (réduction du risque, "
            "hausse du filtre de confiance, blocage des contre-tendances) lorsqu'il détecte "
            "un danger, une série de pertes ou une opportunité pour maximiser les profits."
        )
        auto_adapt_desc.setWordWrap(True)
        auto_adapt_desc.setStyleSheet("color: #9ca3af; font-size: 11px; margin-left: 26px;")
        intel_layout.addWidget(self.auto_adapt_cb)
        intel_layout.addWidget(auto_adapt_desc)

        # Checkbox Annonces Vocales en direct
        self.announcements_cb = QCheckBox("📢 Annonces Vocales de Trading en Direct")
        self.announcements_cb.setFont(QFont("Arial", 10, QFont.Bold))
        announcements_desc = QLabel(
            "Le robot commente vocalement ses actions : détection d'opportunité institutionnelle, "
            "entrée en position, déplacement du Stop Loss à Break-Even et encaissement d'objectifs."
        )
        announcements_desc.setWordWrap(True)
        announcements_desc.setStyleSheet("color: #9ca3af; font-size: 11px; margin-left: 26px;")
        intel_layout.addWidget(self.announcements_cb)
        intel_layout.addWidget(announcements_desc)

        layout.addWidget(intel_group)

        # ── 3. Boutons d'Action ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.cancel_btn = QPushButton("❌ Annuler")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #d1d4dc;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 9px 18px;
            }
            QPushButton:hover {
                background-color: #363c4e;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("💾 Enregistrer et Appliquer")
        self.save_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2962ff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 22px;
            }
            QPushButton:hover {
                background-color: #1e40af;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

    def _load_current_values(self):
        """Remplit l'interface avec les réglages actuels."""
        voice_cfg = self.config.get("voice", {})

        # 1. Remplir les voix disponibles
        if self.voice_manager:
            voices = self.voice_manager.get_available_voices()
            curr_voice = self.voice_manager.get_current_voice_name()
            saved_voice = voice_cfg.get("voice_name", curr_voice)

            self.voice_combo.clear()
            for v_name in voices:
                self.voice_combo.addItem(f"👤 {v_name}", v_name)

            idx = self.voice_combo.findData(saved_voice)
            if idx >= 0:
                self.voice_combo.setCurrentIndex(idx)

        # 2. Vitesse
        rate = float(voice_cfg.get("rate", getattr(self.voice_manager, "rate", 0.0)))
        slider_val = int(rate * 100)
        self.rate_slider.setValue(slider_val)
        self._on_rate_slider_changed(slider_val)

        # 3. Hauteur (Pitch)
        pitch = float(voice_cfg.get("pitch", getattr(self.voice_manager, "pitch", 0.0)))
        pitch_val = int(pitch * 100)
        self.pitch_slider.setValue(pitch_val)
        self._on_pitch_slider_changed(pitch_val)

        # 4. Volume
        volume = float(voice_cfg.get("volume", getattr(self.voice_manager, "volume", 1.0)))
        vol_val = int(volume * 100)
        self.vol_slider.setValue(vol_val)
        self._on_vol_slider_changed(vol_val)

        # 5. Checkboxes
        auto_adapt = bool(voice_cfg.get("auto_adapt_on_danger", getattr(self.voice_manager, "auto_adapt_on_danger", False)))
        self.auto_adapt_cb.setChecked(auto_adapt)

        announcements = bool(voice_cfg.get("voice_announcements", getattr(self.voice_manager, "voice_announcements", True)))
        self.announcements_cb.setChecked(announcements)

    def _on_rate_slider_changed(self, val):
        rate = val / 100.0
        if val == 0:
            txt = "Normale (1.0x)"
        elif val > 0:
            txt = f"Rapide (+{val}%)"
        else:
            txt = f"Lente ({val}%)"
        self.rate_val_lbl.setText(txt)
        if self.voice_manager:
            self.voice_manager.set_rate(rate)

    def _on_pitch_slider_changed(self, val):
        pitch = val / 100.0
        if val == 0:
            txt = "Standard"
        elif val > 0:
            txt = f"Aiguë (+{val}%)"
        else:
            txt = f"Grave ({val}%)"
        self.pitch_val_lbl.setText(txt)
        if self.voice_manager:
            self.voice_manager.set_pitch(pitch)

    def _on_vol_slider_changed(self, val):
        vol = val / 100.0
        self.vol_val_lbl.setText(f"{val} %")
        if self.voice_manager:
            self.voice_manager.set_volume(vol)

    def _on_test_voice(self):
        """Prononce une phrase de test avec les paramètres choisis."""
        if not self.voice_manager:
            return

        # Appliquer temporairement la voix choisie
        chosen_voice = self.voice_combo.currentData()
        if chosen_voice:
            self.voice_manager.set_voice_by_name(chosen_voice)

        self.voice_manager.set_rate(self.rate_slider.value() / 100.0)
        self.voice_manager.set_pitch(self.pitch_slider.value() / 100.0)
        self.voice_manager.set_volume(self.vol_slider.value() / 100.0)

        test_phrase = (
            "Bonjour ! Je suis Focon Assistant, votre intelligence artificielle de trading. "
            "Mes paramètres vocaux et adaptatifs sont configurés."
        )
        self.voice_manager.speak(test_phrase)

    def _on_save(self):
        """Sauvegarde les paramètres et notifie l'application."""
        chosen_voice = self.voice_combo.currentData() or ""
        rate = self.rate_slider.value() / 100.0
        pitch = self.pitch_slider.value() / 100.0
        volume = self.vol_slider.value() / 100.0
        auto_adapt = self.auto_adapt_cb.isChecked()
        announcements = self.announcements_cb.isChecked()

        settings = {
            "enabled": True,
            "voice_name": chosen_voice,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
            "auto_adapt_on_danger": auto_adapt,
            "voice_announcements": announcements,
        }

        if self.voice_manager:
            self.voice_manager.load_config(settings)

        if self.on_save_callback:
            self.on_save_callback(settings)

        self.accept()
