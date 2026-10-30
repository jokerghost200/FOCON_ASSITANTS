import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFrame, QMessageBox, QGroupBox,
    QScrollArea, QDialog, QCheckBox, QDoubleSpinBox, QSpinBox, QFormLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QComboBox,
    QSplitter, QSizePolicy
)
import yaml
import json
import traceback
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

import threading


def _log_crash(where: str):
    """Loggue le traceback complet (visible dans le terminal py main.py)."""
    import logging as _lg
    _lg.getLogger(__name__).exception("CRASH %s", where)


def _qt_excepthook(exc_type, exc_value, exc_tb):
    import logging as _lg
    _lg.getLogger("qt.crash").error(
        "Uncaught exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
    )
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _qt_excepthook

class ScannerThread(QThread):
    scan_finished = Signal(list)
    scan_failed = Signal(str)
    progress = Signal(str)  # emits status messages to keep UI alive

    def __init__(self, orchestrator, symbols, timeframe,
                 allowed_types=None, fallback_symbols=None, direction_filter=None, parent=None, watchdog=None, ai_queue=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.symbols = symbols          # peut être None -> fetch dans run()
        self.timeframe = timeframe
        self.allowed_types = allowed_types
        self.fallback_symbols = fallback_symbols
        self.direction_filter = direction_filter or getattr(orchestrator, 'allowed_trade_direction', 'BOTH')
        self.stop_event = threading.Event()
        self.watchdog = watchdog
        self.ai_queue = ai_queue

    def run(self):
        try:
            # IMPORTANT : les appels MT5 (IPC native C) ne doivent jamais être
            # faits depuis le thread GUI — un crash C là ferme l'app sans
            # aucun traceback. Le fetch des symboles se fait donc ici.
            if self.watchdog:
                self.watchdog.heartbeat("ScannerThread")
            if not self.symbols:
                self.progress.emit("Récupération des symboles MT5...")
                self.symbols = self.orchestrator.mt5.get_visible_symbols(
                    allowed_types=self.allowed_types)
            if not self.symbols:
                # Fallback to config if MT5 doesn't return anything
                self.symbols = list(self.fallback_symbols or ["EURUSD"])
            
            def heartbeat_cb(*args, **kwargs):
                if self.watchdog:
                    self.watchdog.heartbeat("ScannerThread")

            results = self.orchestrator.scan_markets(
                self.symbols, self.timeframe,
                progress_cb=self.progress.emit,
                stop_event=self.stop_event,
                direction_filter=self.direction_filter,
                heartbeat_cb=heartbeat_cb,
                ai_queue=self.ai_queue
            )
            if self.watchdog:
                self.watchdog.heartbeat("ScannerThread")
        except Exception as e:
            _log_crash("ScannerThread.run")
            try:
                self.scan_failed.emit(f"{type(e).__name__}: {e}")
            except Exception:
                pass
            return
        try:
            self.scan_finished.emit(results if isinstance(results, list) else [])
        except Exception:
            _log_crash("ScannerThread.emit")

    def stop(self):
        """Signal the thread to stop as soon as possible."""
        self.stop_event.set()

class ChatThread(QThread):
    chat_finished = Signal(object)  # émet soit str soit dict {"reply": str, "changes": list}
    
    def __init__(self, chat_engine, message, config_advisor=None, context=None):
        super().__init__()
        self.chat_engine = chat_engine
        self.message = message
        self.config_advisor = config_advisor
        self.context = context or {}
        
    def run(self):
        msg_lower = self.message.lower()
        keywords = [
            "param", "réglage", "reglage", "config", "optimis", "ajust", "conseil",
            "chang", "risqu", "capital", "direction", "mode", "forex", "crypto",
            "sell", "buy", "perte", "gain", "stop", "tp", "be", "lot", "volatilit",
            "compte", "micro", "augment", "diminu", "appliqu", "bascul"
        ]
        is_advice = any(k in msg_lower for k in keywords)

        if self.config_advisor and is_advice:
            try:
                advice = self.config_advisor.advise(self.message, self.context)
                if advice and (advice.get("changes") or advice.get("reply")):
                    self.chat_finished.emit(advice)
                    return
            except Exception:
                import logging as _lg
                _lg.getLogger(__name__).exception("Erreur ConfigAdvisor dans ChatThread")

        # Conversation standard
        try:
            result = self.chat_engine.chat(self.message)
            self.chat_finished.emit({"reply": result, "changes": []})
        except Exception as e:
            self.chat_finished.emit({"reply": f"Erreur de communication : {e}", "changes": []})

class AIAnalyzerThread(QThread):
    analysis_finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, orchestrator, ai_queue, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.ai_queue = ai_queue
        self.running = True

    def run(self):
        import queue
        while self.running:
            try:
                item = self.ai_queue.get(timeout=1.0)
                if item is None:
                    continue
                sym, timeframe, direction_filter = item
                self.progress.emit(f"🧠 [IA Async] Analyse approfondie en arrière-plan pour {sym}...")
                res = self.orchestrator.analyze_symbol(sym, timeframe, stop_event=None, direction_filter=direction_filter)
                if res:
                    self.analysis_finished.emit(res)
            except queue.Empty:
                continue
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).exception("Erreur dans AIAnalyzerThread")
                self.progress.emit(f"❌ Erreur IA Async: {e}")

    def stop(self):
        self.running = False

class OBScanThread(QThread):
    """Thread de rescanning Order Block post-trade (2-3 passes apres 5 min)."""
    ob_found = Signal(list)    # emet les signaux OB detectes
    progress = Signal(str)     # messages de progression vers le chat UI

    def __init__(self, orchestrator, symbols, timeframe, direction_filter,
                 passes=3, interval_s=300, initial_ticket=0):
        super().__init__()
        self.orchestrator = orchestrator
        self.symbols = symbols
        self.timeframe = timeframe
        self.direction_filter = direction_filter
        self.passes = passes
        self.interval_s = interval_s   # 300s = 5 min
        self.initial_ticket = initial_ticket   # ticket du trade parent (pour la limite 1 OB/trade)
        self._stop = threading.Event()

    def run(self):
        import logging as _lg
        import time as _time
        log = _lg.getLogger(__name__)
        for i in range(self.passes):
            if self._stop.is_set():
                break
            msg = f"[OB Rescan] Passe {i+1}/{self.passes} dans {self.interval_s // 60} min (ticket #{self.initial_ticket})..."
            log.info(msg)
            self.progress.emit(msg)
            # Attendre l'intervalle (interruptible)
            for _ in range(self.interval_s):
                if self._stop.is_set():
                    return
                _time.sleep(1)
            if self._stop.is_set():
                break
            try:
                scan_msg = f"[OB Rescan] Passe {i+1}/{self.passes} — Scan en cours sur {len(self.symbols)} symboles..."
                log.info(scan_msg)
                self.progress.emit(scan_msg)
                results = self.orchestrator.scan_for_order_blocks(
                    self.symbols, self.timeframe,
                    direction_filter=self.direction_filter,
                    stop_event=self._stop,
                )
                n_found = len(results) if results else 0
                done_msg = f"[OB Rescan] Passe {i+1}/{self.passes} terminee — {n_found} OB valide(s) detecte(s)."
                log.info(done_msg)
                self.progress.emit(done_msg)
                if results:
                    self.ob_found.emit(results)
            except Exception:
                log.exception("[OB Rescan] Erreur lors du scan Order Block")
                self.progress.emit(f"[OB Rescan] Passe {i+1}/{self.passes} — Erreur (voir logs).")

    def stop(self):
        self._stop.set()

class FixedLotAdvisorDialog(QDialog):
    """
    Boîte de dialogue interactive permettant de voir et choisir le Lot Fixe
    qui passe le mieux en fonction du capital réel et de la marge libre MT5.
    """
    def __init__(self, capital: float = 81.03, margin_free: float = 81.03, leverage: int = 500, current_fixed_lot: float = 0.0, parent=None):
        super().__init__(parent)
        self.capital = max(10.0, float(capital or 81.03))
        self.margin_free = max(10.0, float(margin_free or self.capital))
        self.leverage = max(30, int(leverage or 500))
        self.current_fixed_lot = float(current_fixed_lot or 0.0)
        self.selected_lot = self.current_fixed_lot

        self.setWindowTitle("⚖️ Conseiller Lot Fixe — Calibrage selon votre Capital")
        self.setMinimumSize(780, 520)
        self.setStyleSheet("""
            QDialog { background-color: #121622; color: #e2e8f0; }
            QLabel { color: #e2e8f0; }
            QTableWidget {
                background-color: #1a2032;
                color: #e2e8f0;
                gridline-color: #2d3748;
                border: 1px solid #2d3748;
                border-radius: 6px;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #0f1523;
                color: #90caf9;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #2d3748;
            }
            QDoubleSpinBox {
                background-color: #0f1523;
                color: #ffd54f;
                font-weight: bold;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        self._init_ui()
        self._recalculate_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Bandeau explicatif
        info_banner = QLabel(
            "💡 <b>Pourquoi le mode Dynamique (0.00) ouvre uniquement 0.01 lot sur petit capital ?</b><br>"
            "Avec 1% à 2% de risque sur un petit capital (~1$ à 2$ de risque max), tout Stop Loss standard (10-15 pips) "
            "donne mathématiquement le minimum courtier (<b>0.01 lot</b>).<br>"
            "En passant en <b>Lot Fixe</b>, voici les volumes calibrés selon votre capital qui <b>passent les contrôles de marge et de sécurité</b> :"
        )
        info_banner.setWordWrap(True)
        info_banner.setStyleSheet(
            "background-color: #1e293b; color: #cbd5e1; border-left: 4px solid #38bdf8; "
            "padding: 10px; border-radius: 4px; font-size: 12px;"
        )
        layout.addWidget(info_banner)

        # Paramètres du compte (modifiables pour simuler)
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        cap_lbl = QLabel("💰 Capital (Balance/Equity) :")
        cap_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.cap_spin = QDoubleSpinBox()
        self.cap_spin.setRange(5.0, 1000000.0)
        self.cap_spin.setDecimals(2)
        self.cap_spin.setSingleStep(10.0)
        self.cap_spin.setSuffix(" $")
        self.cap_spin.setValue(self.capital)
        self.cap_spin.valueChanged.connect(self._on_params_changed)
        top_bar.addWidget(cap_lbl)
        top_bar.addWidget(self.cap_spin)

        mf_lbl = QLabel("🛡️ Marge Libre :")
        mf_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.mf_spin = QDoubleSpinBox()
        self.mf_spin.setRange(5.0, 1000000.0)
        self.mf_spin.setDecimals(2)
        self.mf_spin.setSingleStep(10.0)
        self.mf_spin.setSuffix(" $")
        self.mf_spin.setValue(self.margin_free)
        self.mf_spin.valueChanged.connect(self._on_params_changed)
        top_bar.addWidget(mf_lbl)
        top_bar.addWidget(self.mf_spin)

        sl_lbl = QLabel("📏 Stop Loss type :")
        sl_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.sl_pips_spin = QDoubleSpinBox()
        self.sl_pips_spin.setRange(3.0, 100.0)
        self.sl_pips_spin.setDecimals(1)
        self.sl_pips_spin.setSingleStep(1.0)
        self.sl_pips_spin.setSuffix(" pips")
        self.sl_pips_spin.setValue(10.0)
        self.sl_pips_spin.setToolTip("Distance moyenne du Stop Loss en pips (10 pips ≈ standard M15 Forex, soit ~1$ par 0.01 lot).")
        self.sl_pips_spin.valueChanged.connect(self._on_params_changed)
        top_bar.addWidget(sl_lbl)
        top_bar.addWidget(self.sl_pips_spin)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Tableau des recommandations
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Profil de Lot",
            "Lot Fixe",
            "Risque / Trade (SL)",
            "Gain Visé (R:R 1:2)",
            "Marge Requise (~)",
            "Verdict Bot & Sécurité",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, stretch=1)

        # Zone de sélection personnalisée + Verdict en direct
        sel_frame = QFrame()
        sel_frame.setStyleSheet("background-color: #1a2234; border: 1px solid #334155; border-radius: 6px; padding: 8px;")
        sel_layout = QHBoxLayout(sel_frame)

        sel_lbl = QLabel("🎯 <b>Lot Fixe sélectionné :</b>")
        sel_lbl.setFont(QFont("Arial", 10))
        sel_layout.addWidget(sel_lbl)

        self.selected_lot_spin = QDoubleSpinBox()
        self.selected_lot_spin.setRange(0.0, 50.0)
        self.selected_lot_spin.setDecimals(2)
        self.selected_lot_spin.setSingleStep(0.01)
        self.selected_lot_spin.setMinimumWidth(100)
        self.selected_lot_spin.setFont(QFont("Arial", 11, QFont.Bold))
        self.selected_lot_spin.valueChanged.connect(self._update_custom_verdict)
        sel_layout.addWidget(self.selected_lot_spin)

        self.live_verdict_lbl = QLabel("")
        self.live_verdict_lbl.setWordWrap(True)
        sel_layout.addWidget(self.live_verdict_lbl, stretch=1)

        layout.addWidget(sel_frame)

        # Boutons d'action
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("✅ Appliquer ce Lot Fixe")
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.setStyleSheet(
            "background-color: #16a34a; color: white; font-weight: bold; font-size: 13px; padding: 8px 18px; border-radius: 5px;"
        )
        self.apply_btn.clicked.connect(self._accept_selected)

        self.auto_btn = QPushButton("🤖 Repasser en Lot Dynamique (0.00)")
        self.auto_btn.setCursor(Qt.PointingHandCursor)
        self.auto_btn.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; font-size: 12px; padding: 8px 14px; border-radius: 5px;"
        )
        self.auto_btn.clicked.connect(self._accept_auto)

        cancel_btn = QPushButton("Annuler")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(
            "background-color: #475569; color: white; font-weight: bold; font-size: 12px; padding: 8px 14px; border-radius: 5px;"
        )
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.auto_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _on_params_changed(self):
        self.capital = max(5.0, float(self.cap_spin.value()))
        self.margin_free = max(5.0, float(self.mf_spin.value()))
        self._recalculate_table()

    def _calc_lot_for_risk_pct(self, risk_pct: float, loss_per_lot: float) -> float:
        import math
        budget = self.capital * (risk_pct / 100.0)
        raw = budget / max(1.0, loss_per_lot)
        steps = max(1, int(math.floor(raw / 0.01 + 1e-9)))
        return round(steps * 0.01, 2)

    def _recalculate_table(self):
        import math
        sl_pips = max(1.0, float(self.sl_pips_spin.value()))
        # Sur Forex standard (EURUSD, GBPUSD...), 1 lot = 10$/pip -> loss_per_lot = sl_pips * 10$
        loss_per_lot = sl_pips * 10.0
        # Marge estimée pour 1 lot (~108 000$ / levier)
        margin_per_lot = 108000.0 / max(30, self.leverage)
        max_safe_margin_lot = round(math.floor((self.margin_free * 0.70 / max(1.0, margin_per_lot)) / 0.01) * 0.01, 2)
        max_safe_risk_lot = round(math.floor(((self.capital * 0.32) / max(1.0, loss_per_lot)) / 0.01) * 0.01, 2)
        hard_ceiling_lot = max(0.01, min(max_safe_margin_lot, max_safe_risk_lot))

        # Calcul des paliers adaptés au capital
        lot_prudent = max(0.02, self._calc_lot_for_risk_pct(3.0, loss_per_lot))
        lot_equilibre = max(lot_prudent + 0.01, self._calc_lot_for_risk_pct(5.5, loss_per_lot))
        lot_dynamique = max(lot_equilibre + 0.01, self._calc_lot_for_risk_pct(10.0, loss_per_lot))
        lot_agressif = max(lot_dynamique + 0.02, self._calc_lot_for_risk_pct(18.0, loss_per_lot))

        # Plafonner intelligemment si le capital est très petit
        lot_prudent = min(lot_prudent, hard_ceiling_lot)
        lot_equilibre = min(lot_equilibre, hard_ceiling_lot)
        lot_dynamique = min(lot_dynamique, hard_ceiling_lot)
        lot_agressif = min(lot_agressif, hard_ceiling_lot)

        tiers = [
            ("🤖 Auto-Pilot (% Dynamique)", 0.00, "Mode dynamique (donne 0.01 sur petit capital)"),
            ("🟢 Prudent (Micro-Croissance)", lot_prudent, "✅ Passe Optimal (Très sûr)"),
            ("⭐ Équilibré (Recommandé)", lot_equilibre, "⭐ Passe Idéal (Meilleur compromis)"),
            ("🚀 Dynamique (Croissance Active)", lot_dynamique, "✅ Passe (Croissance rapide)"),
            ("🔥 Agressif (Boost Capital)", lot_agressif, "⚠️ Passe (Risque soutenu)"),
            ("⚡ Plafond Max Sécurisé", hard_ceiling_lot, "🛡️ Limite haute autorisée par le bot"),
        ]

        self._row_lots = []
        self.table.setRowCount(len(tiers))

        recommended_lot = lot_equilibre

        for row_idx, (label, lot_val, base_verdict) in enumerate(tiers):
            self._row_lots.append(lot_val)
            eff_lot = 0.01 if lot_val == 0.0 else lot_val
            risk_usd = eff_lot * loss_per_lot
            risk_pct = (risk_usd / self.capital) * 100.0 if self.capital > 0 else 0.0
            gain_usd = risk_usd * 2.0
            req_margin = eff_lot * margin_per_lot

            lot_txt = "0.00 (Auto → 0.01)" if lot_val == 0.0 else f"{lot_val:.2f} lot"
            risk_txt = f"{risk_usd:.2f} $ ({risk_pct:.1f}%)"
            gain_txt = f"+{gain_usd:.2f} $"
            margin_txt = f"~{req_margin:.2f} $"

            items = [
                QTableWidgetItem(label),
                QTableWidgetItem(lot_txt),
                QTableWidgetItem(risk_txt),
                QTableWidgetItem(gain_txt),
                QTableWidgetItem(margin_txt),
                QTableWidgetItem(base_verdict),
            ]
            for col_idx, it in enumerate(items):
                if row_idx == 2:  # Recommandé
                    it.setForeground(QColor("#4ade80"))
                    it.setFont(QFont("Arial", 10, QFont.Bold))
                elif row_idx == 1:
                    it.setForeground(QColor("#93c5fd"))
                elif row_idx >= 4:
                    it.setForeground(QColor("#fbbf24"))
                self.table.setItem(row_idx, col_idx, it)

        # Initialiser la valeur du spinbox si pas encore définie
        if self.selected_lot_spin.value() == 0.0:
            init_val = self.current_fixed_lot if self.current_fixed_lot > 0.0 else recommended_lot
            self.selected_lot_spin.setValue(init_val)
            self.table.selectRow(2)
        else:
            self._update_custom_verdict()

    def _on_row_clicked(self, row: int, _col: int):
        if 0 <= row < len(getattr(self, '_row_lots', [])):
            self.selected_lot_spin.setValue(self._row_lots[row])

    def _on_row_double_clicked(self, row: int, _col: int):
        if 0 <= row < len(getattr(self, '_row_lots', [])):
            self.selected_lot_spin.setValue(self._row_lots[row])
            self._accept_selected()

    def _update_custom_verdict(self):
        lot = float(self.selected_lot_spin.value())
        sl_pips = max(1.0, float(self.sl_pips_spin.value()))
        loss_per_lot = sl_pips * 10.0
        margin_per_lot = 108000.0 / max(30, self.leverage)

        if lot <= 0.0:
            self.live_verdict_lbl.setText(
                "🤖 <b style='color:#60a5fa;'>Mode Auto-Pilot (0.00) :</b> Calcul par % de risque "
                "(ouvrira <b>0.01 lot</b> sur un petit capital)."
            )
            return

        risk_usd = lot * loss_per_lot
        risk_pct = (risk_usd / self.capital) * 100.0 if self.capital > 0 else 0.0
        req_margin = lot * margin_per_lot

        if req_margin > self.margin_free * 0.75:
            self.live_verdict_lbl.setText(
                f"🚫 <b style='color:#f87171;'>Marge trop élevée (~{req_margin:.2f}$ > 75% de {self.margin_free:.2f}$) :</b> "
                f"Le bot réduira automatiquement ce lot pour éviter le Margin Call."
            )
        elif risk_pct > 34.0:
            self.live_verdict_lbl.setText(
                f"⚠️ <b style='color:#fb923c;'>Risque élevé ({risk_usd:.2f}$ = {risk_pct:.1f}% du capital) :</b> "
                f"Dépasse le plafond de sécurité (34%). Le bot réduira automatiquement le lot sur les gros Stop Loss."
            )
        elif risk_pct <= 7.5:
            self.live_verdict_lbl.setText(
                f"✅ <b style='color:#4ade80;'>PASSE OPTIMAL :</b> Risque estimé <b>{risk_usd:.2f}$ ({risk_pct:.1f}%)</b> "
                f"| Gain R:R 1:2 : <b>+{risk_usd*2:.2f}$</b> | Marge : <b>~{req_margin:.2f}$</b>."
            )
        else:
            self.live_verdict_lbl.setText(
                f"⚡ <b style='color:#facc15;'>PASSE (Dynamique) :</b> Risque estimé <b>{risk_usd:.2f}$ ({risk_pct:.1f}%)</b> "
                f"| Gain R:R 1:2 : <b>+{risk_usd*2:.2f}$</b> | Marge : <b>~{req_margin:.2f}$</b>."
            )

    def _accept_selected(self):
        self.selected_lot = round(float(self.selected_lot_spin.value()), 2)
        self.accept()

    def _accept_auto(self):
        self.selected_lot = 0.0
        self.accept()


class ConfigDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("⚙️ Paramètres")
        self.setMinimumSize(400, 300)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # --- Onglet Risque ---
        risk_tab = QWidget()
        risk_layout = QFormLayout(risk_tab)

        # Profil de risque prédéfini
        self.risk_profile_combo = QComboBox()
        self.risk_profile_combo.addItem("⚖️ Équilibré (Standard - 1.0% / jour +3.0%)", "BALANCED")
        self.risk_profile_combo.addItem("🛡️ Prudent (Conservateur - 0.5% / jour +1.5%)", "CONSERVATIVE")
        self.risk_profile_combo.addItem("🚀 Dynamique (Agressif - 2.0% / jour +5.0%)", "AGGRESSIVE")
        self.risk_profile_combo.addItem("🎯 Sniper (Ultra-Sélectif - 2.5% / Confluence A+)", "SNIPER")
        self.risk_profile_combo.addItem("⚙️ Personnalisé (Manuel)", "CUSTOM")
        curr_prof = self.config.get('risk_management', {}).get('risk_profile', 'BALANCED')
        idx_p = self.risk_profile_combo.findData(curr_prof)
        if idx_p >= 0:
            self.risk_profile_combo.setCurrentIndex(idx_p)
        self.risk_profile_combo.currentIndexChanged.connect(self.on_profile_changed)

        self.auto_tier_cb = QCheckBox("Adapter selon le capital (Tiers Micro, Standard, Pro)")
        self.auto_tier_cb.setChecked(self.config.get('risk_management', {}).get('auto_tier_enabled', True))
        self.auto_tier_cb.setToolTip("Ajuste les quotas de positions et la protection du lot 0.01 selon la taille réelle du compte MT5.")

        self.manual_max_pos_cb = QCheckBox("Forcer Max Positions (ignorer le plafond strict du palier de capital)")
        self.manual_max_pos_cb.setChecked(self.config.get('risk_management', {}).get('manual_max_positions_override', True))
        self.manual_max_pos_cb.setToolTip(
            "Si coché, le nombre configuré dans 'Max Positions' est toujours respecté,\n"
            "même si le palier de capital (ex: Micro < 250$) recommande moins de positions."
        )

        self.daily_profit_pct_spin = QDoubleSpinBox()
        self.daily_profit_pct_spin.setRange(0.0, 50.0)
        self.daily_profit_pct_spin.setSingleStep(0.5)
        self.daily_profit_pct_spin.setSuffix(" %")
        self.daily_profit_pct_spin.setValue(float(self.config.get('risk_management', {}).get('daily_profit_target_percent', 3.0)))
        self.daily_profit_pct_spin.setToolTip("Objectif en % du solde par jour (ex: 3.0%). Si mis à 0, utilise le montant fixe en $ ci-dessous.")

        self.max_micro_risk_spin = QDoubleSpinBox()
        self.max_micro_risk_spin.setRange(1.0, 20.0)
        self.max_micro_risk_spin.setSingleStep(0.5)
        self.max_micro_risk_spin.setSuffix(" %")
        self.max_micro_risk_spin.setValue(float(self.config.get('risk_management', {}).get('max_micro_risk_cap_percent', 4.0)))
        self.max_micro_risk_spin.setToolTip("Plafond de risque toléré sur micro-compte quand le broker impose le lot min (0.01). Au-delà, le trade est annulé pour préserver le compte.")
        
        self.risk_pct_spin = QDoubleSpinBox()
        self.risk_pct_spin.setRange(0.1, 100.0)
        self.risk_pct_spin.setSingleStep(0.1)
        self.risk_pct_spin.setSuffix(" %")
        self.risk_pct_spin.setValue(self.config.get('risk_management', {}).get('default_risk_percent', 1.0))

        self.fixed_lot_size_spin = QDoubleSpinBox()
        self.fixed_lot_size_spin.setRange(0.0, 100.0)
        self.fixed_lot_size_spin.setSingleStep(0.01)
        self.fixed_lot_size_spin.setDecimals(2)
        init_fixed_lot = float(self.config.get('risk_management', {}).get('fixed_lot_size', 0.0))
        self.fixed_lot_size_spin.setValue(init_fixed_lot)
        self.fixed_lot_size_spin.setToolTip("Volume fixe (ex: 0.04). Si 0.00, le robot calcule le volume en Auto-Pilot selon le % de risque.")

        self.lot_mode_combo = QComboBox()
        self.lot_mode_combo.addItem("🤖 Auto-Pilot (% du Capital)", "AUTO")
        self.lot_mode_combo.addItem("⚖️ Lot Fixe (Calibré Capital)", "FIXED")
        self.lot_mode_combo.setCurrentIndex(1 if init_fixed_lot > 0.0 else 0)
        self.lot_mode_combo.currentIndexChanged.connect(self._on_lot_mode_toggled)

        self.lot_advisor_btn = QPushButton("💡 Conseiller Lot selon Capital")
        self.lot_advisor_btn.setCursor(Qt.PointingHandCursor)
        self.lot_advisor_btn.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        )
        self.lot_advisor_btn.setToolTip("Ouvre une boîte de dialogue qui calcule quel Lot Fixe passe le mieux selon votre capital MT5.")
        self.lot_advisor_btn.clicked.connect(self.open_lot_advisor_dialog)

        fixed_lot_row = QHBoxLayout()
        fixed_lot_row.setSpacing(6)
        fixed_lot_row.addWidget(self.lot_mode_combo)
        fixed_lot_row.addWidget(self.fixed_lot_size_spin)
        fixed_lot_row.addWidget(self.lot_advisor_btn)
        
        
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(0.1, 100.0)
        self.daily_loss_spin.setSingleStep(0.5)
        self.daily_loss_spin.setSuffix(" %")
        self.daily_loss_spin.setValue(self.config.get('risk_management', {}).get('max_daily_loss_percent', 5.0))

        self.tp_target_spin = QDoubleSpinBox()
        self.tp_target_spin.setRange(0.0, 1000.0)
        self.tp_target_spin.setSingleStep(0.5)
        self.tp_target_spin.setDecimals(2)
        self.tp_target_spin.setSuffix(" $")
        self.tp_target_spin.setValue(float(self.config.get('risk_management', {}).get('take_profit_target_per_trade_usd', 4.0)))

        self.giveback_spin = QDoubleSpinBox()
        self.giveback_spin.setRange(0.1, 50.0)
        self.giveback_spin.setSingleStep(0.1)
        self.giveback_spin.setDecimals(2)
        self.giveback_spin.setSuffix(" $")
        self.giveback_spin.setValue(float(self.config.get('risk_management', {}).get('giveback_usd_small', 0.6)))

        self.daily_profit_spin = QDoubleSpinBox()
        self.daily_profit_spin.setRange(0.0, 5000.0)
        self.daily_profit_spin.setSingleStep(5.0)
        self.daily_profit_spin.setDecimals(2)
        self.daily_profit_spin.setSuffix(" $")
        self.daily_profit_spin.setValue(float(self.config.get('risk_management', {}).get('daily_profit_target_usd', 50.0)))
        
        self.exclude_closed_cb = QCheckBox("Ne compter que les marchés ouverts dans le quota")
        self.exclude_closed_cb.setChecked(self.config.get('risk_management', {}).get('exclude_closed_markets_from_quota', True))
        self.exclude_closed_cb.setToolTip("Permet de trader les marchés 24/7 (ex: Crypto) même si 3 positions Forex sont gelées sur des marchés fermés.")

        self.spread_buffer_cb = QCheckBox("Activer le Spread Buffer (élargit SL et réduit le lot)")
        self.spread_buffer_cb.setChecked(self.config.get('risk_management', {}).get('spread_buffer_enabled', True))
        self.spread_buffer_cb.setToolTip("En cas de spread élevé, adapte automatiquement le Stop Loss et réduit le volume du lot pour conserver le même risque en dollars.")

        self.spread_ratio_spin = QDoubleSpinBox()
        self.spread_ratio_spin.setRange(5.0, 50.0)
        self.spread_ratio_spin.setSingleStep(5.0)
        self.spread_ratio_spin.setDecimals(1)
        self.spread_ratio_spin.setSuffix(" %")
        curr_ratio = float(self.config.get('risk_management', {}).get('max_spread_sl_ratio', 0.25)) * 100.0
        self.spread_ratio_spin.setValue(curr_ratio)
        self.spread_ratio_spin.setToolTip("Tolérance maximale du spread par rapport au Stop Loss (défaut: 25%). Au-delà, le Spread Buffer adapte le trade.")

        risk_layout.addRow("🎯 Profil de Risque :", self.risk_profile_combo)
        risk_layout.addRow("💎 Paliers de Capital :", self.auto_tier_cb)
        risk_layout.addRow("📊 Forcer Max Positions :", self.manual_max_pos_cb)
        risk_layout.addRow("Risque par trade :", self.risk_pct_spin)
        risk_layout.addRow("⚖️ Mode Volume / Lot Fixe :", fixed_lot_row)
        risk_layout.addRow("Perte max journalière :", self.daily_loss_spin)
        risk_layout.addRow("🏁 Objectif gain jour (%) :", self.daily_profit_pct_spin)
        risk_layout.addRow("🏁 Objectif gain jour ($ fixe) :", self.daily_profit_spin)
        risk_layout.addRow("🛡️ Plafond risque Micro (0.01 lot) :", self.max_micro_risk_spin)
        risk_layout.addRow("🎯 TP Fixe / Trade ($) :", self.tp_target_spin)
        risk_layout.addRow("🛡️ Recul toléré depuis pic ($) :", self.giveback_spin)
        risk_layout.addRow("🛡️ Spread Buffer Adaptatif :", self.spread_buffer_cb)
        risk_layout.addRow("📊 Seuil Max Spread/SL :", self.spread_ratio_spin)
        risk_layout.addRow("Marchés fermés :", self.exclude_closed_cb)
        tabs.addTab(risk_tab, "💰 Gestion du Risque")
        
        # --- Onglet Marchés ---
        market_tab = QWidget()
        market_layout = QVBoxLayout(market_tab)
        
        dir_box = QHBoxLayout()
        dir_box.addWidget(QLabel("Sens de trading autorisé :"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("🔄 Mixte (BUY & SELL)", "BOTH")
        self.direction_combo.addItem("📈 Hausse Seule (BUY)", "BUY_ONLY")
        self.direction_combo.addItem("📉 Baisse Seule (SELL)", "SELL_ONLY")
        curr_dir = self.config.get('trading', {}).get('allowed_trade_direction', 'BOTH')
        idx = self.direction_combo.findData(curr_dir)
        if idx >= 0:
            self.direction_combo.setCurrentIndex(idx)
        dir_box.addWidget(self.direction_combo)
        market_layout.addLayout(dir_box)

        self.market_checkboxes = {}
        market_types = ["Forex", "Crypto", "Indices", "Metals", "Energies", "Stocks"]
        allowed = self.config.get('trading', {}).get('allowed_market_types', market_types)
        
        market_layout.addWidget(QLabel("Types de marchés autorisés :"))
        for m_type in market_types:
            cb = QCheckBox(m_type)
            cb.setChecked(m_type in allowed)
            market_layout.addWidget(cb)
            self.market_checkboxes[m_type] = cb
            
        market_layout.addStretch()
        tabs.addTab(market_tab, "📈 Types de Marchés")

        # --- Onglet CRT & Order Flow ---
        crt_of_tab = QWidget()
        crt_of_layout = QFormLayout(crt_of_tab)

        # CRT
        self.crt_enabled_cb = QCheckBox("Activer la Stratégie CRT (Candle Range Theory)")
        self.crt_enabled_cb.setChecked(self.config.get('crt', {}).get('enabled', True))

        self.crt_order_mode_combo = QComboBox()
        self.crt_order_mode_combo.addItem("🔄 Tous les ordres (Marché + Pending Limits)", "both")
        self.crt_order_mode_combo.addItem("⚡ Ordres simples au marché uniquement", "market")
        self.crt_order_mode_combo.addItem("📌 Pending Limits (Ordres en attente uniquement)", "pending")
        curr_crt_mode = str(self.config.get('crt', {}).get('order_mode', 'both')).lower()
        idx_crt = self.crt_order_mode_combo.findData(curr_crt_mode)
        if idx_crt >= 0:
            self.crt_order_mode_combo.setCurrentIndex(idx_crt)

        self.crt_min_rr_spin = QDoubleSpinBox()
        self.crt_min_rr_spin.setRange(1.0, 10.0)
        self.crt_min_rr_spin.setSingleStep(0.25)
        self.crt_min_rr_spin.setDecimals(2)
        self.crt_min_rr_spin.setValue(float(self.config.get('crt', {}).get('min_rr', 1.5)))

        self.crt_expiration_spin = QSpinBox()
        self.crt_expiration_spin.setRange(1, 48)
        self.crt_expiration_spin.setSuffix(" bougies")
        self.crt_expiration_spin.setValue(int(self.config.get('crt', {}).get('expiration_bars', 4)))

        # Order Flow
        self.of_enabled_cb = QCheckBox("Activer le Moteur Order Flow (DOM L2, Heatmap, CVD, VP)")
        self.of_enabled_cb.setChecked(self.config.get('order_flow', {}).get('enabled', True))

        self.of_require_conf_cb = QCheckBox("Exiger confirmation stricte Order Flow avant prise de position")
        self.of_require_conf_cb.setChecked(self.config.get('order_flow', {}).get('require_confirmation', False))
        self.of_require_conf_cb.setToolTip("Si coché, aucun trade CRT n'est pris si le score composite Order Flow est inférieur au seuil.")

        self.of_min_score_spin = QDoubleSpinBox()
        self.of_min_score_spin.setRange(20.0, 100.0)
        self.of_min_score_spin.setSingleStep(5.0)
        self.of_min_score_spin.setSuffix(" / 100")
        self.of_min_score_spin.setValue(float(self.config.get('order_flow', {}).get('min_confluence_score', 60.0)))

        self.of_imbalance_spin = QDoubleSpinBox()
        self.of_imbalance_spin.setRange(1.5, 10.0)
        self.of_imbalance_spin.setSingleStep(0.5)
        self.of_imbalance_spin.setValue(float(self.config.get('order_flow', {}).get('imbalance_ratio', 3.0)))

        crt_of_layout.addRow("🎯 CRT Activé :", self.crt_enabled_cb)
        crt_of_layout.addRow("🔄 Mode d'Ordre CRT :", self.crt_order_mode_combo)
        crt_of_layout.addRow("⚖️ Ratio R:R Minimum :", self.crt_min_rr_spin)
        crt_of_layout.addRow("⏳ Expiration Ordres Limites :", self.crt_expiration_spin)
        crt_of_layout.addRow("🌊 Moteur Order Flow :", self.of_enabled_cb)
        crt_of_layout.addRow("🛡️ Confirmation Stricte :", self.of_require_conf_cb)
        crt_of_layout.addRow("📊 Score Confluence Min :", self.of_min_score_spin)
        crt_of_layout.addRow("⚡ Ratio Déséquilibre Footprint :", self.of_imbalance_spin)
        # Outils de diagnostic MT5 et A/B
        tools_box = QHBoxLayout()
        diag_btn = QPushButton("🩺 Tester la Qualité des Flux MT5 Réels")
        diag_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 4px; border-radius: 4px;")
        def _run_diag():
            p = self.parent()
            if not p or not hasattr(p, 'market_data'):
                QMessageBox.warning(self, "Diagnostic", "Module MarketData non disponible.")
                return
            sym = "EURUSD"
            if hasattr(p, 'active_symbol') and p.active_symbol:
                sym = p.active_symbol
            elif hasattr(p, 'symbol_combo') and p.symbol_combo.currentText():
                sym = p.symbol_combo.currentText()
            diag = p.market_data.diagnose_market_data(sym)
            msg = (
                f"Diagnostic MT5 pour {sym} :\n\n"
                f"• Niveau Courtier : {diag['tier']}\n"
                f"• Qualité Flux : {diag['data_quality']}\n"
                f"• Carnet DOM L2 : {'Disponible (' + str(diag['dom_levels']) + ' niveaux)' if diag['dom_available'] else 'Non supporté par le broker'}\n"
                f"• Fraîcheur Ticks : {'Récent (' + str(round(diag['ticks_stale_seconds'], 1)) + 's)' if diag['ticks_fresh'] else 'Périmé (' + str(round(diag['ticks_stale_seconds'], 1)) + 's)'}\n"
                f"• Ticks Réels Observés : {round(diag['real_flags_ratio']*100, 1)}%\n\n"
                f"Recommandation :\n{diag['recommendation']}"
            )
            QMessageBox.information(self, f"Audit Flux MT5 - {sym}", msg)
        diag_btn.clicked.connect(_run_diag)
        tools_box.addWidget(diag_btn)

        ab_btn = QPushButton("📊 Rapport Comparatif A/B")
        ab_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 4px; border-radius: 4px;")
        def _show_ab():
            p = self.parent()
            if not p or not hasattr(p, 'order_flow_journal'):
                QMessageBox.warning(self, "Rapport A/B", "Journal Order Flow non disponible.")
                return
            res = p.order_flow_journal.compare_ab_performance()
            QMessageBox.information(self, "Comparaison A/B : CRT vs CRT+OF", res.get('markdown_report', 'Données insuffisantes.'))
        ab_btn.clicked.connect(_show_ab)
        tools_box.addWidget(ab_btn)

        crt_of_layout.addRow("🧪 Outils de Validation :", tools_box)

        tabs.addTab(crt_of_tab, "🎯 CRT & Order Flow")
        
        # Boutons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Sauvegarder")
        save_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 5px;")
        save_btn.clicked.connect(self.save_config)
        cancel_btn = QPushButton("Annuler")
        cancel_btn.setStyleSheet("background-color: #f44336; font-weight: bold; padding: 5px;")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
    def on_profile_changed(self):
        prof = self.risk_profile_combo.currentData()
        if prof == "CONSERVATIVE":
            self.risk_pct_spin.setValue(0.5)
            self.daily_loss_spin.setValue(3.0)
            self.daily_profit_pct_spin.setValue(1.5)
        elif prof == "BALANCED":
            self.risk_pct_spin.setValue(1.0)
            self.daily_loss_spin.setValue(5.0)
            self.daily_profit_pct_spin.setValue(3.0)
        elif prof == "AGGRESSIVE":
            self.risk_pct_spin.setValue(2.0)
            self.daily_loss_spin.setValue(8.0)
            self.daily_profit_pct_spin.setValue(5.0)
        elif prof == "SNIPER":
            self.risk_pct_spin.setValue(2.5)
            self.daily_loss_spin.setValue(6.0)
            self.daily_profit_pct_spin.setValue(6.0)

    def _on_lot_mode_toggled(self, _index: int):
        mode = self.lot_mode_combo.currentData()
        if mode == "AUTO":
            self.fixed_lot_size_spin.setValue(0.0)
        elif mode == "FIXED":
            # Ouvre automatiquement la boîte de dialogue pour voir quel lot fixe passe le mieux selon le capital
            self.open_lot_advisor_dialog()

    def open_lot_advisor_dialog(self):
        cap = 81.03
        mf = 81.03
        lev = 500
        p = self.parent()
        try:
            if p and hasattr(p, 'orchestrator') and hasattr(p.orchestrator, 'mt5'):
                acc = p.orchestrator.mt5.get_account_info() or {}
                bal = float(acc.get('balance', 0.0) or 0.0)
                eq = float(acc.get('equity', bal) or bal)
                if bal > 0:
                    cap = min(bal, eq) if eq > 0 else bal
                    mf = float(acc.get('margin_free', cap) or cap)
                    lev = int(acc.get('leverage', 500) or 500)
        except Exception:
            pass

        dlg = FixedLotAdvisorDialog(
            capital=cap,
            margin_free=mf,
            leverage=lev,
            current_fixed_lot=self.fixed_lot_size_spin.value(),
            parent=self,
        )
        if dlg.exec():
            chosen = round(float(dlg.selected_lot), 2)
            self.fixed_lot_size_spin.setValue(chosen)
            self.lot_mode_combo.blockSignals(True)
            self.lot_mode_combo.setCurrentIndex(1 if chosen > 0.0 else 0)
            self.lot_mode_combo.blockSignals(False)

    def save_config(self):
        if 'risk_management' not in self.config:
            self.config['risk_management'] = {}
        self.config['risk_management']['risk_profile'] = self.risk_profile_combo.currentData()
        self.config['risk_management']['auto_tier_enabled'] = self.auto_tier_cb.isChecked()
        self.config['risk_management']['manual_max_positions_override'] = self.manual_max_pos_cb.isChecked()
        self.config['risk_management']['daily_profit_target_percent'] = self.daily_profit_pct_spin.value()
        self.config['risk_management']['max_micro_risk_cap_percent'] = self.max_micro_risk_spin.value()
        self.config['risk_management']['default_risk_percent'] = self.risk_pct_spin.value()
        self.config['risk_management']['fixed_lot_size'] = self.fixed_lot_size_spin.value()
        self.config['risk_management']['max_daily_loss_percent'] = self.daily_loss_spin.value()
        self.config['risk_management']['take_profit_target_per_trade_usd'] = self.tp_target_spin.value()
        self.config['risk_management']['giveback_usd_small'] = self.giveback_spin.value()
        self.config['risk_management']['daily_profit_target_usd'] = self.daily_profit_spin.value()
        self.config['risk_management']['exclude_closed_markets_from_quota'] = self.exclude_closed_cb.isChecked()
        self.config['risk_management']['spread_buffer_enabled'] = self.spread_buffer_cb.isChecked()
        self.config['risk_management']['max_spread_sl_ratio'] = round(self.spread_ratio_spin.value() / 100.0, 3)
        
        allowed = [name for name, cb in self.market_checkboxes.items() if cb.isChecked()]
        if 'trading' not in self.config:
            self.config['trading'] = {}
        self.config['trading']['allowed_market_types'] = allowed
        self.config['trading']['allowed_trade_direction'] = self.direction_combo.currentData()

        # Sauvegarde CRT
        if 'crt' not in self.config:
            self.config['crt'] = {}
        self.config['crt']['enabled'] = self.crt_enabled_cb.isChecked()
        self.config['crt']['order_mode'] = self.crt_order_mode_combo.currentData()
        self.config['crt']['use_pending_orders'] = (self.crt_order_mode_combo.currentData() != 'market')
        self.config['crt']['min_rr'] = self.crt_min_rr_spin.value()
        self.config['crt']['expiration_bars'] = self.crt_expiration_spin.value()

        # Sauvegarde Order Flow
        if 'order_flow' not in self.config:
            self.config['order_flow'] = {}
        self.config['order_flow']['enabled'] = self.of_enabled_cb.isChecked()
        self.config['order_flow']['require_confirmation'] = self.of_require_conf_cb.isChecked()
        self.config['order_flow']['min_confluence_score'] = self.of_min_score_spin.value()
        self.config['order_flow']['imbalance_ratio'] = self.of_imbalance_spin.value()
        
        try:
            from pathlib import Path
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
            QMessageBox.information(self, "Succès", "Configuration sauvegardée avec succès !")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde : {e}")

class MemoryDialog(QDialog):
    """
    Dialogue de la Conscience Évolutive :
    - Visualisation des Poids Adaptatifs dynamiques calibrés par le bot.
    - Journal des trades avec auto-critique post-mortem et leçons apprises.
    - Statistiques de performance et d'apprentissage empirique.
    """
    def __init__(self, trade_memory=None, weights_engine=None, parent=None):
        super().__init__(parent)
        self.trade_memory = trade_memory
        self.weights_engine = weights_engine
        self.setWindowTitle("🧠 Conscience Évolutive — Mémoire & Auto-Critique")
        self.resize(950, 680)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # En-tête explicatif
        header = QLabel(
            "<h3>🧠 Conscience Évolutive & Système Réflexif</h3>"
            "Le bot analyse chacun de ses trades clôturés, rédige une auto-critique, "
            "enregistre des leçons pour ne plus répéter les mêmes erreurs, "
            "et recalibre dynamiquement ses poids de scoring quantitatif."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #cfd8dc; background: #263238; padding: 12px; border-radius: 6px;")
        layout.addWidget(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Onglet 1 : Poids Adaptatifs ──
        tab_weights = QWidget()
        lay_weights = QVBoxLayout(tab_weights)

        info_w = QLabel(
            "<b>Poids Actifs de Scoring Institutionnel :</b><br>"
            "Ces coefficients remplacent les règles rigides. Le bot amplifie l'importance "
            "des stratégies qui gagnent et réduit celles qui échouent, dans un cadre sécurisé."
        )
        info_w.setStyleSheet("color: #90caf9; margin-bottom: 6px;")
        lay_weights.addWidget(info_w)

        self.table_weights = QTableWidget()
        self.table_weights.setColumnCount(4)
        self.table_weights.setHorizontalHeaderLabels(["Composante Institutionnelle", "Poids Actuel (Calibré)", "Poids d'Usine", "Évolution"])
        self.table_weights.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_weights.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_weights.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_weights.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        lay_weights.addWidget(self.table_weights)

        btn_row_w = QHBoxLayout()
        self.btn_recalib = QPushButton("🔄 Recalibrer d'après l'historique récent")
        self.btn_recalib.setStyleSheet("background-color: #00897b; color: white; font-weight: bold; padding: 8px;")
        self.btn_recalib.clicked.connect(self.on_recalibrate_clicked)
        self.btn_reset_w = QPushButton("↩️ Réinitialiser Poids Usine")
        self.btn_reset_w.setStyleSheet("background-color: #546e7a; color: white; padding: 8px;")
        self.btn_reset_w.clicked.connect(self.on_reset_weights_clicked)
        btn_row_w.addWidget(self.btn_recalib)
        btn_row_w.addWidget(self.btn_reset_w)
        lay_weights.addLayout(btn_row_w)

        self.tabs.addTab(tab_weights, "📊 Poids Adaptatifs")

        # ── Onglet 2 : Journal Réflexif & Auto-Critique ──
        tab_journal = QWidget()
        lay_journal = QVBoxLayout(tab_journal)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #a7ffeb; font-weight: bold; background: #1c313a; padding: 8px; border-radius: 4px;")
        lay_journal.addWidget(self.stats_label)

        self.table_journal = QTableWidget()
        self.table_journal.setColumnCount(8)
        self.table_journal.setHorizontalHeaderLabels(["Date", "Ticket", "Symbole", "Sens", "Lot", "PnL ($)", "Sortie", "Leçon Retenue"])
        self.table_journal.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.table_journal.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_journal.itemSelectionChanged.connect(self.on_trade_selected)
        lay_journal.addWidget(self.table_journal)

        lbl_detail = QLabel("<b>Détail du Post-Mortem & Contexte d'Entrée :</b>")
        lbl_detail.setStyleSheet("color: #b0bec5; margin-top: 4px;")
        lay_journal.addWidget(lbl_detail)

        self.critique_text = QTextEdit()
        self.critique_text.setReadOnly(True)
        self.critique_text.setMaximumHeight(140)
        self.critique_text.setStyleSheet("background-color: #1a1a24; color: #eceff1; font-family: monospace; font-size: 11px;")
        lay_journal.addWidget(self.critique_text)

        self.tabs.addTab(tab_journal, "📖 Journal & Auto-Critique")

        # ── Onglet 3 : Leçons Actives en Mémoire ──
        tab_lessons = QWidget()
        lay_lessons = QVBoxLayout(tab_lessons)
        lbl_l = QLabel("<b>Règles négatives et leçons actuellement injectées dans l'IA avant chaque trade :</b>")
        lbl_l.setStyleSheet("color: #ffcc80;")
        lay_lessons.addWidget(lbl_l)

        self.lessons_text = QTextEdit()
        self.lessons_text.setReadOnly(True)
        self.lessons_text.setStyleSheet("background-color: #1a1a24; color: #fff59d; font-family: monospace; font-size: 12px; line-height: 1.4;")
        lay_lessons.addWidget(self.lessons_text)
        self.tabs.addTab(tab_lessons, "💡 Leçons en Mémoire")

        # Bouton fermer
        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.refresh_all()

    def refresh_all(self):
        self.refresh_weights()
        self.refresh_journal()
        self.refresh_lessons()

    def refresh_weights(self):
        if not self.weights_engine:
            return
        weights = self.weights_engine.get_weights()
        from src.learning.adaptive_weights import DEFAULT_WEIGHTS
        labels = {
            "ob_weight": "Order Block Retesté (SMC)",
            "boomerang_weight": "Boomerang Setup (Breakout/Retest)",
            "trend_ema_weight": "Tendance Moyennes Mobiles (EMA 50/200)",
            "rsi_pullback_weight": "Pullback Sain (RSI 42-60)",
            "htf_alignment_weight": "Alignement Tendance Fond (HTF)",
            "amd_expansion_weight": "Phase AMD Expansion & FVG",
        }
        self.table_weights.setRowCount(len(labels))
        for r, (key, label) in enumerate(labels.items()):
            cur = float(weights.get(key, DEFAULT_WEIGHTS.get(key, 0)))
            def_w = float(DEFAULT_WEIGHTS.get(key, 0))
            
            diff = cur - def_w
            if diff > 0.5:
                status = f"↗️ +{diff:.1f} (Amplifié)"
                status_color = QColor("#00e676")
            elif diff < -0.5:
                status = f"↘️ {diff:.1f} (Pénalisé)"
                status_color = QColor("#ff5252")
            else:
                status = "➡️ Neutre"
                status_color = QColor("#bdbdbd")

            item_lbl = QTableWidgetItem(label)
            item_cur = QTableWidgetItem(f"{cur:.1f}")
            item_cur.setTextAlignment(Qt.AlignCenter)
            item_cur.setForeground(QColor("#00e676") if cur >= def_w else QColor("#ffb74d"))
            item_def = QTableWidgetItem(f"{def_w:.1f}")
            item_def.setTextAlignment(Qt.AlignCenter)
            item_stat = QTableWidgetItem(status)
            item_stat.setTextAlignment(Qt.AlignCenter)
            item_stat.setForeground(status_color)

            self.table_weights.setItem(r, 0, item_lbl)
            self.table_weights.setItem(r, 1, item_cur)
            self.table_weights.setItem(r, 2, item_def)
            self.table_weights.setItem(r, 3, item_stat)

    def refresh_journal(self):
        if not self.trade_memory:
            self.stats_label.setText("Mémoire épisodique non initialisée.")
            return

        stats = self.trade_memory.get_summary_stats()
        total_c = stats.get('total_closed', 0)
        wins_c = stats.get('wins_count', 0)
        losses_c = stats.get('losses_count', 0)
        win_rate = stats.get('win_rate', 0.0)
        total_p = stats.get('total_profit_usd', 0.0)
        best_t = stats.get('best_trade_usd', 0.0)
        worst_t = stats.get('worst_trade_usd', 0.0)
        self.stats_label.setText(
            f"📊 Total Trades: {total_c} | "
            f"Gagnants: {wins_c} | Perdants: {losses_c} | "
            f"Taux de réussite: {win_rate:.1f}% | "
            f"Gain net total: {total_p:+.2f} $ | "
            f"Meilleur: {best_t:+.2f} $ | Pire: {worst_t:+.2f} $"
        )

        closed = self.trade_memory.get_closed_trades(limit=50)
        self.table_journal.setRowCount(len(closed))
        self._cached_closed = closed

        for r, t in enumerate(closed):
            profit = float(t.get("final_profit_usd", 0.0) or 0.0)
            p_color = QColor("#00e676") if profit > 0 else (QColor("#ff5252") if profit < 0 else QColor("#e0e0e0"))

            date_str = str(t.get("close_time_str") or t.get("open_time_str") or "")
            self.table_journal.setItem(r, 0, QTableWidgetItem(date_str))
            self.table_journal.setItem(r, 1, QTableWidgetItem(str(t.get("ticket", ""))))
            self.table_journal.setItem(r, 2, QTableWidgetItem(str(t.get("symbol", ""))))
            self.table_journal.setItem(r, 3, QTableWidgetItem(str(t.get("action", ""))))
            self.table_journal.setItem(r, 4, QTableWidgetItem(str(t.get("lot", ""))))

            p_item = QTableWidgetItem(f"{profit:+.2f} $")
            p_item.setForeground(p_color)
            p_item.setTextAlignment(Qt.AlignCenter)
            self.table_journal.setItem(r, 5, p_item)

            self.table_journal.setItem(r, 6, QTableWidgetItem(str(t.get("exit_reason", ""))))
            self.table_journal.setItem(r, 7, QTableWidgetItem(str(t.get("lesson") or "En cours d'analyse...")))

    def on_trade_selected(self):
        rows = self.table_journal.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if hasattr(self, '_cached_closed') and idx < len(self._cached_closed):
            t = self._cached_closed[idx]
            critique = t.get("critique", "Aucune auto-critique enregistrée.")
            lesson = t.get("lesson", "")
            tags = ", ".join(t.get("tags", []))
            snap = t.get("snapshot", {}) or {}
            snap_str = json.dumps(snap, indent=2, ensure_ascii=False) if snap else "Non disponible"

            self.critique_text.setPlainText(
                f"=== POST-MORTEM DU TRADE #{t.get('ticket')} ({t.get('symbol')} {t.get('action')}) ===\n"
                f"Tags: [{tags}]\n"
                f"Pic max atteint: {t.get('max_profit_usd', 0.0):+.2f} $ | Pire recul: {t.get('max_drawdown_usd', 0.0):+.2f} $\n\n"
                f"{critique}\n\n"
                f"LEÇON: {lesson}\n\n"
                f"--- Contexte marché à l'entrée ---\n{snap_str}"
            )

    def refresh_lessons(self):
        if not self.trade_memory:
            return
        lessons = self.trade_memory.get_recent_lessons(limit=15)
        if not lessons:
            self.lessons_text.setPlainText("Aucune leçon enregistrée pour l'instant. Les leçons apparaîtront dès la clôture des premiers trades.")
        else:
            txt = "=== BASE DE RÈGLES APPRISES EN TEMPS RÉEL ===\n\n"
            for i, l in enumerate(lessons, 1):
                txt += f"{i}. {l}\n\n"
            self.lessons_text.setPlainText(txt)

    def on_recalibrate_clicked(self):
        if not self.weights_engine or not self.trade_memory:
            return
        closed = self.trade_memory.get_closed_trades()
        res = self.weights_engine.recalibrate(closed)
        if res.get("status") == "RECALIBRATED":
            QMessageBox.information(
                self, "Recalibrage réussi",
                f"Les poids ont été recalibrés avec succès d'après les {res.get('sample_size')} derniers trades !"
            )
        else:
            QMessageBox.information(
                self, "Information",
                f"Recalibrage ignoré : {res.get('reason')}"
            )
        self.refresh_weights()

    def on_reset_weights_clicked(self):
        if not self.weights_engine:
            return
        self.weights_engine.reset_to_defaults()
        QMessageBox.information(self, "Réinitialisation", "Les poids ont été réinitialisés aux valeurs d'usine.")
        self.refresh_weights()


class MainWindow(QMainWindow):
    def __init__(self, orchestrator, trade_execution, chat_engine, config, trade_manager, trade_memory=None, weights_engine=None):
        super().__init__()
        from src.core.state_machine import StateManager, BotState
        from src.core.watchdog import Watchdog
        
        self.state_manager = StateManager()
        self.watchdog = Watchdog()
        self.watchdog.start()
        
        self.state_manager.set_state(BotState.STARTING)

        self.orchestrator = orchestrator
        self.trade_execution = trade_execution
        self.chat_engine = chat_engine
        self.config = config
        self.trade_manager = trade_manager
        self.trade_memory = trade_memory
        self.weights_engine = weights_engine
        
        self.current_signal = None
        # Initialise à None pour que la garde dans on_analyze_clicked
        # ne crashe pas sur un attribut manquant.
        self.scanner_thread = None
        self._scanner_threads = []
        self._trade_manager_running = False
        # Associe le ticket du trade initial au nombre de trades OB additionnels ouverts.
        # Ex: {123456: 1} signifie que le trade #123456 a deja declenche 1 OB additionnel.
        self._ob_trades_by_ticket: dict = {}
        # Etat du scanner cyclique manuel (indépendant de l'autopilot)
        self._scanner_active = False
        self.scanner_timer = None
        self._scan_cycle_count = 0
        
        # State machine visual updater listener
        self.state_manager.add_listener(self.on_state_changed)

        # File d'attente pour l'IA asynchrone
        import queue
        self.ai_analysis_queue = queue.Queue()
        self.ai_analyzer_thread = AIAnalyzerThread(self.orchestrator, self.ai_analysis_queue, self)
        self.ai_analyzer_thread.analysis_finished.connect(self.handle_ai_scan_result)
        self.ai_analyzer_thread.progress.connect(self.append_to_chat)
        self.ai_analyzer_thread.start()


        # Voice Manager & Config Advisor
        try:
            from src.core.voice_manager import VoiceManager
            self.voice_manager = VoiceManager(self)
            voice_cfg = self.config.get("voice", {})
            if voice_cfg:
                self.voice_manager.load_config(voice_cfg)
        except Exception:
            self.voice_manager = None

        try:
            from src.ai.config_advisor import ConfigAdvisor
            advisor_router = getattr(self.chat_engine, 'router', None) or getattr(self.orchestrator, 'router', None)
            self.config_advisor = ConfigAdvisor(advisor_router)
        except Exception:
            self.config_advisor = None

        self._transcription_thread = None

        self.setWindowTitle("🤖 FOCON ASSISTANT TRADE 1.1")
        # Calcul réactif des dimensions de la fenêtre selon la résolution de l'écran
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            init_w = min(1380, max(820, int(avail.width() * 0.88)))
            init_h = min(880, max(560, int(avail.height() * 0.88)))
            self.resize(init_w, init_h)
            self.move(
                avail.x() + max(0, (avail.width() - init_w) // 2),
                avail.y() + max(0, (avail.height() - init_h) // 2),
            )
        else:
            self.resize(1100, 750)
        self.setMinimumSize(780, 520)
        self.init_ui()

        # Trade Manager Timer: surveille les positions ouvertes (TP/SL/Trailing)
        # Démarre dès le lancement en mode scan manuel ou autopilot.
        self.tm_timer = QTimer(self)
        self.tm_timer.timeout.connect(self.on_trade_manager_timer)
        # Démarre maintenant : protège les positions même sans autopilot
        interval_ms = int(self.config.get('ui', {}).get('trade_manager_interval_ms', 3000))
        self.tm_timer.start(max(1000, interval_ms))

        # Initialise l'état visuel du bouton Pro Trader
        if hasattr(self.orchestrator, 'strategy_profile_manager'):
            p_id = self.orchestrator.strategy_profile_manager.get_active_profile_id()
            p_mode = self.orchestrator.strategy_profile_manager.get_active_mode()
            self.on_pro_trader_updated(p_id, p_mode)
        
    def open_memory_dialog(self):
        """Ouvre le panneau de la Conscience Évolutive, Journal Réflexif et Poids Adaptatifs."""
        dlg = MemoryDialog(self.trade_memory, self.weights_engine, self)
        dlg.exec()

    def open_backtest_dialog(self):
        """Ouvre le laboratoire de Backtest sur bougies réelles MT5."""
        try:
            from src.ui.backtest_dialog import BacktestDialog
            risk_mgr = getattr(self.orchestrator, 'risk', None)
            dlg = BacktestDialog(
                mt5_connector=self.orchestrator.mt5,
                risk_manager=risk_mgr,
                weights_engine=self.weights_engine,
                parent=self,
            )
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Erreur Backtest", f"Impossible d'ouvrir le module de backtest : {e}")

    def open_pro_trader_dialog(self):
        """Ouvre le panneau de Copie de Stratégies des 4 Traders Professionnels."""
        try:
            from src.ui.pro_trader_dialog import ProTraderCopyDialog
            mgr = getattr(self.orchestrator, 'strategy_profile_manager', None)
            if not mgr:
                from src.strategies.strategy_profile_manager import StrategyProfileManager
                mgr = StrategyProfileManager()
                self.orchestrator.strategy_profile_manager = mgr

            dlg = ProTraderCopyDialog(mgr, self)
            dlg.profile_updated.connect(self.on_pro_trader_updated)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Erreur Stratégies Pros", f"Impossible d'ouvrir le panneau des traders professionnels : {e}")

    def on_pro_trader_updated(self, profile_id: str, mode: str):
        """Met à jour l'intitulé et le style du bouton Pro Trader sur la barre principale."""
        if not hasattr(self, 'pro_trader_btn'):
            return

        mgr = getattr(self.orchestrator, 'strategy_profile_manager', None)
        if profile_id and mgr and profile_id in mgr.strategies:
            strat = mgr.strategies[profile_id]
            strat_short = strat.name.split()[0].upper()
            mode_tag = "SIMU" if mode == "simulation" else ("DÉMO" if mode == "copy" else "CUSTOM")
            self.pro_trader_btn.setText(f"👑 {strat_short} [{mode_tag}]")
            self.pro_trader_btn.setToolTip(
                f"Stratégie active : {strat.name} (Mode: {mode.upper()})\n"
                f"Cliquer pour changer de profil ou désactiver."
            )
            self.pro_trader_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1b5e20, stop:1 #2e7d32);
                    font-weight: bold;
                    padding: 10px 14px;
                    color: #ffd54f;
                    border-radius: 4px;
                    border: 2px solid #81c784;
                }
                QPushButton:hover {
                    background: #388e3c;
                }
            """)
        else:
            self.pro_trader_btn.setText("👑 COPIE TRADERS PROS")
            self.pro_trader_btn.setToolTip(
                "Activer et configurer la copie des traders professionnels\n"
                "(Corentin, Benjamin, Romain, Nada FX, George Soros)"
            )
            self.pro_trader_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8f00, stop:1 #ff6f00);
                    font-weight: bold;
                    padding: 10px 14px;
                    color: white;
                    border-radius: 4px;
                    border: 1px solid #ffd54f;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffa000, stop:1 #ff8f00);
                }
            """)


    def on_trade_manager_timer(self):
        """Surveille les positions ouvertes (TP fixe, Trailing, SL, Break-Even).

        Fonctionne TOUJOURS — aussi bien en mode Scanner Manuel qu'en Autopilot.
        En mode 'assist' le TradeManager protège uniquement les positions déjà
        ouvertes (il n'ouvre jamais de nouveau trade).
        """
        import logging as _lg
        try:
            if self._trade_manager_running:
                return
            mode = self.config.get('trading', {}).get('automation_mode', 'assist').lower()
            # NE JAMAIS bloquer la surveillance des positions ouvertes :
            # La protection des gains en dollars et le Break-Even doivent s'exécuter
            # en temps réel (toutes les 2-3s) même si un scan de marché tourne en arrière-plan !
            self._trade_manager_running = True
            self.trade_manager.manage_open_positions(auto_mode=mode)
        except Exception:
            _lg.getLogger(__name__).exception("on_trade_manager_timer a echoue")
            return
        finally:
            self._trade_manager_running = False
        # Refresh UI (tolérant lui aussi)
        try:
            self.refresh_positions()
        except Exception:
            _lg.getLogger(__name__).exception("refresh_positions a echoue")

        # Mise à jour en direct du badge de Tier de Capital
        try:
            self.update_tier_badge()
        except Exception:
            pass

    def update_tier_badge(self):
        """Met à jour en temps réel l'affichage du palier de capital détecté sur MT5."""
        try:
            if hasattr(self, 'tier_badge') and hasattr(self.orchestrator, 'mt5') and hasattr(self.orchestrator.mt5, 'get_account_info'):
                acc = self.orchestrator.mt5.get_account_info() or {}
                balance = float(acc.get('balance', 0.0) or 0.0)
                if balance > 0 and hasattr(self.orchestrator, 'risk') and hasattr(self.orchestrator.risk, 'get_account_tier'):
                    tier = self.orchestrator.risk.get_account_tier(balance)
                    t_name = tier['name']
                    base_pos = int(self.config.get('risk_management', {}).get('max_open_positions', tier['max_open_positions']))
                    is_manual = bool(self.config.get('risk_management', {}).get('manual_max_positions_override', False))
                    max_pos = base_pos if is_manual else tier['max_open_positions']
                    prof = getattr(self.orchestrator.risk, 'risk_profile', 'BALANCED')
                    self.tier_badge.setText(f"💎 {t_name.upper()} (${balance:,.2f}) • Max {max_pos} pos • Profil {prof}")
                    if tier['tier_id'] == 'MICRO':
                        self.tier_badge.setStyleSheet("background-color: #1b2e1b; color: #4caf50; font-weight: bold; padding: 8px 12px; border-radius: 4px; border: 1px solid #2e7d32; font-size: 11px;")
                    elif tier['tier_id'] == 'STANDARD':
                        self.tier_badge.setStyleSheet("background-color: #102a43; color: #38bec9; font-weight: bold; padding: 8px 12px; border-radius: 4px; border: 1px solid #1976d2; font-size: 11px;")
                    elif tier['tier_id'] == 'GROWTH':
                        self.tier_badge.setStyleSheet("background-color: #2a1b3d; color: #ba68c8; font-weight: bold; padding: 8px 12px; border-radius: 4px; border: 1px solid #7b1fa2; font-size: 11px;")
                    else:
                        self.tier_badge.setStyleSheet("background-color: #3e2723; color: #ffb74d; font-weight: bold; padding: 8px 12px; border-radius: 4px; border: 1px solid #ff9800; font-size: 11px;")
        except Exception:
            pass
        
    def init_ui(self):
        # Apply dark theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(11, 14, 23))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(13, 18, 30))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(30, 41, 59))
        palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0e17;
                color: #f1f5f9;
            }
            QWidget#central_widget {
                background-color: #0b0e17;
                color: #f1f5f9;
            }
            QTabWidget::pane {
                background-color: #0d121e;
                border: 1px solid #1e2638;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #151b29;
                color: #94a3b8;
                padding: 8px 18px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #1e2638;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e2638;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
            QTabBar::tab:hover {
                background-color: #24324a;
                color: #ffffff;
            }
            QScrollArea {
                background-color: #0d121e;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #0d121e;
            }
            QScrollBar:vertical {
                background: #0d121e;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #1e293b;
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

        central_widget = QWidget()
        central_widget.setObjectName("central_widget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        symbol_label = QLabel("SCANNER MULTI-MARCHÉS")
        symbol_label.setFont(QFont("Arial", 16, QFont.Bold))
        status_label = QLabel("🟢 MT5 CONNECTED")
        status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self.tier_badge = QLabel("📊 TIER DÉTECTION...")
        self.tier_badge.setStyleSheet("""
            QLabel {
                background-color: #1e222d;
                color: #29b6f6;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #0288d1;
                font-size: 11px;
            }
        """)
        
        self.analyze_btn = QPushButton("🔍 LANCER LE SCANNER")
        self.analyze_btn.setCheckable(True)
        self.analyze_btn.setToolTip(
            "Démarre le scanner automatique.\n"
            "L'analyse des marchés se relance toutes les 5 minutes.\n"
            "Les signaux trouvés sont affichés — vous choisissez si vous entrez ou non."
        )
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565c0;
                color: #e3f2fd;
                font-weight: bold;
                padding: 10px 14px;
                border-radius: 4px;
                border: 1px solid #42a5f5;
            }
            QPushButton:hover { background-color: #1976d2; color: white; }
            QPushButton:checked {
                background-color: #e65100;
                color: #fff3e0;
                border: 1px solid #ff9800;
            }
        """)
        self.analyze_btn.toggled.connect(self.on_scanner_toggled)

        self.autopilot_btn = QPushButton("🤖 AUTOPILOT : OFF")
        self.autopilot_btn.setCheckable(True)
        self.autopilot_btn.setStyleSheet("background-color: #F44336; font-weight: bold; padding: 10px; color: white;")
        self.autopilot_btn.toggled.connect(self.on_autopilot_toggled)
        
        self.memory_btn = QPushButton("🧠 CONSCIENCE ÉVOLUTIVE")
        self.memory_btn.setToolTip("Mémoire des Trades, Auto-Critique & Poids Adaptatifs")
        self.memory_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #512da8);
                font-weight: bold;
                padding: 10px 14px;
                color: white;
                border-radius: 4px;
                border: 1px solid #ba68c8;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9c27b0, stop:1 #673ab7);
            }
        """)
        self.memory_btn.clicked.connect(self.open_memory_dialog)

        self.backtest_btn = QPushButton("🔬 BACKTEST MT5")
        self.backtest_btn.setToolTip("Tester virtuellement le robot sur les bougies réelles MT5 avec n'importe quel capital")
        self.backtest_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00897b, stop:1 #00b0ff);
                font-weight: bold;
                padding: 10px 14px;
                color: white;
                border-radius: 4px;
                border: 1px solid #4dd0e1;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00bfa5, stop:1 #40c4ff);
            }
        """)
        self.backtest_btn.clicked.connect(self.open_backtest_dialog)

        self.pro_trader_btn = QPushButton("👑 COPIE TRADERS PROS")
        self.pro_trader_btn.setToolTip("Activer et configurer la copie des traders professionnels\n(Corentin, Benjamin, Romain, Nada FX, George Soros)")
        self.pro_trader_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8f00, stop:1 #ff6f00);
                font-weight: bold;
                padding: 10px 14px;
                color: white;
                border-radius: 4px;
                border: 1px solid #ffd54f;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffa000, stop:1 #ff8f00);
            }
        """)
        self.pro_trader_btn.clicked.connect(self.open_pro_trader_dialog)

        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setToolTip("Paramètres")
        self.settings_btn.setStyleSheet("background-color: #555555; padding: 10px;")
        self.settings_btn.clicked.connect(self.open_settings)
        
        header_layout.addWidget(symbol_label)
        header_layout.addWidget(self.tier_badge)
        header_layout.addStretch()
        header_layout.addWidget(self.pro_trader_btn)
        header_layout.addWidget(self.backtest_btn)
        header_layout.addWidget(self.memory_btn)
        header_layout.addWidget(self.autopilot_btn)
        header_layout.addWidget(self.analyze_btn)
        header_layout.addWidget(self.settings_btn)
        header_layout.addWidget(status_label)
        main_layout.addLayout(header_layout)

        
        # --- Quick Profit Management Bar (2-row premium design) ---
        profit_bar = QFrame()
        profit_bar.setFrameShape(QFrame.StyledPanel)
        profit_bar.setStyleSheet("""
            QFrame#profit_bar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1c2030, stop:1 #161b27);
                border: 1px solid #2d3555;
                border-radius: 8px;
                margin-top: 2px;
                margin-bottom: 4px;
            }
            QLabel {
                color: #c8cfe0;
                font-size: 12px;
            }
            QDoubleSpinBox {
                background-color: #0f1520;
                color: #00e676;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #374151;
                border-radius: 5px;
                padding: 4px 8px;
                min-width: 80px;
                max-width: 95px;
            }
            QDoubleSpinBox:hover {
                border: 1px solid #2962ff;
                background-color: #111827;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 16px;
                background-color: #1e2640;
                border: 1px solid #374151;
            }
        """)
        profit_bar.setObjectName("profit_bar")

        # Outer vertical layout: row1 + separator + row2
        profit_outer = QVBoxLayout(profit_bar)
        profit_outer.setContentsMargins(10, 6, 10, 6)
        profit_outer.setSpacing(4)

        # ── ROW 1 : Contrôles numériques ─────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        # Title
        profit_title = QLabel("⚡ GESTION PROFITS")
        profit_title.setFont(QFont("Arial", 10, QFont.Bold))
        profit_title.setStyleSheet("color: #4CAF50; letter-spacing: 0.5px;")
        profit_title.setFixedWidth(140)
        row1.addWidget(profit_title)

        # Separator vertical
        sep0 = QFrame(); sep0.setFrameShape(QFrame.VLine)
        sep0.setStyleSheet("color: #2d3555; min-width:1px; max-width:1px;")
        row1.addWidget(sep0)

        # 1. TP Fixe
        tp_lbl = QLabel("🎯 TP / Trade")
        tp_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        tp_lbl.setToolTip("Take-Profit fixe : ferme automatiquement le trade dès ce gain en $")
        self.quick_tp_spin = QDoubleSpinBox()
        self.quick_tp_spin.setRange(0.0, 500.0)
        self.quick_tp_spin.setSingleStep(0.5)
        self.quick_tp_spin.setDecimals(2)
        self.quick_tp_spin.setSuffix(" $")
        init_tp = float(self.config.get('risk_management', {}).get('take_profit_target_per_trade_usd', 4.0))
        self.quick_tp_spin.setValue(init_tp)
        self.quick_tp_spin.setToolTip("Coupe net dès que ce montant en $ est atteint.\nMettre 0.00 $ pour activer le mode Trailing.")
        self.quick_tp_spin.valueChanged.connect(self.on_quick_tp_changed)
        tp_grp = QVBoxLayout(); tp_grp.setSpacing(1)
        tp_grp.addWidget(tp_lbl); tp_grp.addWidget(self.quick_tp_spin)
        row1.addLayout(tp_grp)

        # 2. Recul toléré (Giveback)
        gb_lbl = QLabel("🛡️ Recul Max")
        gb_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.quick_giveback_spin = QDoubleSpinBox()
        self.quick_giveback_spin.setRange(0.1, 50.0)
        self.quick_giveback_spin.setSingleStep(0.1)
        self.quick_giveback_spin.setDecimals(2)
        self.quick_giveback_spin.setSuffix(" $")
        init_gb = float(self.config.get('risk_management', {}).get('giveback_usd_small', 0.6))
        self.quick_giveback_spin.setValue(init_gb)
        self.quick_giveback_spin.setToolTip("Coupe le trade si le profit chute de ce montant depuis son pic maximal atteint.")
        self.quick_giveback_spin.valueChanged.connect(self.on_quick_giveback_changed)
        gb_grp = QVBoxLayout(); gb_grp.setSpacing(1)
        gb_grp.addWidget(gb_lbl); gb_grp.addWidget(self.quick_giveback_spin)
        row1.addLayout(gb_grp)

        # 3. Objectif Jour
        daily_lbl = QLabel("🏁 Objectif Jour")
        daily_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.quick_daily_spin = QDoubleSpinBox()
        self.quick_daily_spin.setRange(0.0, 5000.0)
        self.quick_daily_spin.setSingleStep(5.0)
        self.quick_daily_spin.setDecimals(2)
        self.quick_daily_spin.setSuffix(" $")
        init_daily = float(self.config.get('risk_management', {}).get('daily_profit_target_usd', 50.0))
        self.quick_daily_spin.setValue(init_daily)
        self.quick_daily_spin.setToolTip("Le bot arrête de trader dès que le gain journalier total atteint ce montant.")
        self.quick_daily_spin.valueChanged.connect(self.on_quick_daily_changed)
        daily_grp = QVBoxLayout(); daily_grp.setSpacing(1)
        daily_grp.addWidget(daily_lbl); daily_grp.addWidget(self.quick_daily_spin)
        row1.addLayout(daily_grp)

        # Separator vertical
        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #2d3555; min-width:1px; max-width:1px;")
        row1.addWidget(sep1)

        # 4. Sens de Trading
        dir_lbl = QLabel("🔄 Sens")
        dir_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.quick_direction_combo = QComboBox()
        self.quick_direction_combo.setFixedHeight(30)
        self.quick_direction_combo.setStyleSheet("""
            QComboBox {
                background-color: #0f1520;
                color: #42a5f5;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #374151;
                border-radius: 5px;
                padding: 4px 8px;
                min-width: 155px;
            }
            QComboBox:hover { border: 1px solid #2962ff; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #1a1f30;
                color: #c8cfe0;
                selection-background-color: #2962ff;
                border: 1px solid #2d3555;
            }
        """)
        self.quick_direction_combo.addItem("🔄 Mixte (BUY & SELL)", "BOTH")
        self.quick_direction_combo.addItem("📈 Hausse Seule (BUY)", "BUY_ONLY")
        self.quick_direction_combo.addItem("📉 Baisse Seule (SELL)", "SELL_ONLY")
        curr_dir = self.config.get('trading', {}).get('allowed_trade_direction', 'BOTH')
        idx = self.quick_direction_combo.findData(curr_dir)
        if idx >= 0:
            self.quick_direction_combo.setCurrentIndex(idx)
        self.quick_direction_combo.setToolTip("Sens de trading autorisé : Mixte (Achat & Vente), Hausse uniquement (BUY) ou Baisse uniquement (SELL).")
        self.quick_direction_combo.currentIndexChanged.connect(self.on_quick_direction_changed)
        dir_grp = QVBoxLayout(); dir_grp.setSpacing(1)
        dir_grp.addWidget(dir_lbl); dir_grp.addWidget(self.quick_direction_combo)
        row1.addLayout(dir_grp)

        # Separator vertical
        sep2r1 = QFrame(); sep2r1.setFrameShape(QFrame.VLine)
        sep2r1.setStyleSheet("color: #2d3555; min-width:1px; max-width:1px;")
        row1.addWidget(sep2r1)

        # 5. Nombre max de positions simultanées
        maxpos_lbl = QLabel("📊 Max Positions")
        maxpos_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        maxpos_lbl.setToolTip("Nombre maximum de positions ouvertes simultanément (1 à 100).")
        self.quick_maxpos_spin = QSpinBox()
        self.quick_maxpos_spin.setRange(1, 100)
        self.quick_maxpos_spin.setSingleStep(1)
        self.quick_maxpos_spin.setFixedHeight(30)
        self.quick_maxpos_spin.setMinimumWidth(70)
        self.quick_maxpos_spin.setMaximumWidth(85)
        self.quick_maxpos_spin.setStyleSheet("""
            QSpinBox {
                background-color: #0f1520;
                color: #ffd54f;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #374151;
                border-radius: 5px;
                padding: 4px 8px;
            }
            QSpinBox:hover { border: 1px solid #ffb300; background-color: #111827; }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                background-color: #1e2640;
                border: 1px solid #374151;
            }
        """)
        init_maxpos = int(self.config.get('risk_management', {}).get('max_open_positions', 3))
        self.quick_maxpos_spin.setValue(init_maxpos)
        self.quick_maxpos_spin.setToolTip(
            "Nombre maximum de trades ouverts en même temps.\n"
            "Ex: 3 = le bot s'arrête d'entrer après 3 positions actives.\n"
            "Mettez 1 pour trader une seule position à la fois."
        )
        self.quick_maxpos_spin.valueChanged.connect(self.on_quick_maxpos_changed)
        maxpos_grp = QVBoxLayout(); maxpos_grp.setSpacing(1)
        maxpos_grp.addWidget(maxpos_lbl); maxpos_grp.addWidget(self.quick_maxpos_spin)
        row1.addLayout(maxpos_grp)

        # Separator vertical
        sep3r1 = QFrame(); sep3r1.setFrameShape(QFrame.VLine)
        sep3r1.setStyleSheet("color: #2d3555; min-width:1px; max-width:1px;")
        row1.addWidget(sep3r1)

        # 6. Mode d'exécution : Scanner Manuel vs Autopilot
        mode_lbl = QLabel("⚙️ Mode")
        mode_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        self.quick_mode_combo = QComboBox()
        self.quick_mode_combo.setFixedHeight(30)
        self.quick_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #0f1520;
                color: #ef9a9a;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #374151;
                border-radius: 5px;
                padding: 4px 6px;
                min-width: 140px;
            }
            QComboBox:hover { border: 1px solid #e53935; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #1a1f30;
                color: #c8cfe0;
                selection-background-color: #c62828;
                border: 1px solid #2d3555;
            }
        """)
        self.quick_mode_combo.addItem("🔍 Scanner Manuel", "assist")
        self.quick_mode_combo.addItem("🤖 Autopilot Complet", "auto")
        curr_mode = self.config.get('trading', {}).get('automation_mode', 'assist').lower()
        mode_idx = self.quick_mode_combo.findData(curr_mode)
        if mode_idx >= 0:
            self.quick_mode_combo.setCurrentIndex(mode_idx)
        self.quick_mode_combo.setToolTip(
            "🔍 Scanner Manuel : L'IA analyse et propose les opportunités.\n"
            "   Vous choisissez [PRENDRE] ou [IGNORER] pour chaque signal.\n\n"
            "🤖 Autopilot Complet : L'IA analyse ET ouvre les trades automatiquement.\n"
            "   Aucune confirmation requise. Activer aussi le bouton AUTOPILOT."
        )
        self.quick_mode_combo.currentIndexChanged.connect(self.on_quick_mode_changed)
        mode_grp = QVBoxLayout(); mode_grp.setSpacing(1)
        mode_grp.addWidget(mode_lbl); mode_grp.addWidget(self.quick_mode_combo)
        row1.addLayout(mode_grp)

        row1.addStretch()
        profit_outer.addLayout(row1)

        # ── SEPARATEUR HORIZONTAL ──────────────────────────────────────────
        h_sep = QFrame()
        h_sep.setFrameShape(QFrame.HLine)
        h_sep.setStyleSheet("color: #2d3555; max-height: 1px; margin: 0px 0px;")
        profit_outer.addWidget(h_sep)

        # ── ROW 2 : Boutons Preset TP rapides + Bouton SNIPER ─────────────
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        presets_lbl = QLabel("⚡ Sortie rapide :")
        presets_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        presets_lbl.setStyleSheet("color: #90caf9;")
        presets_lbl.setFixedWidth(100)
        row2.addWidget(presets_lbl)

        # Preset buttons
        self.tp_preset_buttons = {}
        _PRESETS = [
            (4.0,  "4 $",         "#1b5e20", "#4caf50", "Sortie automatique dès +4 $ de gain net"),
            (5.0,  "5 $",         "#1b5e20", "#4caf50", "Sortie automatique dès +5 $ de gain net"),
            (10.0, "10 $",        "#0d47a1", "#2196f3", "Sortie automatique dès +10 $ de gain net"),
            (12.0, "12 $",        "#0d47a1", "#2196f3", "Sortie automatique dès +12 $ de gain net"),
            (15.0, "15 $",        "#4a148c", "#ab47bc", "Sortie automatique dès +15 $ de gain net"),
            (0.0,  "🏃 Trailing", "#1a237e", "#5c6bc0", "Laisser courir sans TP fixe (trailing dynamique)"),
        ]
        for amt, lbl_txt, bg_off, border_off, tip in _PRESETS:
            btn = QPushButton(lbl_txt)
            btn.setToolTip(tip)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(54)
            btn.setFont(QFont("Arial", 9, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_off};
                    color: #cfd8dc;
                    border: 1px solid {border_off};
                    border-radius: 5px;
                    padding: 2px 8px;
                }}
                QPushButton:hover {{
                    background-color: {border_off};
                    color: #ffffff;
                    border: 1px solid #ffffff;
                }}
            """)
            btn.clicked.connect(lambda checked=False, val=amt: self.set_quick_tp_target(val))
            row2.addWidget(btn)
            self.tp_preset_buttons[amt] = btn

        # Séparateur vertical
        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #2d3555; min-width:1px; max-width:1px; margin: 2px 4px;")
        row2.addWidget(sep2)

        # Bouton Sniper Mode
        self.sniper_toggle_btn = QPushButton("🎯  SNIPER : OFF")
        self.sniper_toggle_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.sniper_toggle_btn.setCheckable(True)
        self.sniper_toggle_btn.setFixedHeight(28)
        self.sniper_toggle_btn.setMinimumWidth(160)
        self.sniper_toggle_btn.setCursor(Qt.PointingHandCursor)
        curr_prof = self.config.get('risk_management', {}).get('risk_profile', 'BALANCED')
        is_sniper = (curr_prof == 'SNIPER')
        self.sniper_toggle_btn.setChecked(is_sniper)
        self.sniper_toggle_btn.setToolTip(
            "Mode Sniper A+ : Le bot attend patiemment une opportunité institutionnelle parfaite\n"
            "(Order Block + Boomerang + Tendance EMA + Pullback RSI, Confiance >= 75%).\n"
            "Prend position avec un volume optimisé (Risque 2.5%, R:R 2.0+) pour maximiser les gains."
        )
        self.sniper_toggle_btn.clicked.connect(self.on_toggle_sniper_mode)
        row2.addWidget(self.sniper_toggle_btn)

        # Bouton Conseiller Lot Fixe (accès direct barre rapide)
        self.quick_lot_btn = QPushButton("⚖️ LOT : AUTO")
        self.quick_lot_btn.setFont(QFont("Arial", 9, QFont.Bold))
        self.quick_lot_btn.setFixedHeight(28)
        self.quick_lot_btn.setMinimumWidth(155)
        self.quick_lot_btn.setCursor(Qt.PointingHandCursor)
        self.quick_lot_btn.setToolTip(
            "Cliquez pour ouvrir le Conseiller Lot Fixe :\n"
            "• Voir quel Lot Fixe passe le mieux selon votre capital MT5\n"
            "• Basculer entre Lot Dynamique (%) et Lot Fixe en 1 clic."
        )
        self.quick_lot_btn.clicked.connect(self.open_fixed_lot_advisor)
        row2.addWidget(self.quick_lot_btn)

        row2.addStretch()
        profit_outer.addLayout(row2)

        self._refresh_tp_buttons(init_tp)
        self._refresh_sniper_button(is_sniper)
        self._refresh_quick_lot_button()

        main_layout.addWidget(profit_bar)
        
        # Main Tabs
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 11, QFont.Bold))
        
        # --- Tab 1: Signal Panel (Scanner) ---
        self.scanner_tab = QWidget()
        self.scanner_tab.setStyleSheet("background-color: #0d121e;")
        scanner_layout = QVBoxLayout(self.scanner_tab)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("background-color: #0d121e; border: none;")
        self.scroll_area.viewport().setStyleSheet("background-color: #0d121e;")
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: #0d121e;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        
        # Add default message
        default_label = QLabel("Aucune analyse en cours.\nCliquez sur 'LANCER LE SCANNER' pour analyser vos actifs.")
        default_label.setFont(QFont("Arial", 12, QFont.Bold))
        default_label.setAlignment(Qt.AlignCenter)
        default_label.setStyleSheet("color: #94a3b8; padding: 40px; background: transparent;")
        self.scroll_layout.addWidget(default_label)
        
        self.scroll_area.setWidget(self.scroll_content)
        scanner_layout.addWidget(self.scroll_area)
        
        self.tabs.addTab(self.scanner_tab, "🔍 Opportunités (Scanner)")
        
        # --- Tab 2: Positions Ouvertes ---
        self.positions_tab = QWidget()
        positions_layout = QVBoxLayout(self.positions_tab)
        
        refresh_layout = QHBoxLayout()
        self.refresh_pos_btn = QPushButton("🔄 Rafraîchir les positions")
        self.refresh_pos_btn.setStyleSheet("background-color: #555555; font-weight: bold; padding: 8px;")
        self.refresh_pos_btn.clicked.connect(self.refresh_positions)
        refresh_layout.addStretch()
        refresh_layout.addWidget(self.refresh_pos_btn)
        positions_layout.addLayout(refresh_layout)
        
        self.positions_table = QTableWidget(0, 9)
        self.positions_table.setHorizontalHeaderLabels(['Ticket', 'Symbole', 'Type', 'Volume', 'Ouverture', 'SL', 'TP', 'Prix Actuel', 'Profit'])
        self.positions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.positions_table.setStyleSheet("""
            QTableWidget {
                background-color: #0d121e;
                color: #f1f5f9;
                border: 1px solid #1e2638;
                border-radius: 6px;
                gridline-color: #1e2638;
            }
            QTableWidget::item {
                background-color: #0d121e;
                color: #f1f5f9;
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #161e30;
                color: #93c5fd;
                font-weight: bold;
                border: 1px solid #1e2638;
                padding: 6px;
            }
        """)
        p_hdr = self.positions_table.horizontalHeader()
        p_hdr.setSectionResizeMode(QHeaderView.Interactive)
        p_hdr.setStretchLastSection(True)
        p_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        p_hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        p_hdr.setSectionResizeMode(8, QHeaderView.Stretch)
        positions_layout.addWidget(self.positions_table)

        # ── Tab 2b : Pendings (Order Blocks en attente) — nom modifiable ──
        self.pendings_tab = QWidget()
        pendings_layout = QVBoxLayout(self.pendings_tab)
        pendings_top_bar = QHBoxLayout()
        pendings_hint = QLabel("✏️ Double-cliquez sur la colonne « Nom » pour renommer un Pending Block (31 car. max, envoyé à MT5 comme commentaire).")
        pendings_hint.setWordWrap(True)
        pendings_hint.setStyleSheet("color: #a0c4ff; font-size: 11px; padding: 4px;")
        pendings_top_bar.addWidget(pendings_hint, stretch=1)
        self.clean_pendings_btn = QPushButton("🧹 Nettoyer terminés")
        self.clean_pendings_btn.setStyleSheet("background-color: #2a2e39; color: #e0e0e0; border: 1px solid #363c4e; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self.clean_pendings_btn.setToolTip("Supprime de la liste les ordres limites fermés, annulés ou remplis.")
        self.clean_pendings_btn.clicked.connect(self.clean_closed_pendings)
        pendings_top_bar.addWidget(self.clean_pendings_btn)
        pendings_layout.addLayout(pendings_top_bar)
        self.pendings_table = QTableWidget(0, 6)
        self.pendings_table.setHorizontalHeaderLabels(['Ticket', 'Symbole', 'Sens', 'Nom (modifiable)', 'Entrée', 'Statut'])
        self.pendings_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pendings_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pendings_table.setStyleSheet("""
            QTableWidget {
                background-color: #0d121e;
                color: #f1f5f9;
                border: 1px solid #1e2638;
                border-radius: 6px;
                gridline-color: #1e2638;
            }
            QTableWidget::item {
                background-color: #0d121e;
                color: #f1f5f9;
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #161e30;
                color: #93c5fd;
                font-weight: bold;
                border: 1px solid #1e2638;
                padding: 6px;
            }
        """)
        self.pendings_table.setToolTip("Double-cliquez sur un nom pour le modifier.")
        self.pendings_table.cellChanged.connect(self.on_pending_name_edited)
        p_hdr2 = self.pendings_table.horizontalHeader()
        p_hdr2.setSectionResizeMode(QHeaderView.Interactive)
        p_hdr2.setStretchLastSection(True)
        p_hdr2.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        p_hdr2.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        p_hdr2.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        p_hdr2.setSectionResizeMode(3, QHeaderView.Stretch)
        p_hdr2.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        p_hdr2.setSectionResizeMode(5, QHeaderView.Stretch)
        pendings_layout.addWidget(self.pendings_table)
        self.tabs.addTab(self.pendings_tab, "📌 Pendings (nommables)")

        self.tabs.addTab(self.positions_tab, "📊 Positions Ouvertes")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Chat Panel
        chat_group = QGroupBox("💬 Assistant")
        chat_group.setFont(QFont("Arial", 11, QFont.Bold))
        chat_group.setStyleSheet("""
            QGroupBox {
                background-color: #0d121e;
                border: 1px solid #1e2638;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 14px;
                color: #60a5fa;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 2px 8px;
                color: #93c5fd;
                background-color: #161e30;
                border: 1px solid #283754;
                border-radius: 4px;
            }
        """)
        chat_layout = QVBoxLayout(chat_group)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #070a10;
                color: #f1f5f9;
                border: 1px solid #1e2638;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Barre d'outils Vocale & Conseiller Stratégique
        voice_toolbar = QHBoxLayout()
        voice_toolbar.setSpacing(6)

        self.mic_btn = QPushButton("🎙️ Parler")
        self.mic_btn.setFont(QFont("Arial", 9, QFont.Bold))
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #e0e0e0;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #363c4e;
                color: #ffffff;
            }
        """)
        self.mic_btn.setToolTip("Cliquez pour parler au micro (reconnaissance vocale Whisper). Vous pouvez aussi utiliser Win + H.")
        self.mic_btn.clicked.connect(self.on_toggle_microphone)

        self.voice_btn = QPushButton("🔊 Voix ON")
        self.voice_btn.setFont(QFont("Arial", 9, QFont.Bold))
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e3a8a;
                color: #60a5fa;
                border: 1px solid #2563eb;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                color: white;
            }
        """)
        self.voice_btn.setToolTip("Activer ou couper la synthèse vocale (le robot parle en français).")
        self.voice_btn.clicked.connect(self.on_toggle_voice_output)

        self.opt_btn = QPushButton("💡 Optimiser Réglages")
        self.opt_btn.setFont(QFont("Arial", 9, QFont.Bold))
        self.opt_btn.setStyleSheet("""
            QPushButton {
                background-color: #065f46;
                color: #34d399;
                border: 1px solid #059669;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #059669;
                color: white;
            }
        """)
        self.opt_btn.setToolTip("Demande à l'IA d'analyser le compte et de proposer les meilleurs réglages.")
        self.opt_btn.clicked.connect(self.on_request_optimization)

        self.voice_settings_btn = QPushButton("⚙️ Voix & IA")
        self.voice_settings_btn.setFont(QFont("Arial", 9, QFont.Bold))
        self.voice_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #e0e0e0;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #363c4e;
                color: white;
            }
        """)
        self.voice_settings_btn.setToolTip("Personnaliser la voix (vitesse, hauteur, choix de voix) et l'Auto-Adaptation intelligente.")
        self.voice_settings_btn.clicked.connect(self.open_voice_settings)

        voice_toolbar.addWidget(self.mic_btn)
        voice_toolbar.addWidget(self.voice_btn)
        voice_toolbar.addWidget(self.opt_btn)
        voice_toolbar.addWidget(self.voice_settings_btn)
        chat_layout.addLayout(voice_toolbar)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Écris ton message ou clique sur 🎙️ Parler...")
        self.chat_input.setFixedHeight(36)
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #070a10;
                color: #f1f5f9;
                border: 1px solid #1e2638;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #3b82f6;
                background-color: #0b0f19;
            }
        """)
        self.chat_input.returnPressed.connect(self.on_send_chat)
        
        self.send_btn = QPushButton("➤ Envoyer")
        self.send_btn.setFixedHeight(36)
        self.send_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 16px;
                border: 1px solid #3b82f6;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        self.send_btn.clicked.connect(self.on_send_chat)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)
        
        # Agencement réactif et redimensionnable entre Onglets et Chat via QSplitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setObjectName("main_splitter")
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.addWidget(chat_group)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStyleSheet("""
            QSplitter#main_splitter::handle {
                background-color: #212636;
                height: 6px;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSplitter#main_splitter::handle:hover {
                background-color: #2962ff;
            }
        """)
        main_layout.addWidget(self.main_splitter, stretch=1)
        
    def _cleanup_scanner_thread(self):
        """Détruit proprement l'ancien ScannerThread avant d'en créer un nouveau.
        
        Sans ce nettoyage, réassigner self.scanner_thread alors que le QThread
        C++ sous-jacent est encore vivant provoque un access violation (crash
        natif sans traceback Python).
        """
        old = self.scanner_thread
        if old is None:
            return
        try:
            old.stop()              # demande l'arrêt du travail en cours
        except Exception:
            pass
        try:
            # Déconnecte les signaux pour éviter des callbacks orphelins
            old.scan_finished.disconnect()
            old.scan_failed.disconnect()
            old.progress.disconnect()
        except Exception:
            pass
        try:
            if old.isRunning():
                old.quit()          # demande la fin de la boucle Qt
                old.wait(2000)      # attend max 2s
            old.deleteLater()       # libère le QObject C++ proprement
        except Exception:
            pass
        self.scanner_thread = None

    def _on_scanner_thread_finished(self, thread):
        """Release a scanner thread only after Qt has fully stopped it."""
        try:
            if self.scanner_thread is thread:
                self.scanner_thread = None
            if thread in self._scanner_threads:
                self._scanner_threads.remove(thread)
            thread.deleteLater()
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("_on_scanner_thread_finished a echoue")

    # ─────────────────────────────────────────────────────────────────────────
    # SCANNER CYCLIQUE MANUEL (analyse auto, prise de position manuelle)
    # ─────────────────────────────────────────────────────────────────────────

    def on_scanner_toggled(self, checked: bool):
        """Bascule le scanner cyclique ON / OFF."""
        if checked:
            self.start_scanner_loop()
        else:
            self.stop_scanner_loop()

    def start_scanner_loop(self):
        """Lance le scanner automatique toutes les N minutes (prise de position manuelle)."""
        import logging as _lg
        log = _lg.getLogger(__name__)
        try:
            self._scanner_active = True
            self._scan_cycle_count = 0
            self.analyze_btn.setChecked(True)
            self.analyze_btn.setText("⏹ STOPPER LE SCANNER")
            self.append_to_chat(
                "🔍 [Scanner] Démarré — analyse automatique des marchés.\n"
                "Les signaux trouvés seront affichés ici. "
                "Vous choisissez vous-même si vous entrez en position."
            )
            # Timer cyclique de scan (par défaut 5 min, configurable)
            scan_interval_min = int(self.config.get('trading', {}).get('scanner_interval_minutes', 5))
            scan_interval_ms  = scan_interval_min * 60 * 1000
            if self.scanner_timer is None:
                self.scanner_timer = QTimer(self)
                self.scanner_timer.timeout.connect(self.on_scanner_timer)
            if not self.scanner_timer.isActive():
                self.scanner_timer.start(scan_interval_ms)
            # Premier scan immédiat
            try:
                self.on_scanner_timer()
            except Exception:
                log.exception("Premier cycle scanner a echoue")
        except Exception:
            log.exception("start_scanner_loop a echoue")

    def stop_scanner_loop(self):
        """Arrête le scanner cyclique."""
        import logging as _lg
        try:
            self._scanner_active = False
            if self.scanner_timer is not None and self.scanner_timer.isActive():
                self.scanner_timer.stop()
            self.analyze_btn.setChecked(False)
            self.analyze_btn.setText("🔍 LANCER LE SCANNER")
            self.append_to_chat("⏹ [Scanner] Arrêté.")
        except Exception:
            _lg.getLogger(__name__).exception("stop_scanner_loop a echoue")

    def on_scanner_timer(self):
        """Déclencheur cyclique du scanner manuel."""
        import logging as _lg
        try:
            if not self._scanner_active:
                return
            # Ne pas empiler 2 scans
            if self.scanner_thread is not None and self.scanner_thread.isRunning():
                _lg.getLogger(__name__).warning("[Scanner] Scan précédent toujours en cours, cycle ignoré.")
                return
            self._scan_cycle_count += 1
            self.append_to_chat(f"🔍 [Scanner] Cycle #{self._scan_cycle_count} — analyse en cours...")
            self.is_autopilot_scan = False  # mode manuel = affichage des signaux, pas d'exécution auto
            self.on_analyze_clicked()
        except Exception:
            _lg.getLogger(__name__).exception("on_scanner_timer a echoue")

    # ─────────────────────────────────────────────────────────────────────────

    def on_analyze_clicked(self):
        """Lance un scan unique du marché (appelé aussi par les timers scanner et autopilot)."""
        import logging as _lg
        try:
            # Prevent starting a new scan if one is already running
            if self.scanner_thread is not None and self.scanner_thread.isRunning():
                _lg.getLogger(__name__).warning("Scan déjà en cours, demande ignorée.")
                return
            # Keep completed QThread objects around until their finished signal
            # cleanup runs; immediate deleteLater + recreate can crash natively.
            if self.scanner_thread is not None and not self.scanner_thread.isRunning():
                self.scanner_thread = None

            # Efface les signaux précédents SEULEMENT si c'est un cycle manuel
            # (pas l'autopilot, qui conserve les propositions en attente)
            if not getattr(self, 'is_autopilot_scan', False):
                for i in reversed(range(self.scroll_layout.count())):
                    item = self.scroll_layout.takeAt(i)
                    widget = item.widget() if item is not None else None
                    if widget:
                        widget.deleteLater()

            # Mise à jour du bouton seulement si c'est le scanner manuel
            if not getattr(self, 'is_autopilot_scan', False) and self._scanner_active:
                self.analyze_btn.setText("⏳ Scan en cours...")

            timeframe = self.config.get('trading', {}).get('timeframes', ['M15'])[0]
            allowed_types = self.config.get('trading', {}).get('allowed_market_types', None)
            fallback_symbols = self.config.get('trading', {}).get('symbols', ['EURUSD'])

            # IMPORTANT : ne plus appeler MT5 (IPC native C) depuis le thread
            # GUI — un crash C ici fermait l'app sans aucun traceback. Le fetch
            # des symboles est fait dans ScannerThread.run().
            direction_filter = self.config.get('trading', {}).get('allowed_trade_direction', 'BOTH')
            thread = ScannerThread(
                self.orchestrator, None, timeframe,
                allowed_types=allowed_types,
                fallback_symbols=fallback_symbols,
                direction_filter=direction_filter,
                watchdog=self.watchdog,
                ai_queue=getattr(self, 'ai_analysis_queue', None),
                parent=self)
            self.scanner_thread = thread
            self._scanner_threads.append(thread)
            thread.scan_finished.connect(self.handle_scan_result)
            thread.scan_failed.connect(self.on_scan_failed)
            thread.progress.connect(self._on_scan_progress)
            thread.finished.connect(lambda t=thread: self._on_scanner_thread_finished(t))
            thread.start()
        except Exception:
            _lg.getLogger(__name__).exception("on_analyze_clicked a echoue")
            try:
                if self._scanner_active:
                    self.analyze_btn.setText("⏹ STOPPER LE SCANNER")
            except Exception:
                pass

    def on_scan_failed(self, reason: str):
        import logging as _lg
        _lg.getLogger(__name__).error("Scan echoue: %s", reason)
        try:
            # Rétablit le libellé du bouton selon l'état actif
            if getattr(self, '_scanner_active', False):
                self.analyze_btn.setText("⏹ STOPPER LE SCANNER")
            else:
                self.analyze_btn.setChecked(False)
                self.analyze_btn.setText("🔍 LANCER LE SCANNER")
            self.is_autopilot_scan = False
            err_lbl = QLabel(f"⚠️ Scan interrompu : {reason}")
            err_lbl.setStyleSheet("color: #ff6b6b; font-weight: bold; padding: 4px;")
            err_lbl.setWordWrap(True)
            self.scroll_layout.addWidget(err_lbl)
            self.append_to_chat(f"[Scanner] Echec : {reason}")
        except Exception:
            _lg.getLogger(__name__).exception("on_scan_failed affichage echoue")

    def _on_scan_progress(self, msg: str):
        """Update button text with live progress without blocking the UI."""
        short = msg if len(msg) <= 50 else msg[:47] + "..."
        # N'écrase pas le label si le scanner est en mode cyclique
        if not getattr(self, '_scanner_active', False):
            self.analyze_btn.setText(short)

    def on_tab_changed(self, index):
        if index == 1: # Pendings tab
            self.refresh_pendings_tab()
        elif index == 2: # Positions tab
            self.refresh_positions()

    def clean_closed_pendings(self):
        """Nettoie les ordres limites inactifs (annulés/remplis/expirés)."""
        mgr = self._get_pending_manager()
        if mgr:
            count = mgr.clear_inactive()
            self.append_to_chat(f"🧹 [PENDING] {count} ordre(s) inactif(s) nettoyé(s).")
            self.refresh_pendings_tab()
            
    def refresh_positions(self):
        positions = self.orchestrator.mt5.get_open_positions()
        self.positions_table.setRowCount(len(positions))
        
        for row, pos in enumerate(positions):
            self.positions_table.setItem(row, 0, QTableWidgetItem(str(pos['ticket'])))
            self.positions_table.setItem(row, 1, QTableWidgetItem(pos['symbol']))
            
            type_item = QTableWidgetItem(pos['type'])
            type_item.setForeground(QColor("#4CAF50") if pos['type'] == 'BUY' else QColor("#F44336"))
            self.positions_table.setItem(row, 2, type_item)
            
            self.positions_table.setItem(row, 3, QTableWidgetItem(str(pos['volume'])))
            self.positions_table.setItem(row, 4, QTableWidgetItem(self._fmt_price(pos['price_open'])))
            self.positions_table.setItem(row, 5, QTableWidgetItem(self._fmt_price(pos['sl'])))
            self.positions_table.setItem(row, 6, QTableWidgetItem(self._fmt_price(pos['tp'])))
            self.positions_table.setItem(row, 7, QTableWidgetItem(self._fmt_price(pos['price_current'])))
            
            profit_item = QTableWidgetItem(f"{float(pos['profit']):.2f}")
            profit_item.setForeground(QColor("#4CAF50") if pos['profit'] >= 0 else QColor("#F44336"))
            self.positions_table.setItem(row, 8, profit_item)
        # ── Rafraîchit aussi l'onglet "Pendings" (noms modifiables) ──
        try:
            self.refresh_pendings_tab()
        except Exception:
            pass

    def refresh_pendings_tab(self):
        """Remplit l'onglet Pendings avec les ordres limites suivis (nom modifiable)."""
        if not hasattr(self, 'pendings_table'):
            return
        mgr = self._get_pending_manager()
        pendings = []
        try:
            if mgr is not None:
                pendings = list(mgr.get_all_pendings() or [])
        except Exception:
            pendings = []
        try:
            pendings.sort(key=lambda p: (0 if getattr(p, 'status', '') == 'ACTIVE' else 1,
                                         -(getattr(p, 'created_at', 0.0) or 0.0)))
        except Exception:
            pass
        self.pendings_table.blockSignals(True)
        try:
            self.pendings_table.setRowCount(len(pendings))
            for row, p in enumerate(pendings):
                t_item = QTableWidgetItem(str(getattr(p, 'ticket', '')))
                t_item.setFlags(t_item.flags() & ~Qt.ItemIsEditable)
                self.pendings_table.setItem(row, 0, t_item)
                s_item = QTableWidgetItem(str(getattr(p, 'symbol', '')))
                s_item.setFlags(s_item.flags() & ~Qt.ItemIsEditable)
                self.pendings_table.setItem(row, 1, s_item)
                d_item = QTableWidgetItem(str(getattr(p, 'direction', '')))
                d_item.setFlags(d_item.flags() & ~Qt.ItemIsEditable)
                d_item.setForeground(QColor("#4CAF50") if str(getattr(p, 'direction', '')).upper() == 'BUY' else QColor("#F44336"))
                self.pendings_table.setItem(row, 2, d_item)
                name_item = QTableWidgetItem(str(getattr(p, 'label', '') or ''))
                name_item.setToolTip("Double-cliquez pour modifier le nom du Pending Block (31 car. max).")
                self.pendings_table.setItem(row, 3, name_item)
                e_item = QTableWidgetItem(self._fmt_price(getattr(p, 'entry_price', None)))
                e_item.setFlags(e_item.flags() & ~Qt.ItemIsEditable)
                self.pendings_table.setItem(row, 4, e_item)
                st_val = str(getattr(p, 'status', '')).upper()
                st_item = QTableWidgetItem(st_val)
                st_item.setFlags(st_item.flags() & ~Qt.ItemIsEditable)
                if st_val == 'ACTIVE':
                    st_item.setForeground(QColor("#4CAF50"))
                elif st_val == 'FILLED':
                    st_item.setForeground(QColor("#2196F3"))
                elif st_val in ('CANCELLED', 'ANNULÉ'):
                    st_item.setForeground(QColor("#9E9E9E"))
                else:
                    st_item.setForeground(QColor("#FF9800"))
                self.pendings_table.setItem(row, 5, st_item)
        finally:
            self.pendings_table.blockSignals(False)

    def on_pending_name_edited(self, row: int, col: int):
        """Appelé quand l'utilisateur modifie le nom d'un pending dans le tableau."""
        if col != 3 or not hasattr(self, 'pendings_table'):
            return
        try:
            ticket_item = self.pendings_table.item(row, 0)
            name_item = self.pendings_table.item(row, 3)
            if ticket_item is None or name_item is None:
                return
            ticket = int(str(ticket_item.text()).strip())
            new_name = str(name_item.text() or "").strip()[:31]
            mgr = self._get_pending_manager()
            if mgr is not None and mgr.rename_pending(ticket, new_name):
                self.append_to_chat(f"✏️ Pending #{ticket} renommé en « {new_name} ».")
                if hasattr(self, 'voice_manager') and self.voice_manager and not self.voice_manager.is_muted:
                    try:
                        self.voice_manager.speak(f"Pending {ticket} renommé en {new_name}.")
                    except Exception:
                        pass
            else:
                self.append_to_chat(f"⚠️ Impossible de renommer le pending #{ticket}.")
                self.refresh_pendings_tab()
        except Exception:
            try:
                self.refresh_pendings_tab()
            except Exception:
                pass
            
    def append_to_chat(self, msg, sender="IA"):
        self.chat_display.append(f"<b>{sender} :</b> {msg}<br>")
            
    def on_autopilot_toggled(self, checked):
        import logging as _lg
        try:
            if checked:
                self.autopilot_btn.setText("🤖 AUTOPILOT : ON")
                self.autopilot_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 10px; color: white;")
                self.append_to_chat("Mode Autopilot activé. Scans automatiques toutes les 5 minutes et exécution automatique des trades.")
                self.start_autopilot()
            else:
                self.autopilot_btn.setText("🤖 AUTOPILOT : OFF")
                self.autopilot_btn.setStyleSheet("background-color: #F44336; font-weight: bold; padding: 10px; color: white;")
                self.append_to_chat("Mode Autopilot désactivé.")
                self.stop_autopilot()
        except Exception:
            _lg.getLogger(__name__).exception("on_autopilot_toggled a echoue")
            try:
                self.stop_autopilot()
                self.autopilot_btn.setChecked(False)
            except Exception:
                pass
            
    def start_autopilot(self):
        import logging as _lg
        log = _lg.getLogger(__name__)
        try:
            self.is_autopilot_scan = False
            # Arrêter le scanner manuel s'il tourne (les deux modes sont mutuellement exclusifs)
            if getattr(self, '_scanner_active', False):
                try:
                    self.stop_scanner_loop()
                except Exception:
                    log.exception("Impossible d'arreter le scanner manuel avant l'autopilot")
            # Start the market scanner autopilot (every 5 minutes).
            # Ne jamais recréer un QTimer par-dessus l'ancien : stop + réutilise si présent.
            try:
                if not hasattr(self, 'autopilot_timer') or self.autopilot_timer is None:
                    self.autopilot_timer = QTimer(self)
                    self.autopilot_timer.timeout.connect(self.on_autopilot_timer)
                if not self.autopilot_timer.isActive():
                    self.autopilot_timer.start(300000)  # 300,000 ms = 5 minutes
            except Exception:
                log.exception("autopilot_timer.start a echoue")
                return
            # Lancement du 1er scan immédiat, avec garde anti double-scan.
            # IMPORTANT : ne pas lever si le scan echoue au demarrage, sinon
            # l'exception remonte dans le slot toggled -> fermeture de l'app.
            try:
                self.on_autopilot_timer()
            except Exception:
                log.exception("Premier scan autopilot a echoue")
                try:
                    self.is_autopilot_scan = False
                    self.analyze_btn.setEnabled(True)
                    self.analyze_btn.setText("SCANNER LES MARCHES")
                except Exception:
                    pass
        except Exception:
            log.exception("start_autopilot a echoue")
        
    def stop_autopilot(self):
        import logging as _lg
        try:
            if hasattr(self, 'autopilot_timer') and self.autopilot_timer is not None:
                try:
                    self.autopilot_timer.stop()
                except Exception:
                    pass
                self.is_autopilot_scan = False
            # Note: tm_timer reste actif — il surveille les positions en continu
            # même hors autopilot (TP fixe, Trailing, Break-Even).
        except Exception:
            _lg.getLogger(__name__).exception("stop_autopilot a echoue")

    def on_autopilot_timer(self):
        import logging as _lg
        try:
            # Garde : ne jamais empiler 2 scans (le précédent peut durer 10+ min
            # avec les retries IA). Sinon threads concurrents -> crash Qt/MT5.
            if self.scanner_thread is not None and self.scanner_thread.isRunning():
                _lg.getLogger(__name__).warning(
                    "Autopilot : scan précédent toujours en cours, cycle ignoré.")
                try:
                    self.append_to_chat(
                        "[Autopilot] Scan précédent toujours en cours — nouveau cycle ignoré.")
                except Exception:
                    pass
                return
            self.append_to_chat("[Autopilot] Démarrage d'un nouveau cycle de scan automatique...")
            self.is_autopilot_scan = True
            self.on_analyze_clicked()
        except Exception:
            _lg.getLogger(__name__).exception("on_autopilot_timer a echoue")

    def on_state_changed(self, state):
        import logging as _lg
        try:
            state_msg = f"🟢 Statut Bot: {state.name}"
            if state.name == 'ERROR':
                state_msg = "🔴 ERREUR CRITIQUE (Watchdog/Connexion) - Bot en sécurité"
            elif state.name == 'PAUSED':
                state_msg = "⏸️ Bot en Pause"
            
            # Appends the state update to the chat or sets it to the window title
            self.setWindowTitle(f"🤖 FOCON ASSISTANT TRADE 1.1 - {state.name}")
            self.append_to_chat(state_msg)
        except Exception:
            pass
        
    def handle_ai_scan_result(self, result):
        import logging as _lg
        try:
            self.append_to_chat(f"🧠 [IA Async] Résultat reçu pour {result.get('symbol', 'Inconnu')}.")
            self.handle_scan_result([result])
        except Exception:
            _lg.getLogger(__name__).exception("handle_ai_scan_result a echoue")

    def handle_scan_result(self, results):
        import logging as _lg
        try:
            # Rétablir le libellé du bouton selon le mode actif
            if getattr(self, '_scanner_active', False):
                scan_interval_min = int(self.config.get('trading', {}).get('scanner_interval_minutes', 5))
                self.analyze_btn.setText(f"⏹ Scanner actif (prochain cycle dans {scan_interval_min} min)")
            elif not getattr(self, 'is_autopilot_scan', False):
                self.analyze_btn.setChecked(False)
                self.analyze_btn.setText("🔍 LANCER LE SCANNER")

            # Sanitize : numpy/objets MT5 -> types simples (evite crash Qt).
            try:
                import math
                def _safe(o):
                    if o is None or isinstance(o, (bool, int, float, str)):
                        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                            return None
                        return o
                    if isinstance(o, dict):
                        return {str(k): _safe(v) for k, v in o.items()}
                    if isinstance(o, (list, tuple)):
                        return [_safe(v) for v in o]
                    return str(o)
                safe_results = _safe(results)
            except Exception:
                safe_results = str(results)
            # Inject context for chat
            self.chat_engine.inject_context({"type": "scan_results", "results": safe_results})

            if not results:
                lbl = QLabel("Aucun r\u00e9sultat.")
                self.scroll_layout.addWidget(lbl)
                return

            # Tri des opportunités : prioritiser la meilleure opportunité (A+ / plus haute conviction et R:R)
            def _rank_opportunity(r):
                if not isinstance(r, dict):
                    return -100.0
                status = str(r.get("status", "")).upper()
                if status in ("WAIT", "CLOSED") or "error" in r:
                    return -50.0
                
                score = 0.0
                if status == "OB_SIGNAL":
                    score += 100.0  # OB institutionnel retesté
                elif status in ("PENDING_CRT", "MARKET_CRT"):
                    score += 95.0   # Setup CRT institutionnel
                    of_data = r.get("order_flow")
                    if of_data:
                        is_conf = getattr(of_data, 'is_confirmed', False) or (isinstance(of_data, dict) and of_data.get('is_confirmed'))
                        dq = getattr(of_data, 'data_quality', None) or (of_data.get('data_quality') if isinstance(of_data, dict) else 'ESTIMATED_QUOTE_RULE')
                        if is_conf:
                            if dq == 'REAL_OBSERVED':
                                score += 20.0  # Confluence réelle vérifiée avec flux d'ordres institutionnels
                            elif dq == 'ESTIMATED_QUOTE_RULE':
                                score += 10.0  # Estimation Quote-Rule Forex
                            else:
                                score += 0.0
                        elif dq == 'DEGRADED_STALE':
                            score -= 10.0  # Pénalité stricte de données périmées
                elif status == "PENDING_OB":
                    score += 85.0
                elif status == "SIGNAL":
                    score += 90.0

                dec = r.get("decision", {})
                if isinstance(dec, dict):
                    action = str(dec.get("action", "")).upper()
                    if action in ("BUY", "SELL"):
                        score += 50.0
                    conf = float(dec.get("confidence", 0) or 0)
                    score += conf
                    rr = float(dec.get("risk_reward", 0) or dec.get("reward_risk_ratio", 0) or 0)
                    score += min(rr * 10.0, 50.0)
                return score

            sorted_results = sorted(results, key=_rank_opportunity, reverse=True)

            for res in sorted_results:
                try:
                    if "error" in res:
                        err_lbl = QLabel(f"Erreur: {res['error']}")
                        err_lbl.setStyleSheet("color: red;")
                        self.scroll_layout.addWidget(err_lbl)
                        continue

                    if res.get("status") == "WAIT":
                        sym = res.get('symbol', 'Inconnu')
                        wait_lbl = QLabel(f"Status: WAIT ({sym}) - Pas d'opportunit\u00e9 claire.")
                        self.scroll_layout.addWidget(wait_lbl)
                        continue

                    if res.get("status") == "CLOSED":
                        syms = ", ".join(res.get("symbols", []))
                        closed_lbl = QLabel(f"March\u00e9s ferm\u00e9s ignor\u00e9s : {syms}")
                        closed_lbl.setStyleSheet("color: orange; font-weight: bold;")
                        self.scroll_layout.addWidget(closed_lbl)
                        continue

                    # ─── Routage : Scanner Manuel vs Autopilot ──────────────────────
                    # En Scanner Manuel (_scanner_active=True) → toujours afficher
                    # les signaux pour décision manuelle, même si automation_mode='auto'.
                    # En Autopilot (is_autopilot_scan=True) → exécution automatique.
                    is_manual_scan = getattr(self, '_scanner_active', False) and not getattr(self, 'is_autopilot_scan', False)
                    mode = self.config.get('trading', {}).get('automation_mode', 'assist').lower()
                    should_auto_execute = getattr(self, 'is_autopilot_scan', False) or (mode == 'auto' and not is_manual_scan)

                    if res.get("status") == "OB_SIGNAL":
                        if should_auto_execute:
                            self.execute_autopilot_trade(res)
                        else:
                            self.add_ob_signal_widget(res)
                        continue

                    if res.get("status") == "PENDING_OB":
                        if should_auto_execute:
                            self.execute_pending_ob_trade(res)
                        else:
                            self.add_pending_ob_widget(res)
                        continue

                    if should_auto_execute:
                        self.execute_autopilot_trade(res)
                    else:
                        self.add_signal_widget(res)
                except Exception as e:
                    import logging as _lg
                    _lg.getLogger(__name__).exception("Erreur affichage resultat: %s", e)
                    try:
                        err_lbl = QLabel(f"Erreur affichage: {e}")
                        err_lbl.setStyleSheet("color: red;")
                        self.scroll_layout.addWidget(err_lbl)
                    except Exception:
                        pass
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("handle_scan_result a echoue")
            try:
                self.analyze_btn.setEnabled(True)
                self.analyze_btn.setText("SCANNER LES MARCHÉS")
            except Exception:
                pass

    def execute_autopilot_trade(self, result):
        import logging as _lg
        mode = self.config.get('trading', {}).get('automation_mode', 'assist').lower()
        base_max_pos = int(self.config.get('risk_management', {}).get('max_open_positions', 3))
        max_pos = base_max_pos
        is_manual_override = bool(self.config.get('risk_management', {}).get('manual_max_positions_override', False))
        tier_clamped = False
        t_name = ""
        if not is_manual_override and hasattr(self.orchestrator, 'risk') and hasattr(self.orchestrator.risk, 'get_effective_max_open_positions'):
            acc = self.orchestrator.mt5.get_account_info() if hasattr(self.orchestrator.mt5, 'get_account_info') else {}
            bal = float((acc or {}).get('balance', 0.0) or 0.0)
            eff_pos = self.orchestrator.risk.get_effective_max_open_positions(bal, base_max_pos)
            if eff_pos < base_max_pos:
                tier_clamped = True
                if hasattr(self.orchestrator.risk, 'get_account_tier'):
                    t_name = self.orchestrator.risk.get_account_tier(bal).get('name', 'Micro-Capital')
            max_pos = eff_pos
        exclude_closed = self.config.get('risk_management', {}).get('exclude_closed_markets_from_quota', True)

        allowed_types = self.config.get('trading', {}).get('allowed_market_types', None)
        open_positions = self.orchestrator.mt5.get_open_positions() or []
        active_positions = []
        closed_count = 0
        unallowed_count = 0
        for p in open_positions:
            sym = p.get('symbol', '')
            if allowed_types and hasattr(self.orchestrator.mt5, 'is_symbol_in_allowed_types'):
                if not self.orchestrator.mt5.is_symbol_in_allowed_types(sym, allowed_types):
                    unallowed_count += 1
                    continue
            if exclude_closed and hasattr(self.orchestrator.mt5, 'is_market_open'):
                if not self.orchestrator.mt5.is_market_open(sym):
                    closed_count += 1
                    continue
            active_positions.append(p)

        if len(active_positions) >= max_pos:
            symbol = result.get('symbol', 'Unknown')
            msg = f"⚠️ [Autopilot] Limite de {max_pos} positions actives atteinte ({len(active_positions)}/{max_pos})."
            notes = []
            if tier_clamped:
                notes.append(f"palier {t_name} actif ({max_pos} pos conseillée vs {base_max_pos} configurée)")
            if closed_count > 0:
                notes.append(f"{closed_count} sur marché fermé")
            if unallowed_count > 0:
                notes.append(f"{unallowed_count} hors types autorisés")
            if notes:
                msg += f" (Note: {', '.join(notes)})."
            msg += f" Exécution ignorée pour {symbol}."
            self.append_to_chat(msg)
            return

        decision = result.get('decision', {})
        action = str(decision.get('action', '')).upper()
        symbol = result.get('symbol', 'Unknown')

        # Vérification du sens de trading autorisé
        allowed_dir = getattr(self.orchestrator, 'allowed_trade_direction', 'BOTH').upper()
        if allowed_dir == 'BUY_ONLY' and action == 'SELL':
            self.append_to_chat(f"🚫 [Autopilot] {symbol} - Signal SELL ignoré car politique configurée sur BUY_ONLY (Hausse seule).")
            return
        elif allowed_dir == 'SELL_ONLY' and action == 'BUY':
            self.append_to_chat(f"🚫 [Autopilot] {symbol} - Signal BUY ignoré car politique configurée sur SELL_ONLY (Baisse seule).")
            return

        # In assist/semi-auto modes, only surface the signal for manual validation.
        if mode != 'auto':
            self.append_to_chat(f"💡 [Autopilot] Opportunité détectée sur {symbol} ({action}) — mode '{mode}' : signal affiché, exécution manuelle requise.")
            self.add_signal_widget(result)
            return

        self.append_to_chat(f"🚀 [Autopilot] Exécution automatique pour {symbol} ({action})...")

        request = self.trade_execution.prepare_order_request(result)
        if not request:
            self.append_to_chat(f"❌ [Autopilot] {symbol} - Impossible de préparer la requête MT5.")
            return

        check = self.trade_execution.check_order(request)
        if not check['valid']:
            self.append_to_chat(f"❌ [Autopilot] {symbol} - MT5 Check a échoué: {check['reason']}")
            return

        exec_result = self.trade_execution.execute_order(request)
        if exec_result['success']:
            deal_id = exec_result.get('deal_id', 0)
            ticket  = exec_result.get('ticket', deal_id)   # ticket de la position ouverte ou de l'ordre limite
            is_p = bool(decision.get('is_pending') or request.get('action') == 5)
            kind_txt = f"Ordre limite ({decision.get('order_type', 'LIMIT')}) placé" if is_p else f"Ordre marché exécuté (Deal #{deal_id})"
            self.append_to_chat(f"🚀 [Autopilot] {symbol} - {kind_txt} ! Ticket MT5: #{ticket}")
            if hasattr(self, 'voice_manager') and self.voice_manager and getattr(self.voice_manager, 'voice_announcements', True):
                action_fr = "d'achat" if action == "BUY" else "de vente"
                self.voice_manager.speak(f"Ordre {action_fr} exécuté sur {symbol}.")
            # Initialiser le compteur OB pour ce ticket (0 OB additionnel pour l'instant)
            self._ob_trades_by_ticket[ticket] = 0
            # --- Rescanning Order Block post-trade (désactivé par défaut pour éviter le sur-trading) ---
            try:
                rescan_enabled = self.config.get('trading', {}).get('post_trade_ob_rescan_enabled', False)
                acc = self.orchestrator.mt5.get_account_info() if hasattr(self.orchestrator.mt5, 'get_account_info') else {}
                bal = float((acc or {}).get('balance', 0.0) or 0.0)
                if rescan_enabled and bal >= 100.0:
                    allowed_types = (
                        self.config.get('trading', {}).get('allowed_market_types')
                        or self.config.get('trading', {}).get('allowed_types')
                    )
                    scanned_symbols = self.orchestrator.mt5.get_visible_symbols(
                        allowed_types=allowed_types
                    ) or [symbol]
                    direction_filter = action if action in ("BUY", "SELL") else "BOTH"
                    max_ob = int(self.config.get('risk_management', {}).get('max_ob_per_initial_trade', 1))
                    ob_thread = OBScanThread(
                        orchestrator=self.orchestrator,
                        symbols=scanned_symbols,
                        timeframe=self.config.get('trading', {}).get('timeframes', ['M15'])[0],
                        direction_filter=direction_filter,
                        passes=3,
                        interval_s=300,
                        initial_ticket=ticket,
                    )
                    ob_thread.ob_found.connect(
                        lambda results, tk=ticket: self.handle_ob_signals(results, initial_ticket=tk)
                    )
                    ob_thread.progress.connect(self.append_to_chat)
                    ob_thread.finished.connect(
                        lambda t=ob_thread: self._on_ob_thread_finished(t)
                    )
                    ob_thread.start()
                    if not hasattr(self, '_ob_threads'):
                        self._ob_threads = []
                    self._ob_threads.append(ob_thread)
                    self.append_to_chat(
                        f"[OB] Rescanning Order Block activé — 3 passes × {300//60} min "
                        f"sur {len(scanned_symbols)} symboles (ticket #{ticket}, max {max_ob} OB/trade)."
                    )
                else:
                    self.append_to_chat(
                        f"🛡️ [Autopilot] {symbol} (Ticket #{ticket}) en cours. "
                        f"Gestion automatique Stop/TP active (mode Sniper)."
                    )
            except Exception:
                _lg.getLogger(__name__).exception("Erreur demarrage OBScanThread")
        else:
            err_reason = str(exec_result.get('reason', ''))
            if "10018" in err_reason or "Market closed" in err_reason:
                if hasattr(self.orchestrator.mt5, 'mark_market_closed') and symbol:
                    self.orchestrator.mt5.mark_market_closed(symbol, duration_seconds=1800)
                self.append_to_chat(f"⏸️ [Autopilot] {symbol} - Marché fermé chez le courtier (Code 10018). Ordre annulé automatiquement.")
            else:
                self.append_to_chat(f"[Autopilot] {symbol} - Erreur d'execution: {exec_result['reason']}")

    def handle_ob_signals(self, ob_results: list, initial_ticket: int = 0):
        """
        Slot appelé par OBScanThread quand des Order Blocks sont détectés.

        initial_ticket : ticket du trade parent ayant déclenché ce rescan.
                         0 = inconnu (rescan manuel ou ancienne connexion).
        """
        import logging as _lg
        log = _lg.getLogger(__name__)
        mode = self.config.get('trading', {}).get('automation_mode', 'assist').lower()
        base_max_pos = int(self.config.get('risk_management', {}).get('max_open_positions', 3))
        max_pos = base_max_pos
        is_manual_override = bool(self.config.get('risk_management', {}).get('manual_max_positions_override', False))
        if not is_manual_override and hasattr(self.orchestrator, 'risk') and hasattr(self.orchestrator.risk, 'get_effective_max_open_positions'):
            acc = self.orchestrator.mt5.get_account_info() if hasattr(self.orchestrator.mt5, 'get_account_info') else {}
            bal = float((acc or {}).get('balance', 0.0) or 0.0)
            max_pos = self.orchestrator.risk.get_effective_max_open_positions(bal, base_max_pos)
        max_ob_per_trade = int(self.config.get('risk_management', {}).get('max_ob_per_initial_trade', 1))

        for res in ob_results:
            try:
                symbol = res.get('symbol', '?')
                decision = res.get('decision', {})
                action = decision.get('action', '?')
                score = decision.get('confidence', '?')
                self.append_to_chat(
                    f"[ORDER BLOCK] {symbol} - {action} détecté! Score {score}%"
                    + (f" (trade parent #{initial_ticket})" if initial_ticket else "")
                )

                if mode == 'auto':
                    # --- Direction filter check ---
                    allowed_dir = getattr(self.orchestrator, 'allowed_trade_direction', 'BOTH').upper()
                    if allowed_dir == 'BUY_ONLY' and str(action).upper() == 'SELL':
                        self.append_to_chat(f"🚫 [OB] {symbol} - Signal SELL ignoré car politique configurée sur BUY_ONLY (Hausse seule).")
                        continue
                    elif allowed_dir == 'SELL_ONLY' and str(action).upper() == 'BUY':
                        self.append_to_chat(f"🚫 [OB] {symbol} - Signal BUY ignoré car politique configurée sur SELL_ONLY (Baisse seule).")
                        continue

                    # --- Limite globale de positions ouvertes (excluant marchés fermés et types désactivés si configuré) ---
                    open_pos = self.orchestrator.mt5.get_open_positions() or []
                    exclude_closed = self.config.get('risk_management', {}).get('exclude_closed_markets_from_quota', True)
                    allowed_types = self.config.get('trading', {}).get('allowed_market_types', None)
                    active_pos = []
                    for p in open_pos:
                        sym = p.get('symbol', '')
                        if allowed_types and hasattr(self.orchestrator.mt5, 'is_symbol_in_allowed_types'):
                            if not self.orchestrator.mt5.is_symbol_in_allowed_types(sym, allowed_types):
                                continue
                        if exclude_closed and hasattr(self.orchestrator.mt5, 'is_market_open'):
                            if not self.orchestrator.mt5.is_market_open(sym):
                                continue
                        active_pos.append(p)

                    if len(active_pos) >= max_pos:
                        self.append_to_chat(
                            f"[OB] Limite globale de {max_pos} positions actives atteinte — OB ignoré pour {symbol}."
                        )
                        continue

                    # --- Limite d'OB additionnels par trade initial ---
                    if initial_ticket:
                        ob_count = self._ob_trades_by_ticket.get(initial_ticket, 0)
                        if ob_count >= max_ob_per_trade:
                            self.append_to_chat(
                                f"[OB] Limite de {max_ob_per_trade} OB/trade atteinte pour le ticket "
                                f"#{initial_ticket} — {symbol} ignoré."
                            )
                            continue

                    request = self.trade_execution.prepare_order_request(res)
                    if not request:
                        self.append_to_chat(f"[OB] {symbol} - Impossible de préparer la requête MT5.")
                        continue
                    check = self.trade_execution.check_order(request)
                    if not check['valid']:
                        self.append_to_chat(f"[OB] {symbol} - Check MT5 échoué: {check['reason']}")
                        continue
                    exec_result = self.trade_execution.execute_order(request)
                    if exec_result['success']:
                        self.append_to_chat(
                            f"[OB AUTO] {symbol} - Position OB ouverte! Deal: {exec_result['deal_id']}"
                        )
                        # Incrémenter le compteur OB pour ce ticket parent
                        if initial_ticket:
                            self._ob_trades_by_ticket[initial_ticket] = (
                                self._ob_trades_by_ticket.get(initial_ticket, 0) + 1
                            )
                    else:
                        self.append_to_chat(f"[OB AUTO] {symbol} - Échec exécution: {exec_result['reason']}")
                else:
                    # Mode assist/semi : afficher le signal dans l'UI pour validation manuelle
                    self.add_ob_signal_widget(res)
            except Exception:
                log.exception("[OB] Erreur traitement signal OB")

    def _on_ob_thread_finished(self, thread):
        """Nettoie un OBScanThread après sa fin pour éviter les fuites mémoire."""
        try:
            if hasattr(self, '_ob_threads') and thread in self._ob_threads:
                self._ob_threads.remove(thread)
            thread.deleteLater()
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("_on_ob_thread_finished a echoué")

    def _fmt_price(self, value) -> str:
        """Formate un prix de façon tolérante (None/str/numpy acceptés)."""
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.5f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_confidence(self, value) -> str:
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.0f}%"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_lot(self, value) -> str:
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.2f} lots"
        except (TypeError, ValueError):
            return str(value)


    def add_signal_widget(self, result):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        
        decision = result.get('decision', {})
        action = decision.get('action')
        symbol = result.get('symbol', 'Unknown')
        confidence = decision.get('confidence')
        entry = decision.get('entry')
        sl = decision.get('sl')
        tp = decision.get('tp')
        lot = decision.get('lot')
        reason = decision.get('reason', '')
        status = str(result.get('status', '')).upper()
        
        is_pending = bool(decision.get('is_pending') or status == 'PENDING_CRT' or decision.get('order_type') in ('BUY_LIMIT', 'SELL_LIMIT'))
        order_type = decision.get('order_type') or ('BUY_LIMIT' if action == 'BUY' else 'SELL_LIMIT' if is_pending else 'MARKET')
        
        # Order Flow confluence
        of_score = decision.get('order_flow_score')
        if of_score is None and result.get('order_flow'):
            of_score = getattr(result['order_flow'], 'confluence_score', None)
        of_confirmed = bool(decision.get('order_flow_confirmed') or (result.get('order_flow') and getattr(result['order_flow'], 'is_confirmed', False)))

        # Distinct style depending on pending vs market
        if is_pending:
            frame.setStyleSheet("QFrame { background-color: #152238; border: 2px solid #00bcd4; border-radius: 6px; margin-bottom: 10px; }")
            tag_header = f"📌 ORDRE LIMITE PENDING [{order_type}] : {action} sur {symbol}"
        elif status == 'MARKET_CRT':
            frame.setStyleSheet("QFrame { background-color: #1a2c20; border: 2px solid #4caf50; border-radius: 6px; margin-bottom: 10px; }")
            tag_header = f"🎯 CRT MARCHÉ DIRECT : {action} sur {symbol}"
        else:
            color = "🟢" if action == "BUY" else "🔴"
            frame.setStyleSheet("QFrame { background-color: #2b2b2b; border: 1px solid #444; border-radius: 6px; margin-bottom: 10px; }")
            tag_header = f"{color} SIGNAL AU MARCHÉ : {action} sur {symbol}"

        layout = QVBoxLayout(frame)
        
        of_line = ""
        if of_score is not None:
            of_obj = result.get('order_flow')
            dq = getattr(of_obj, 'data_quality', None) if of_obj else None
            if not dq and isinstance(decision, dict):
                dq = decision.get('order_flow_data_quality')

            dq_label = ""
            if dq == 'REAL_OBSERVED':
                dq_label = " [Flux Réel ECN]"
            elif dq == 'ESTIMATED_QUOTE_RULE':
                dq_label = " [Quote-Rule]"
            elif dq == 'DEGRADED_STALE':
                dq_label = " [⚠️ Ticks Périmés]"

            badge_of = "🌊 ORDER FLOW : CONFIRMÉ" if of_confirmed else "📊 ORDER FLOW : ANALYSÉ"
            of_line = f"\n{badge_of}{dq_label} (Score: {float(of_score):.1f}/100)"
            
        exp_line = ""
        if is_pending:
            bars = decision.get('expiration_bars', 4)
            exp_line = f"\nExpiration : {bars} bougies (annulation auto si non touché)"

        display_text = f"{tag_header}\n\n" \
                       f"Confiance  : {self._fmt_confidence(confidence)}\n" \
                       f"Entrée     : {self._fmt_price(entry)}\n" \
                       f"Stop Loss  : {self._fmt_price(sl)}\n" \
                       f"Take Profit: {self._fmt_price(tp)}\n" \
                       f"Volume     : {self._fmt_lot(lot)}" \
                       f"{exp_line}" \
                       f"{of_line}\n\n" \
                       f"Justification : {reason}"
                       
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setPlainText(display_text)
        text_edit.setMinimumHeight(160)
        layout.addWidget(text_edit)
        
        btn_layout = QHBoxLayout()

        # --- Lot manuel (override) ---
        fixed_lot_cfg = self.config.get('risk_management', {}).get('fixed_lot_size', 0.0)
        suggested_lot = float(lot or fixed_lot_cfg or 0.01)
        lot_label = QLabel("Volume :")
        lot_label.setStyleSheet("color: #ffd700; font-weight: bold;")
        lot_spin = QDoubleSpinBox()
        lot_spin.setRange(0.01, 100.0)
        lot_spin.setSingleStep(0.01)
        lot_spin.setDecimals(2)
        lot_spin.setValue(suggested_lot)
        lot_spin.setToolTip("Volume à trader. En Autopilot le robot calcule automatiquement selon le % de risque ou le volume fixe des Paramètres.")
        lot_spin.setFixedWidth(90)
        lot_spin.setStyleSheet("background: #1a1a2e; color: #ffd700; border: 1px solid #ffd700; border-radius: 4px; padding: 2px;")
        btn_layout.addWidget(lot_label)
        btn_layout.addWidget(lot_spin)
        btn_layout.addSpacing(10)
        # ---

        take_label = f"📌 PLACER {order_type}" if is_pending else "⚡ PRENDRE POSITION"
        take_btn = QPushButton(take_label)
        btn_bg = "#00838f" if is_pending else "#2e7d32"
        take_btn.setStyleSheet(f"background-color: {btn_bg}; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        take_btn.clicked.connect(lambda checked=False, res=result, f=frame, btn=take_btn, ls=lot_spin: self.on_take_trade(res, f, btn, manual_lot=ls.value()))
        
        ignore_btn = QPushButton("IGNORER")
        ignore_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        ignore_btn.clicked.connect(lambda checked=False, f=frame: f.deleteLater())
        
        btn_layout.addWidget(take_btn)
        btn_layout.addWidget(ignore_btn)
        layout.addLayout(btn_layout)
        
        self.scroll_layout.addWidget(frame)

    def add_ob_signal_widget(self, result):
        """Affiche un signal Order Block dans l'UI (mode manuel ou assist)."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        # Couleur distincte pour les OB : fond violette/indigo
        frame.setStyleSheet("QFrame { background-color: #2a1a4a; border: 2px solid #8a2be2; border-radius: 5px; margin-bottom: 10px; }")
        layout = QVBoxLayout(frame)

        decision = result.get('decision', {})
        action = decision.get('action', '?')
        symbol = result.get('symbol', 'Unknown')
        confidence = decision.get('confidence')
        entry = decision.get('entry')
        sl = decision.get('sl')
        tp = decision.get('tp')
        lot = decision.get('lot')
        reason = decision.get('reason', '')
        ob_summary = result.get('ob_summary', '')
        ob_zone_low = decision.get('ob_zone_low')
        ob_zone_high = decision.get('ob_zone_high')

        tag = "[OB BUY]" if action == "BUY" else "[OB SELL]"
        display_text = (
            f"{tag} ORDER BLOCK {action} sur {symbol}\n"
            f"Zone OB   : [{self._fmt_price(ob_zone_low)} - {self._fmt_price(ob_zone_high)}]\n"
            f"Confiance : {self._fmt_confidence(confidence)}\n"
            f"Entree    : {self._fmt_price(entry)}\n"
            f"Stop Loss : {self._fmt_price(sl)}\n"
            f"Tp        : {self._fmt_price(tp)}\n"
            f"Volume    : {self._fmt_lot(lot)}\n\n"
            f"Details   : {reason}"
        )

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setStyleSheet("color: #c9b8ff; background-color: #1e1030;")
        text_edit.setPlainText(display_text)
        text_edit.setMinimumHeight(160)
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()

        # --- Lot manuel (override OB) ---
        fixed_lot_cfg = self.config.get('risk_management', {}).get('fixed_lot_size', 0.0)
        suggested_lot_ob = float(lot or fixed_lot_cfg or 0.01)
        lot_label_ob = QLabel("Volume :")
        lot_label_ob.setStyleSheet("color: #c9b8ff; font-weight: bold;")
        lot_spin_ob = QDoubleSpinBox()
        lot_spin_ob.setRange(0.01, 100.0)
        lot_spin_ob.setSingleStep(0.01)
        lot_spin_ob.setDecimals(2)
        lot_spin_ob.setValue(suggested_lot_ob)
        lot_spin_ob.setFixedWidth(90)
        lot_spin_ob.setStyleSheet("background: #1e1030; color: #c9b8ff; border: 1px solid #8a2be2; border-radius: 4px; padding: 2px;")
        btn_layout.addWidget(lot_label_ob)
        btn_layout.addWidget(lot_spin_ob)
        btn_layout.addSpacing(10)
        # ---

        take_btn = QPushButton("PRENDRE OB")
        take_btn.setStyleSheet("background-color: #8a2be2; color: white; font-weight: bold; padding: 10px;")
        take_btn.clicked.connect(lambda checked=False, res=result, f=frame, btn=take_btn, ls=lot_spin_ob: self.on_take_trade(res, f, btn, manual_lot=ls.value()))
        ignore_btn = QPushButton("IGNORER")
        ignore_btn.setStyleSheet("background-color: #555; font-weight: bold; padding: 10px;")
        ignore_btn.clicked.connect(lambda checked=False, f=frame: f.deleteLater())
        btn_layout.addWidget(take_btn)
        btn_layout.addWidget(ignore_btn)
        layout.addLayout(btn_layout)

        self.scroll_layout.addWidget(frame)

    def add_pending_ob_widget(self, result):
        """Affiche un signal PENDING Order Block dans l'UI (pour placement manuel d'ordre limite)."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background-color: #1a2a4a; border: 2px solid #4a9eff; "
            "border-radius: 5px; margin-bottom: 10px; }"
        )
        layout = QVBoxLayout(frame)

        decision = result.get('decision', {})
        action = decision.get('action', '?')
        symbol = result.get('symbol', 'Unknown')
        confidence = decision.get('confidence')
        entry = decision.get('entry')
        sl = decision.get('sl')
        tp = decision.get('tp')
        lot = decision.get('lot')
        reason = decision.get('reason', '')
        ob_summary = result.get('ob_summary', '')
        # Nom personnalisé : modifiable dans l'UI, envoyé à MT5 comme commentaire.
        default_label = str(result.get('pending_label', '') or '').strip()
        if not default_label:
            default_label = f"OB {symbol} {action}"

        tag = "📌 [PENDING BUY]" if action == "BUY" else "📌 [PENDING SELL]"
        # ── Champ nom modifiable ──
        name_row = QHBoxLayout()
        name_lbl = QLabel("📝 Nom du Pending Block :")
        name_lbl.setStyleSheet("color: #a0c4ff; font-weight: bold;")
        name_edit = QLineEdit()
        name_edit.setText(default_label)
        name_edit.setMaxLength(31)  # limite commentaire MT5
        name_edit.setPlaceholderText("Ex : OB EURUSD 1, Mon bloc London...")
        name_edit.setToolTip(
            "Nom personnalisé du Pending Block.\n"
            "Utilisé comme commentaire de l'ordre MT5 (31 car. max)\n"
            "et réutilisé pour l'annonce vocale."
        )
        name_edit.setStyleSheet(
            "background-color: #0d1830; color: #ffffff; "
            "border: 1px solid #4a9eff; border-radius: 4px; padding: 6px;"
        )
        name_row.addWidget(name_lbl)
        name_row.addWidget(name_edit, stretch=1)
        layout.addLayout(name_row)
        display_text = (
            f"{tag} ORDRE LIMITE sur {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Le prix n'a PAS encore retesté cette zone.\n"
            f"Un ordre LIMITE sera placé pour entrer automatiquement\n"
            f"quand le prix reviendra dans la zone OB.\n\n"
            f"Confiance : {self._fmt_confidence(confidence)}\n"
            f"Entrée    : {self._fmt_price(entry)} (Ordre Limite)\n"
            f"Stop Loss : {self._fmt_price(sl)}\n"
            f"Take Profit: {self._fmt_price(tp)}\n"
            f"Volume    : {self._fmt_lot(lot)}\n\n"
            f"Détails   : {reason}\n\n"
            f"─── Résumé OB ───\n{ob_summary}"
        )

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setStyleSheet("color: #a0c4ff; background-color: #0d1830;")
        text_edit.setPlainText(display_text)
        text_edit.setMinimumHeight(200)
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()

        # --- Lot manuel (override pour pending) ---
        fixed_lot_cfg = self.config.get('risk_management', {}).get('fixed_lot_size', 0.0)
        suggested_lot = float(lot or fixed_lot_cfg or 0.01)
        lot_label = QLabel("Volume :")
        lot_label.setStyleSheet("color: #a0c4ff; font-weight: bold;")
        lot_spin = QDoubleSpinBox()
        lot_spin.setRange(0.01, 100.0)
        lot_spin.setSingleStep(0.01)
        lot_spin.setDecimals(2)
        lot_spin.setValue(suggested_lot)
        lot_spin.setFixedWidth(90)
        lot_spin.setStyleSheet("background: #0d1830; color: #a0c4ff; border: 1px solid #4a9eff; border-radius: 4px; padding: 2px;")
        btn_layout.addWidget(lot_label)
        btn_layout.addWidget(lot_spin)
        btn_layout.addSpacing(10)

        place_btn = QPushButton("📌 PLACER ORDRE LIMITE")
        place_btn.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; padding: 10px;"
        )
        place_btn.clicked.connect(
            lambda checked=False, res=result, f=frame, btn=place_btn, ne=name_edit, ls=lot_spin: self.on_place_pending_ob(res, f, btn, ne, manual_lot=ls.value())
        )
        ignore_btn = QPushButton("IGNORER")
        ignore_btn.setStyleSheet("background-color: #555; font-weight: bold; padding: 10px;")
        ignore_btn.clicked.connect(lambda checked=False, f=frame: f.deleteLater())
        btn_layout.addWidget(place_btn)
        btn_layout.addWidget(ignore_btn)
        layout.addLayout(btn_layout)

        self.scroll_layout.addWidget(frame)

    def on_place_pending_ob(self, result, frame, btn, name_edit=None, manual_lot: float = None):
        """Place un ordre limite (pending) sur une zone Order Block."""
        btn.setEnabled(False)

        decision = result.get('decision', {})
        symbol = result.get('symbol', 'Unknown')
        ob_data = result.get('ob_data')
        htf_trend = result.get('htf_trend', 'Unknown')

        # Nom personnalisé saisi dans le champ modifiable de la carte UI.
        custom_label = ""
        try:
            if name_edit is not None and hasattr(name_edit, "text"):
                custom_label = str(name_edit.text() or "").strip()[:31]
        except Exception:
            custom_label = ""
        if custom_label:
            result['pending_label'] = custom_label
        
        if not ob_data:
            QMessageBox.warning(self, "Erreur", "Données de l'Order Block manquantes.")
            btn.setEnabled(True)
            return

        # Calculate lot from manual input, fixed lot config, or decision
        fixed_lot_cfg = float(self.config.get('risk_management', {}).get('fixed_lot_size', 0.0) or 0.0)
        if manual_lot is not None and manual_lot > 0:
            lot = round(float(manual_lot), 2)
        elif fixed_lot_cfg > 0.0:
            lot = fixed_lot_cfg
        else:
            lot = float(decision.get('lot', 0.01) or 0.01)

        # Place the pending order
        pending_mgr = self._get_pending_manager()
        if not pending_mgr:
            QMessageBox.warning(self, "Erreur", "Gestionnaire de pending orders non disponible.")
            btn.setEnabled(True)
            return

        pending = pending_mgr.place_ob_pending(
            symbol=symbol,
            ob=ob_data,
            timeframe=result.get('timeframe', 'M15'),
            lot=lot,
            htf_trend=htf_trend,
            label=custom_label,
        )

        if pending:
            label_txt = f" « {pending.label} »" if getattr(pending, 'label', '') else ""
            self.append_to_chat(
                f"📌 [PENDING] Ordre limite {pending.direction} placé sur {symbol}"
                f"{label_txt} "
                f"@ {pending.entry_price:.5f} (ticket #{pending.ticket})"
            )
            if hasattr(self, 'voice_manager') and self.voice_manager and getattr(self.voice_manager, 'voice_announcements', True):
                action_fr = "d'achat" if pending.direction == "BUY" else "de vente"
                spoken_name = f" nommé {pending.label}." if getattr(pending, 'label', '') else "."
                self.voice_manager.speak(
                    f"Ordre limite {action_fr} placé sur {symbol}{spoken_name} en attente du retest."
                )
            frame.deleteLater()
        else:
            QMessageBox.warning(
                self, "Échec",
                f"Impossible de placer l'ordre limite sur {symbol}.\n"
                "Vérifiez les logs pour plus de détails."
            )
            btn.setEnabled(True)

    def execute_pending_ob_trade(self, result):
        """[Autopilot] Place automatiquement un ordre limite sur une zone OB."""
        import logging as _lg

        decision = result.get('decision', {})
        symbol = result.get('symbol', 'Unknown')
        action = str(decision.get('action', '')).upper()
        ob_data = result.get('ob_data')
        htf_trend = result.get('htf_trend', 'Unknown')

        if not ob_data:
            self.append_to_chat(f"❌ [PENDING] {symbol} — Données OB manquantes. Ignoré.")
            return

        pending_mgr = self._get_pending_manager()
        if not pending_mgr:
            self.append_to_chat(f"❌ [PENDING] Gestionnaire non disponible pour {symbol}.")
            return

        # Vérifier le quota d'ordres limites actifs
        active_pendings = pending_mgr.get_active_pendings()
        max_pendings = getattr(pending_mgr, 'max_pending_total', 5)
        if len(active_pendings) >= max_pendings:
            self.append_to_chat(f"ℹ️ [PENDING] Quota de {max_pendings} ordre(s) limite(s) actif(s) déjà atteint. Ordre ignoré pour {symbol}.")
            return

        # Calculate lot from decision (validated by risk manager) or fixed lot config
        lot = float(decision.get('lot') or 0.0)
        if lot <= 0:
            fixed_lot_cfg = float(self.config.get('risk_management', {}).get('fixed_lot_size', 0.0) or 0.0)
            lot = fixed_lot_cfg if fixed_lot_cfg > 0 else 0.01

        pending = pending_mgr.place_ob_pending(
            symbol=symbol,
            ob=ob_data,
            timeframe=result.get('timeframe', 'M15'),
            lot=lot,
            htf_trend=htf_trend,
            label=str(result.get('pending_label', '') or ''),
        )

        if pending:
            label_txt = f" « {pending.label} »" if getattr(pending, 'label', '') else ""
            self.append_to_chat(
                f"📌 [Autopilot PENDING] {action} LIMIT placé sur {symbol}{label_txt} "
                f"@ {pending.entry_price:.5f} (ticket #{pending.ticket}, "
                f"SL={pending.sl:.5f}, TP={pending.tp:.5f})"
            )
            if hasattr(self, 'voice_manager') and self.voice_manager and getattr(self.voice_manager, 'voice_announcements', True):
                spoken_name = f" nommé {pending.label}." if getattr(pending, 'label', '') else "."
                self.voice_manager.speak(
                    f"Ordre limite {action.lower()} placé sur {symbol}{spoken_name}"
                )
        else:
            self.append_to_chat(
                f"⚠️ [PENDING] {symbol} — Ordre limite non placé (conditions non remplies)."
            )

    def _get_pending_manager(self):
        """Lazily initializes and returns the PendingOrderManager."""
        if not hasattr(self, '_pending_order_manager') or self._pending_order_manager is None:
            try:
                from src.trading.pending_order_manager import PendingOrderManager
                self._pending_order_manager = PendingOrderManager(
                    mt5_connector=self.orchestrator.mt5,
                    ob_detector=self.orchestrator.ob_detector,
                    tech_indicators=self.orchestrator.tech,
                    market_data=self.orchestrator.market_data,
                    risk_manager=self.orchestrator.risk,
                )
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).exception("Failed to create PendingOrderManager")
                return None
        return self._pending_order_manager

    def rename_pending_order(self, ticket: int, new_label: str) -> bool:
        """Renomme un Pending Block déjà placé (ticket MT5) depuis l'UI ou la voix."""
        mgr = self._get_pending_manager()
        if not mgr:
            return False
        ok = mgr.rename_pending(ticket, new_label)
        if ok:
            clean = str(new_label or "").strip()[:31]
            self.append_to_chat(f"✏️ Pending #{ticket} renommé en « {clean} ».")
            try:
                self.refresh_positions()
            except Exception:
                pass
        return ok

        
    def on_take_trade(self, signal, frame, take_btn, manual_lot: float = None):
        take_btn.setEnabled(False)
        # Override lot si l'utilisateur a saisi un volume manuellement
        if manual_lot is not None and manual_lot > 0:
            import copy
            signal = copy.deepcopy(signal)
            if 'decision' not in signal:
                signal['decision'] = {}
            signal['decision']['lot'] = round(manual_lot, 2)
            # Désactiver le calcul auto en passant le lot comme volume fixe temporaire
            signal['decision']['manual_lot_override'] = True
        request = self.trade_execution.prepare_order_request(signal)
        if not request:
            QMessageBox.critical(self, "Erreur", "Impossible de préparer la requête.")
            take_btn.setEnabled(True)
            return
            
        check = self.trade_execution.check_order(request)
        if not check['valid']:
            QMessageBox.warning(self, "Ordre Rejeté", f"MT5 Check a échoué:\n{check['reason']}")
            take_btn.setEnabled(True)
            return
            
        # Confirm and Execute
        decision = signal.get('decision', {})
        is_pending = bool(decision.get('is_pending') or request.get('action') == 5)
        action_name = decision.get('action', '')
        order_type_str = decision.get('order_type') or ('BUY_LIMIT' if action_name == 'BUY' else 'SELL_LIMIT' if is_pending else 'MARCHÉ')
        lot_display = f"{decision.get('lot', '?')} lot" if manual_lot else "volume calculé auto"
        
        if is_pending:
            prompt_msg = (
                f"Voulez-vous vraiment placer cet ORDRE LIMITE PENDING ({order_type_str}) sur {signal.get('symbol')} ?\n\n"
                f"• Prix Limite : {decision.get('entry')}\n"
                f"• Stop Loss   : {decision.get('sl')}\n"
                f"• Take Profit : {decision.get('tp')}\n"
                f"• Volume      : {lot_display}"
            )
        else:
            prompt_msg = f"Voulez-vous vraiment exécuter cette POSITION AU MARCHÉ sur {signal.get('symbol')} ?\n• Volume : {lot_display}"

        reply = QMessageBox.question(self, "Confirmation", prompt_msg, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            exec_result = self.trade_execution.execute_order(request)
            if exec_result['success']:
                ticket_id = exec_result.get('ticket') or exec_result.get('deal_id') or 0
                deal_id = exec_result.get('deal_id', 0)
                if is_pending:
                    msg_txt = f"Ordre Limite {order_type_str} placé avec succès !\nTicket MT5: #{ticket_id}"
                else:
                    msg_txt = f"Ordre au marché exécuté avec succès !\nTicket MT5: #{ticket_id} (Deal #{deal_id})"
                QMessageBox.information(self, "Succès", msg_txt)
                frame.deleteLater() # Remove from list if taken successfully
            else:
                err_reason = str(exec_result.get('reason', ''))
                sym = signal.get('symbol', '')
                if "10018" in err_reason or "Market closed" in err_reason:
                    if hasattr(self.orchestrator.mt5, 'mark_market_closed') and sym:
                        self.orchestrator.mt5.mark_market_closed(sym, duration_seconds=1800)
                    QMessageBox.warning(
                        self,
                        "Marché fermé",
                        f"Le marché pour {sym} est actuellement fermé chez votre courtier (Code 10018).\n\n"
                        f"Cette opportunité a été automatiquement retirée de la liste."
                    )
                    self.append_to_chat(f"⏸️ [Marché Fermé] {sym} est actuellement fermé chez le courtier (Code 10018). Opportunité ignorée.")
                    frame.deleteLater()
                else:
                    QMessageBox.critical(self, "Erreur d'exécution", exec_result['reason'])
                    take_btn.setEnabled(True)
        else:
            take_btn.setEnabled(True)

    def on_send_chat(self):
        message = self.chat_input.text().strip()
        if not message:
            return
        # Prevent sending while a reply is still being fetched
        if hasattr(self, 'chat_thread') and self.chat_thread.isRunning():
            return
            
        self.chat_input.clear()
        self.chat_display.append(f"<b>Toi :</b> {message}")

        # Commandes directes de diagnostic MT5 et rapport A/B
        msg_cmd = message.strip().lower()
        if msg_cmd.startswith(("/diag", "/checkmt5", "/marketdata", "audit mt5", "diagnostic mt5")):
            parts = message.split()
            sym = parts[1].upper() if len(parts) > 1 else (getattr(self, 'active_symbol', None) or "EURUSD")
            try:
                diag = self.market_data.diagnose_market_data(sym)
                tier_badge = {
                    "TIER_A_INSTITUTIONAL": "🏆 TIER A (Institutionnel / Futures)",
                    "TIER_B_FOREX_ECN": "⚖️ TIER B (Forex ECN / Quote-Rule)",
                    "TIER_C_LIMITED": "⚠️ TIER C (Limité / Fallback)",
                }.get(diag['tier'], diag['tier'])

                diag_html = (
                    f"<b>🩺 Diagnostic MT5 des Données Marché : {sym}</b><br>"
                    f"• <b>Niveau de Qualité :</b> {tier_badge}<br>"
                    f"• <b>Qualité Détectée :</b> {diag['data_quality']}<br>"
                    f"• <b>Carnet d'Ordres (DOM L2) :</b> {'✅ Disponible (' + str(diag['dom_levels']) + ' niveaux)' if diag['dom_available'] else '❌ Non fourni par le courtier'}<br>"
                    f"• <b>Fraîcheur des Ticks :</b> {'✅ Ticks récents (' + str(round(diag['ticks_stale_seconds'], 1)) + 's)' if diag['ticks_fresh'] else '⚠️ Périmé / Pas de ticks récents (' + str(round(diag['ticks_stale_seconds'], 1)) + 's)'}<br>"
                    f"• <b>Drapeaux Volume Réel :</b> {round(diag['real_flags_ratio'] * 100, 1)}% ({'✅ Vrai flux transactionnel' if diag['has_real_volume'] else 'ℹ️ Estimation Quote-Rule active'})<br>"
                    f"• <b>Recommandation Sécurité :</b> {diag['recommendation']}"
                )
                self.append_to_chat(diag_html, sender="IA")
                return
            except Exception as _de:
                self.append_to_chat(f"❌ Erreur lors du diagnostic de {sym} : {_de}", sender="IA")
                return

        if msg_cmd.startswith(("/ab", "/ab_test", "comparaison crt", "stats ab")):
            try:
                ab_report = self.order_flow_journal.compare_ab_performance()
                report_md = ab_report.get('markdown_report', 'Rapport A/B indisponible.')
                self.append_to_chat(f"<pre style='font-family: monospace;'>{report_md}</pre>", sender="IA")
                return
            except Exception as _ae:
                self.append_to_chat(f"❌ Erreur rapport A/B : {_ae}", sender="IA")
                return

        # 1. Dispatching de commande vocale directe (Autopilot, Profils, Modes, Fermeture, Solde, etc.)
        try:
            from src.core.voice_commands import VoiceCommandDispatcher
            direct_result = VoiceCommandDispatcher.dispatch(message, self)
            if direct_result is not None:
                reply = direct_result.get("reply", "")
                spoken = direct_result.get("spoken", reply)
                self.append_to_chat(reply, sender="IA")
                if hasattr(self, 'voice_manager') and self.voice_manager and not self.voice_manager.is_muted and spoken:
                    self.voice_manager.speak(spoken)
                return
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de l'exécution de la commande vocale")

        self.chat_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        ctx = self._get_advisor_context()
        self.chat_thread = ChatThread(
            self.chat_engine,
            message,
            config_advisor=getattr(self, 'config_advisor', None),
            context=ctx
        )
        self.chat_thread.chat_finished.connect(self.handle_chat_response)
        self.chat_thread.start()
        
    def _save_risk_param_to_config(self, key: str, value: float):
        """Met à jour un paramètre de risque en mémoire et sur disque sans bloquer."""
        try:
            if 'risk_management' not in self.config:
                self.config['risk_management'] = {}
            self.config['risk_management'][key] = value

            from pathlib import Path
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de la sauvegarde du parametre %s", key)

    def _refresh_tp_buttons(self, current_val: float):
        """Met à jour l'apparence des boutons de TP rapide pour refléter la valeur actuelle."""
        if not hasattr(self, 'tp_preset_buttons'):
            return
        for amt, btn in self.tp_preset_buttons.items():
            is_active = abs(current_val - amt) < 0.01
            if is_active:
                if amt == 0.0:
                    btn.setStyleSheet("background-color: #1565c0; color: #ffffff; border: 1px solid #2196f3; border-radius: 3px; font-weight: bold; padding: 2px 6px;")
                else:
                    btn.setStyleSheet("background-color: #2e7d32; color: #ffffff; border: 1px solid #4caf50; border-radius: 3px; font-weight: bold; padding: 2px 6px;")
            else:
                btn.setStyleSheet("background-color: #1e222d; color: #90caf9; border: 1px solid #374151; border-radius: 3px; font-weight: normal; padding: 2px 6px;")

    def set_quick_tp_target(self, target: float):
        """Définit la cible de take profit fixe en $ par trade et synchronise l'UI."""
        target = max(0.0, float(target))
        if hasattr(self, 'quick_tp_spin'):
            self.quick_tp_spin.blockSignals(True)
            self.quick_tp_spin.setValue(target)
            self.quick_tp_spin.blockSignals(False)
        self.on_quick_tp_changed(target)
        if target > 0.0:
            self.append_to_chat(f"🎯 <b>Cible TP par trade :</b> Fermeture automatique programmée à <b>+{target:.2f} $</b>.")
        else:
            self.append_to_chat("🏃 <b>Mode Trailing / Libre :</b> Aucun TP fixe en $, gain sécurisé dynamiquement.")

    def _refresh_quick_lot_button(self):
        """Met à jour le bouton Lot Fixe / Auto de la barre rapide."""
        if not hasattr(self, 'quick_lot_btn'):
            return
        fixed_lot = float(self.config.get('risk_management', {}).get('fixed_lot_size', 0.0) or 0.0)
        if fixed_lot > 0.0:
            self.quick_lot_btn.setText(f"⚖️ LOT FIXE : {fixed_lot:.2f}")
            self.quick_lot_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0284c7;
                    color: #ffffff;
                    border: 2px solid #38bdf8;
                    border-radius: 5px;
                    padding: 4px 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0369a1;
                    border: 2px solid #7dd3fc;
                }
            """)
        else:
            self.quick_lot_btn.setText("⚖️ LOT : AUTO (%)")
            self.quick_lot_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e222d;
                    color: #38bdf8;
                    border: 1px solid #0284c7;
                    border-radius: 5px;
                    padding: 4px 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0f172a;
                    color: #7dd3fc;
                    border: 1px solid #38bdf8;
                }
            """)

    def open_fixed_lot_advisor(self):
        """Ouvre la boîte de dialogue Conseiller Lot Fixe depuis la fenêtre principale et applique le choix."""
        cap = 81.03
        mf = 81.03
        lev = 500
        try:
            if getattr(self, 'orchestrator', None) and hasattr(self.orchestrator, 'mt5'):
                acc = self.orchestrator.mt5.get_account_info() or {}
                bal = float(acc.get('balance', 0.0) or 0.0)
                eq = float(acc.get('equity', bal) or bal)
                if bal > 0:
                    cap = min(bal, eq) if eq > 0 else bal
                    mf = float(acc.get('margin_free', cap) or cap)
                    lev = int(acc.get('leverage', 500) or 500)
        except Exception:
            pass

        curr_fixed = float(self.config.get('risk_management', {}).get('fixed_lot_size', 0.0) or 0.0)
        dlg = FixedLotAdvisorDialog(
            capital=cap,
            margin_free=mf,
            leverage=lev,
            current_fixed_lot=curr_fixed,
            parent=self,
        )
        if dlg.exec():
            chosen = round(float(dlg.selected_lot), 2)
            self._save_risk_param_to_config('fixed_lot_size', chosen)
            if getattr(self, 'orchestrator', None) and getattr(self.orchestrator, 'risk', None):
                self.orchestrator.risk.fixed_lot_size = chosen
            if getattr(self, 'trade_execution', None):
                self.trade_execution.fixed_lot_size = chosen
            self._refresh_quick_lot_button()
            self.update_tier_badge()

            if chosen > 0.0:
                self.append_to_chat(
                    f"⚖️ <b>Mode Lot Fixe ACTIVÉ : {chosen:.2f} lot</b> (calibré sur capital de <b>{cap:.2f} $</b>).<br>"
                    f"Le bot utilisera désormais <b>{chosen:.2f} lot</b> sur vos prochains trades au lieu de rester bloqué à 0.01 lot."
                )
            else:
                self.append_to_chat(
                    "🤖 <b>Mode Lot Dynamique ACTIVÉ (0.00) :</b> Le bot calcule le lot selon le % de risque par trade."
                )

    def _refresh_sniper_button(self, is_sniper: bool):
        if not hasattr(self, 'sniper_toggle_btn'):
            return
        if is_sniper:
            self.sniper_toggle_btn.setText("🎯 SNIPER : ACTIF (A+)")
            self.sniper_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #b71c1c;
                    color: #fffde7;
                    border: 2px solid #ff5722;
                    border-radius: 5px;
                    padding: 4px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                    border: 2px solid #ff9800;
                }
            """)
        else:
            self.sniper_toggle_btn.setText("🎯 SNIPER : OFF")
            self.sniper_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e222d;
                    color: #9e9e9e;
                    border: 1px solid #424242;
                    border-radius: 5px;
                    padding: 4px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #262b3d;
                    color: #ff9800;
                    border: 1px solid #ff9800;
                }
            """)

    def on_toggle_sniper_mode(self, checked: bool = None):
        """Active ou désactive le Mode Sniper ultra-sélectif."""
        if checked is None:
            is_active = self.sniper_toggle_btn.isChecked() if hasattr(self, 'sniper_toggle_btn') else False
        else:
            is_active = bool(checked)
            if hasattr(self, 'sniper_toggle_btn'):
                self.sniper_toggle_btn.blockSignals(True)
                self.sniper_toggle_btn.setChecked(is_active)
                self.sniper_toggle_btn.blockSignals(False)

        target_profile = "SNIPER" if is_active else "BALANCED"

        # 1. Update risk manager & orchestrator
        if getattr(self, 'orchestrator', None) and getattr(self.orchestrator, 'risk', None):
            self.orchestrator.risk.apply_risk_profile(target_profile)
            if hasattr(self.orchestrator, 'local_brain') and self.orchestrator.local_brain:
                self.orchestrator.local_brain.min_confidence = self.orchestrator.risk.min_confidence
        if getattr(self, 'risk_manager', None):
            self.risk_manager.apply_risk_profile(target_profile)
        if getattr(self, 'trade_execution', None):
            from src.trading.risk_manager import RISK_PROFILES
            if target_profile in RISK_PROFILES:
                self.trade_execution.min_confidence = RISK_PROFILES[target_profile]["min_confidence"]

        # 2. Update config
        if 'risk_management' not in self.config:
            self.config['risk_management'] = {}
        self.config['risk_management']['risk_profile'] = target_profile
        self._save_risk_param_to_config('risk_profile', target_profile)

        # 3. Update UI button & combo if present
        self._refresh_sniper_button(is_active)
        if hasattr(self, 'risk_profile_combo'):
            idx = self.risk_profile_combo.findData(target_profile)
            if idx >= 0:
                self.risk_profile_combo.blockSignals(True)
                self.risk_profile_combo.setCurrentIndex(idx)
                self.risk_profile_combo.blockSignals(False)

        # 4. Feedback
        if is_active:
            msg = (
                "🎯 <b>Mode Sniper ACTIVÉ (Confluence A+) :</b><br>"
                "• <b>Sélectivité maximale</b> : Le bot patiente et filtre impitoyablement le bruit.<br>"
                "• <b>Critères stricts</b> : Order Block retesté + Boomerang + Tendance EMA + Pullback RSI (Confiance >= 75%).<br>"
                "• <b>Volume optimisé</b> : Risque calibré à <b>2.5%</b> avec Stop Loss serré et R:R >= 2.0 pour des gains percutants."
            )
            spoken = "Mode Sniper activé. Le robot patiente pour une confluence parfaite A plus avec volume optimisé."
        else:
            msg = "⚖️ <b>Mode Sniper DÉSACTIVÉ :</b> Retour au profil Équilibré standard (Risque 1.0%, seuil de confiance 60%)."
            spoken = "Mode Sniper désactivé. Retour au profil équilibré standard."

        self.append_to_chat(msg)
        if hasattr(self, 'voice_manager') and self.voice_manager:
            self.voice_manager.speak(spoken)

    def on_quick_tp_changed(self, val: float):
        target = float(val)
        if getattr(self, 'trade_manager', None):
            self.trade_manager.take_profit_target_per_trade_usd = target
        self._save_risk_param_to_config('take_profit_target_per_trade_usd', target)
        self._refresh_tp_buttons(target)

    def on_quick_giveback_changed(self, val: float):
        gb = float(val)
        if getattr(self, 'trade_manager', None):
            self.trade_manager.giveback_usd_small = gb
        self._save_risk_param_to_config('giveback_usd_small', gb)

    def on_quick_daily_changed(self, val: float):
        daily = float(val)
        if getattr(self, 'trade_manager', None):
            self.trade_manager.daily_profit_target_usd = daily
        self._save_risk_param_to_config('daily_profit_target_usd', daily)

    def on_quick_direction_changed(self, index: int):
        dir_val = self.quick_direction_combo.currentData() or "BOTH"
        if getattr(self, 'orchestrator', None):
            self.orchestrator.allowed_trade_direction = dir_val
        if getattr(self, 'trade_execution', None):
            self.trade_execution.allowed_trade_direction = dir_val
        if 'trading' not in self.config:
            self.config['trading'] = {}
        self.config['trading']['allowed_trade_direction'] = dir_val

        try:
            from pathlib import Path
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de la sauvegarde de allowed_trade_direction")

        dir_labels = {
            "BOTH": "🔄 Mixte (Achat & Vente autorisés)",
            "BUY_ONLY": "📈 Hausse Seule (BUY uniquement)",
            "SELL_ONLY": "📉 Baisse Seule (SELL uniquement)"
        }
        self.append_to_chat(f"⚙️ Sens de trading configuré : <b>{dir_labels.get(dir_val, dir_val)}</b>")

    def on_quick_maxpos_changed(self, val: int):
        """Met à jour le nombre maximum de positions ouvertes simultanément."""
        max_pos = max(1, int(val))
        # Mise à jour du config en mémoire
        if 'risk_management' not in self.config:
            self.config['risk_management'] = {}
        self.config['risk_management']['max_open_positions'] = max_pos
        self.config['risk_management']['manual_max_positions_override'] = True
        # Propagation au risk manager si disponible
        if getattr(self, 'orchestrator', None) and hasattr(self.orchestrator, 'risk'):
            try:
                self.orchestrator.risk.max_open_positions = max_pos
            except Exception:
                pass
        # Propagation au trade manager
        if getattr(self, 'trade_manager', None):
            try:
                self.trade_manager.max_open_positions = max_pos
            except Exception:
                pass
        # Sauvegarde dans config.yaml
        try:
            from pathlib import Path
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de la sauvegarde de max_open_positions")

        tier_notice = ""
        if getattr(self, 'orchestrator', None) and hasattr(self.orchestrator, 'risk'):
            try:
                acc = self.orchestrator.mt5.get_account_info() if hasattr(self.orchestrator.mt5, 'get_account_info') else {}
                bal = float((acc or {}).get('balance', 0.0) or 0.0)
                if bal > 0 and hasattr(self.orchestrator.risk, 'get_account_tier'):
                    tier = self.orchestrator.risk.get_account_tier(bal)
                    rec = tier.get('max_open_positions', 1)
                    if max_pos > rec:
                        tier_notice = f" <i>(Plafond du palier {tier['name']} de {rec} position{'s' if rec > 1 else ''} outrepassé)</i>"
            except Exception:
                pass

        self.append_to_chat(
            f"📊 <b>Max positions :</b> Le bot peut maintenant ouvrir jusqu'à "
            f"<b>{max_pos}</b> position{'s' if max_pos > 1 else ''} simultanée{'s' if max_pos > 1 else ''}.{tier_notice}"
        )
        self.update_tier_badge()

    def on_quick_mode_changed(self, index: int):
        """Bascule le mode d'exécution : Scanner Manuel (assist) ou Autopilot Complet (auto)."""
        mode_val = self.quick_mode_combo.currentData() or "assist"
        # Mise à jour du config en mémoire
        if 'trading' not in self.config:
            self.config['trading'] = {}
        self.config['trading']['automation_mode'] = mode_val
        # Sauvegarde dans config.yaml
        try:
            from pathlib import Path
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de la sauvegarde de automation_mode")

        if mode_val == "auto":
            self.append_to_chat(
                "🤖 <b>Mode Autopilot Complet activé.</b><br>"
                "L'IA analyse ET exécute les trades automatiquement.<br>"
                "💡 Activez aussi le bouton <b>AUTOPILOT</b> pour les scans cycliques automatiques."
            )
            # Active automatiquement le bouton Autopilot si pas déjà ON
            if hasattr(self, 'autopilot_btn') and not self.autopilot_btn.isChecked():
                self.autopilot_btn.setChecked(True)
        else:
            self.append_to_chat(
                "🔍 <b>Mode Scanner Manuel activé.</b><br>"
                "L'IA propose les opportunités — vous choisissez [PRENDRE] ou [IGNORER].<br>"
                "Lancez le scanner avec le bouton <b>🔍 LANCER LE SCANNER</b>."
            )
            # Désactive le bouton Autopilot si actif
            if hasattr(self, 'autopilot_btn') and self.autopilot_btn.isChecked():
                self.autopilot_btn.setChecked(False)

    def open_settings(self):
        dialog = ConfigDialog(self.config, self)
        if dialog.exec():
            # Synchroniser la barre rapide et le trade manager avec les nouvelles valeurs
            rm = self.config.get('risk_management', {})
            tp_val = float(rm.get('take_profit_target_per_trade_usd', 4.0))
            gb_val = float(rm.get('giveback_usd_small', 0.6))
            daily_val = float(rm.get('daily_profit_target_usd', 50.0))
            dir_val = self.config.get('trading', {}).get('allowed_trade_direction', 'BOTH')

            if hasattr(self, 'quick_tp_spin'):
                self.quick_tp_spin.blockSignals(True)
                self.quick_tp_spin.setValue(tp_val)
                self.quick_tp_spin.blockSignals(False)

            if hasattr(self, 'quick_giveback_spin'):
                self.quick_giveback_spin.blockSignals(True)
                self.quick_giveback_spin.setValue(gb_val)
                self.quick_giveback_spin.blockSignals(False)

            if hasattr(self, 'quick_daily_spin'):
                self.quick_daily_spin.blockSignals(True)
                self.quick_daily_spin.setValue(daily_val)
                self.quick_daily_spin.blockSignals(False)

            if hasattr(self, 'quick_direction_combo'):
                self.quick_direction_combo.blockSignals(True)
                idx = self.quick_direction_combo.findData(dir_val)
                if idx >= 0:
                    self.quick_direction_combo.setCurrentIndex(idx)
                self.quick_direction_combo.blockSignals(False)

            if getattr(self, 'trade_manager', None):
                self.trade_manager.take_profit_target_per_trade_usd = tp_val
                self.trade_manager.giveback_usd_small = gb_val
                self.trade_manager.daily_profit_target_usd = daily_val
                self.trade_manager.daily_profit_target_percent = float(rm.get('daily_profit_target_percent', 3.0))
                self.trade_manager.auto_tier_enabled = rm.get('auto_tier_enabled', True)

            if getattr(self, 'orchestrator', None):
                self.orchestrator.allowed_trade_direction = dir_val
                if getattr(self.orchestrator, 'risk', None):
                    self.orchestrator.risk.spread_buffer_enabled = rm.get('spread_buffer_enabled', True)
                    self.orchestrator.risk.max_spread_sl_ratio = float(rm.get('max_spread_sl_ratio', 0.25))
                    self.orchestrator.risk.auto_tier_enabled = rm.get('auto_tier_enabled', True)
                    self.orchestrator.risk.max_micro_risk_cap_percent = float(rm.get('max_micro_risk_cap_percent', 4.0))
                    self.orchestrator.risk.default_risk_percent = float(rm.get('default_risk_percent', 1.0))
                    self.orchestrator.risk.daily_profit_target_percent = float(rm.get('daily_profit_target_percent', 3.0))
                    self.orchestrator.risk.fixed_lot_size = float(rm.get('fixed_lot_size', 0.0))
                    self.orchestrator.risk.risk_profile = rm.get('risk_profile', 'BALANCED')
                    self.orchestrator.risk.apply_risk_profile(self.orchestrator.risk.risk_profile)
            if getattr(self, 'trade_execution', None):
                self.trade_execution.allowed_trade_direction = dir_val
                self.trade_execution.fixed_lot_size = float(rm.get('fixed_lot_size', 0.0))
            self._refresh_quick_lot_button()
            self.update_tier_badge()
            
    def handle_chat_response(self, result):
        if isinstance(result, dict):
            reply = str(result.get("reply", "")).strip()
            changes = result.get("changes", [])
        else:
            reply = str(result or "").strip()
            changes = []

        self.append_to_chat(reply, sender="IA")
        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.chat_input.setFocus()

        # Lecture vocale si activée
        if hasattr(self, 'voice_manager') and self.voice_manager and not self.voice_manager.is_muted and reply:
            self.voice_manager.speak(reply)

        # Si l'IA a proposé des ajustements concrets de configuration
        if changes:
            try:
                from src.ui.config_proposal_dialog import ConfigProposalDialog
                dlg = ConfigProposalDialog(
                    reply_text=reply,
                    changes=changes,
                    voice_manager=self.voice_manager,
                    parent=self
                )
                if dlg.exec():
                    if getattr(dlg, 'applied', False):
                        self.apply_config_changes(changes)
            except Exception:
                import logging as _lg
                _lg.getLogger(__name__).exception("Erreur lors de l'affichage de ConfigProposalDialog")

    def on_toggle_microphone(self):
        """Démarre ou arrête l'enregistrement micro pour la dictée vocale."""
        if not hasattr(self, 'voice_manager') or not self.voice_manager:
            self.append_to_chat("⚠️ <i>Gestionnaire vocal non initialisé sur cette machine.</i>")
            return

        if not self.voice_manager.is_recording:
            import time as _t
            self._recording_start_time = _t.time()
            started = self.voice_manager.start_recording()
            if started:
                self.mic_btn.setText("⏹️ Terminer (Écoute...)")
                self.mic_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #b71c1c;
                        color: #ffffff;
                        font-weight: bold;
                        border: 1px solid #ef5350;
                        border-radius: 4px;
                        padding: 5px 10px;
                    }
                    QPushButton:hover {
                        background-color: #c62828;
                    }
                """)
                self.append_to_chat("🎙️ <i>Écoute micro en cours... Parlez maintenant, puis recliquez sur 'Terminer'.</i>")
            else:
                self.append_to_chat("⚠️ <i>Impossible d'accéder au microphone. Vérifiez vos périphériques Windows.</i>")
        else:
            import time as _t
            elapsed = _t.time() - getattr(self, '_recording_start_time', 0.0)
            if elapsed < 0.4:
                _t.sleep(0.25)
            self.mic_btn.setText("⏳ Transcription...")
            self.mic_btn.setEnabled(False)
            audio_path = self.voice_manager.stop_recording()

            if audio_path:
                from src.core.voice_manager import TranscriptionThread
                self._transcription_thread = TranscriptionThread(self.voice_manager, audio_path, parent=self)
                self._transcription_thread.transcription_done.connect(self._on_transcription_success)
                self._transcription_thread.transcription_error.connect(self._on_transcription_error)
                self._transcription_thread.start()
            else:
                self._reset_mic_button()

    def _on_transcription_success(self, text: str):
        self._reset_mic_button()
        text = str(text or "").strip()
        if text:
            self.append_to_chat(f"🎙️ <i>Dictée vocale : « {text} »</i>")
            self.chat_input.setText(text)
            self.on_send_chat()
        else:
            self.append_to_chat("⚠️ <i>Aucune parole détectée. Conseil : parlez distinctement au micro puis attendez 1 seconde avant de cliquer sur 'Terminer'.</i>")

    def _on_transcription_error(self, err: str):
        self._reset_mic_button()
        self.append_to_chat(f"⚠️ <i>Erreur transcription micro : {err}</i>")

    def _reset_mic_button(self):
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("🎙️ Parler")
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2e39;
                color: #e0e0e0;
                border: 1px solid #363c4e;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #363c4e;
                color: #ffffff;
            }
        """)

    def on_toggle_voice_output(self):
        """Active ou coupe la synthèse vocale."""
        if not hasattr(self, 'voice_manager') or not self.voice_manager:
            return
        is_muted = self.voice_manager.toggle_mute()
        if is_muted:
            self.voice_btn.setText("🔇 Voix OFF")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #374151;
                    color: #9ca3af;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 5px 10px;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
            """)
        else:
            self.voice_btn.setText("🔊 Voix ON")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e3a8a;
                    color: #60a5fa;
                    border: 1px solid #2563eb;
                    border-radius: 4px;
                    padding: 5px 10px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                    color: white;
                }
            """)
            self.voice_manager.speak("Synthèse vocale activée.")

    def on_request_optimization(self):
        """Déclenche une demande d'optimisation intelligente à l'assistant."""
        self.chat_input.setText("Analyse mon capital et mes marchés, et propose les meilleurs réglages (incluant pending orders, lot, risk).")
        self.on_send_chat()

    def open_voice_settings(self):
        """Ouvre la boîte de dialogue des paramètres vocaux et de l'Auto-Adaptation IA."""
        try:
            from src.ui.voice_settings_dialog import VoiceSettingsDialog
            dlg = VoiceSettingsDialog(
                self.voice_manager,
                self.config,
                on_save_callback=self.on_save_voice_settings,
                parent=self
            )
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Erreur Réglages Vocaux", f"Impossible d'ouvrir les réglages vocaux : {e}")

    def on_save_voice_settings(self, settings: dict):
        """Sauvegarde les réglages vocaux dans config.yaml et synchronise VoiceManager."""
        self.config["voice"] = settings
        try:
            from pathlib import Path
            import yaml
            cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
            self.append_to_chat("⚙️ <i>Paramètres vocaux et intelligence adaptative enregistrés avec succès.</i>")
        except Exception:
            import logging as _lg
            _lg.getLogger(__name__).exception("Erreur lors de la sauvegarde de config.yaml (voice)")

    def _get_advisor_context(self) -> dict:
        """Rassemble le contexte réel (compte MT5, positions, config) pour le conseiller IA."""
        account_view = {}
        open_pos_count = 0
        try:
            if hasattr(self.orchestrator, 'mt5') and self.orchestrator.mt5.connected:
                acc = self.orchestrator.mt5.get_account_info() or {}
                account_view = {k: acc.get(k) for k in ("balance", "equity", "margin_free", "profit") if k in acc}
                positions = self.orchestrator.mt5.get_open_positions() or []
                open_pos_count = len(positions)
        except Exception:
            pass

        return {
            "config": self.config,
            "account": account_view,
            "open_positions_count": open_pos_count,
            "autopilot_enabled": bool(getattr(self, 'is_autopilot_scan', False)),
        }

    def apply_config_changes(self, changes: list):
        """Applique les modifications validées par l'utilisateur à chaud et sur disque."""
        if not changes:
            return

        import yaml
        from pathlib import Path
        import logging as _lg
        log = _lg.getLogger(__name__)

        try:
            for ch in changes:
                section = ch.get("section")
                key = ch.get("key")
                value = ch.get("value")
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = value

            # 1. Sauvegarder sur disque config/config.yaml
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)

            # 2. Synchroniser les moteurs en temps réel (runtime sync)
            rm = self.config.get("risk_management", {})
            tr = self.config.get("trading", {})

            # Risk Manager
            risk = getattr(self.orchestrator, "risk", None)
            if risk:
                if "default_risk_percent" in rm:
                    risk.default_risk_percent = float(rm["default_risk_percent"])
                if "max_daily_loss_percent" in rm:
                    risk.max_daily_loss_percent = float(rm["max_daily_loss_percent"])
                if "max_lot_size" in rm:
                    risk.global_max_lot = float(rm["max_lot_size"])
                if "min_confidence" in rm:
                    risk.min_confidence = float(rm["min_confidence"])
                if "min_reward_risk" in rm:
                    risk.min_reward_risk = float(rm["min_reward_risk"])
                if "risk_profile" in rm:
                    risk.risk_profile = rm["risk_profile"]
                    risk.apply_risk_profile(risk.risk_profile)
                if "fixed_lot_size" in rm:
                    risk.fixed_lot_size = float(rm["fixed_lot_size"])

            # Trade Execution
            if getattr(self, "trade_execution", None):
                if "min_confidence" in rm:
                    self.trade_execution.min_confidence = float(rm["min_confidence"])
                if "min_reward_risk" in rm:
                    self.trade_execution.min_reward_risk = float(rm["min_reward_risk"])
                if "daily_profit_target_usd" in rm:
                    self.trade_execution.daily_profit_target_usd = float(rm["daily_profit_target_usd"])
                if "allowed_trade_direction" in tr:
                    self.trade_execution.allowed_trade_direction = tr["allowed_trade_direction"]
                if "automation_mode" in tr:
                    self.trade_execution.mode = tr["automation_mode"].lower()
                if "fixed_lot_size" in rm:
                    self.trade_execution.fixed_lot_size = float(rm["fixed_lot_size"])

            # Orchestrator
            if getattr(self, "orchestrator", None):
                if "allowed_trade_direction" in tr:
                    self.orchestrator.allowed_trade_direction = tr["allowed_trade_direction"]

            # Trade Manager
            if getattr(self, "trade_manager", None):
                if "daily_profit_target_usd" in rm:
                    self.trade_manager.daily_profit_target_usd = float(rm["daily_profit_target_usd"])
                if "take_profit_target_per_trade_usd" in rm:
                    self.trade_manager.take_profit_target_per_trade_usd = float(rm["take_profit_target_per_trade_usd"])
                if "giveback_usd_small" in rm:
                    self.trade_manager.giveback_usd_small = float(rm["giveback_usd_small"])

            # 3. Synchroniser les widgets rapides de l'interface
            if hasattr(self, "quick_direction_combo") and "allowed_trade_direction" in tr:
                idx = self.quick_direction_combo.findData(tr["allowed_trade_direction"])
                if idx >= 0:
                    self.quick_direction_combo.blockSignals(True)
                    self.quick_direction_combo.setCurrentIndex(idx)
                    self.quick_direction_combo.blockSignals(False)

            if hasattr(self, "quick_daily_spin") and "daily_profit_target_usd" in rm:
                self.quick_daily_spin.blockSignals(True)
                self.quick_daily_spin.setValue(float(rm["daily_profit_target_usd"]))
                self.quick_daily_spin.blockSignals(False)

            if hasattr(self, "_refresh_quick_lot_button"):
                self._refresh_quick_lot_button()
            if hasattr(self, "update_tier_badge"):
                self.update_tier_badge()

            # 4. Audit log
            self._audit_advisor_action("applied", {"changes": changes})

            # 5. Confirmation utilisateur
            self.append_to_chat("✅ <b>[Conseiller IA]</b> Nouveaux réglages appliqués avec succès !")
            if hasattr(self, 'voice_manager') and self.voice_manager and not self.voice_manager.is_muted:
                self.voice_manager.speak("Vos nouveaux réglages ont été appliqués avec succès.")

        except Exception as e:
            log.exception("Erreur lors de l'application des réglages")
            self.append_to_chat(f"❌ <b>Erreur application réglages :</b> {e}")

    def _audit_advisor_action(self, status: str, proposal: dict, error: str = ""):
        from datetime import datetime, timezone
        from pathlib import Path
        import json
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "changes": proposal.get("changes", []),
            "error": error,
        }
        try:
            audit_path = Path(__file__).resolve().parent.parent.parent / "logs" / "config_advisor_audit.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def closeEvent(self, event):
        """Properly stop all threads and timers before closing the window."""
        # Stop autopilot timers
        if getattr(self, 'autopilot_timer', None) is not None:
            self.autopilot_timer.stop()
        if getattr(self, 'tm_timer', None) is not None:
            self.tm_timer.stop()

        # Stop voice speaking and recording
        if hasattr(self, 'voice_manager') and self.voice_manager:
            try:
                self.voice_manager.stop_speaking()
                if self.voice_manager.is_recording:
                    self.voice_manager.stop_recording()
            except Exception:
                pass

        # Stop transcription thread if running
        if getattr(self, '_transcription_thread', None) is not None:
            try:
                if self._transcription_thread.isRunning():
                    self._transcription_thread.wait(2000)
            except Exception:
                pass

        # Signal scanner thread to abort backoff sleeps immediately
        scanner = getattr(self, 'scanner_thread', None)
        if scanner is not None and scanner.isRunning():
            scanner.stop()       # sets stop_event
            scanner.wait(8000)   # give it up to 8s to finish current op

        # Stop all OB rescan threads
        for ob_t in getattr(self, '_ob_threads', []):
            try:
                ob_t.stop()
                ob_t.wait(3000)
            except Exception:
                pass

        # Wait for chat thread to finish
        chat_thread = getattr(self, 'chat_thread', None)
        if chat_thread is not None and chat_thread.isRunning():
            chat_thread.wait(3000)

        # Stop AI analyzer thread
        if getattr(self, 'ai_analyzer_thread', None) is not None:
            try:
                self.ai_analyzer_thread.stop()
                self.ai_analyzer_thread.wait(2000)
            except Exception:
                pass

        # Stop watchdog
        if getattr(self, 'watchdog', None) is not None and hasattr(self.watchdog, 'stop'):
            try:
                self.watchdog.stop()
            except Exception:
                pass

        event.accept()
