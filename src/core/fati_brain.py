# -*- coding: utf-8 -*-
"""
FOCON Autonomous Trading Intelligence (FATI)
============================================
Architecture unifiée des 5 Cerveaux Spécialisés ("Los Cinco Cerebros") :
1. OBSERVER     -> El Observador    (src.cerebros.el_observador)
2. DÉCIDER      -> El Estratega     (src.cerebros.el_estratega)
3. APPRENDRE    -> El Sabio         (src.cerebros.el_sabio)
4. EXPÉRIMENTER -> El Investigador  (src.cerebros.el_investigador)
5. SÉCURISER    -> El Centinela     (src.cerebros.el_centinela)
"""

from typing import Dict, Any, List, Optional
import logging

from src.cerebros.el_observador import ElObservador
from src.cerebros.el_estratega import ElEstratega
from src.cerebros.el_sabio import ElSabio
from src.cerebros.el_investigador import ElInvestigador
from src.cerebros.el_centinela import ElCentinela
from src.cerebros.el_economista import ElEconomista

from src.services.learning_scheduler import LearningScheduler
from src.services.health_monitor import SystemHealthMonitor

logger = logging.getLogger(__name__)


class FATIBrain:
    """
    Système unifié d'Intelligence Autonome de Trading (FATI).
    Orchestre les 6 Cerveaux Espagnols spécialisés ("Los Seis Cerebros") tout en assurant
    une rétro-compatibilité intégrale avec l'ensemble du projet :
    1. OBSERVER        -> El Observador    (src.cerebros.el_observador)
    2. DÉCIDER         -> El Estratega     (src.cerebros.el_estratega)
    3. APPRENDRE       -> El Sabio         (src.cerebros.el_sabio)
    4. EXPÉRIMENTER    -> El Investigador  (src.cerebros.el_investigador)
    5. SÉCURISER       -> El Centinela     (src.cerebros.el_centinela)
    6. MACROÉCONOMIE   -> El Economista    (src.cerebros.el_economista)
    """

    def __init__(
        self,
        mt5_connector=None,
        local_brain=None,
        weights_engine=None,
        critic=None,
        trade_memory=None,
    ):
        # 1. CEREBRO OBSERVADOR : Microstructure, régimes et dynamique des flux
        self.observador = ElObservador(mt5_connector=mt5_connector)
        self.regime_detector = self.observador.regime_detector

        # 2. CEREBRO ESTRATEGA : Décision, probabilités logistiques, SMC, CRT, Boomerang, E(R)
        self.estratega = ElEstratega(
            local_brain=local_brain,
            observador=self.observador,
            weights_engine=weights_engine,
        )
        self.decision_engine = self.estratega.decision_engine

        # 3. CEREBRO SABIO : Mémoire, apprentissage réflexif, pondérations dynamiques et journal
        self.sabio = ElSabio(
            weights_engine=weights_engine,
            critic=critic,
            trade_memory=trade_memory,
        )
        self.journal = self.sabio.journal
        self.experience = self.sabio.experience
        self.knowledge = self.sabio.knowledge
        self.analyzer = self.sabio.analyzer
        self.drift_detector = self.sabio.drift_detector
        self.feedback = self.sabio.feedback

        # Connexion de la sentinelle de dérive (DriftDetector) à l'arbitrage décisionnel
        self.estratega.drift_detector = self.drift_detector
        self.decision_engine.drift_detector = self.drift_detector

        # 4. CEREBRO INVESTIGADOR : Formulation d'hypothèses, A/B testing sandbox, Walk-Forward
        self.investigador = ElInvestigador()
        self.hypotheses = self.investigador.hypotheses
        self.experiments = self.investigador.experiments
        self.walk_forward = self.investigador.walk_forward
        self.registry = self.investigador.registry

        # 5. CEREBRO CENTINELA : Garde-fou indépendant, sas de promotion et circuit-breaker
        self.centinela = ElCentinela(
            registry=self.registry,
            weights_engine=weights_engine,
        )
        self.risk_guard = self.centinela.risk_guard
        self.promotion_gate = self.centinela.promotion_gate
        self.rollback = self.centinela.rollback

        # 6. CEREBRO ECONOMISTA : Intelligence macroéconomique, calendrier, CSI et gel d'annonces
        self.economista = ElEconomista(mt5_connector=mt5_connector)

        # Services d'infrastructure
        self.scheduler = LearningScheduler(interval_seconds=1800)
        self.health = SystemHealthMonitor(
            mt5_connector=mt5_connector,
            journal=self.journal,
            drift_detector=self.drift_detector,
            safety_guard=self.risk_guard,
            registry=self.registry,
        )

        # Enregistrement des tâches périodiques du planificateur
        self.scheduler.register_task("audit_drift", self._scheduled_drift_audit)
        self.scheduler.register_task("discover_toxic_patterns", self._scheduled_pattern_discovery)
        self.scheduler.register_task("research_cycle", self._scheduled_research_cycle)
        self.scheduler.register_task("refresh_economic_data", self._scheduled_economic_refresh)

        logger.info("🧠 [FATI] Los 6 Cerebros Autónomos inicializados con éxito.")
        if hasattr(self, "economista") and self.economista:
            try:
                self.economista.imprimir_tabla_salud_economica()
            except Exception as _ite:
                logger.debug(f"[FATI] Erreur affichage initial santé économique: {_ite}")

    def observe_and_decide(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        boomerang_data: Optional[Dict[str, Any]] = None,
        order_blocks: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        direction_filter: str = "BOTH",
        observation: Optional[Dict[str, Any]] = None,
        ai_advice: Optional[Dict[str, Any]] = None,
        economic_context: Optional[Dict[str, Any]] = None,
        df: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Cycle Pilier 1, 2, 3 & 6 : El Observador, El Estratega, El Sabio & El Economista.
        Intègre l'observation complète (50 bougies, ZigZag/Fractals, OTE, Liquidité, VSA, MTF, Crypto Week-End),
        le contexte macroéconomique, l'affinité historique et journalise la décision via El Sabio.
        """
        # 1. Observation complète automatique via El Observador si df fourni et observation absente
        if observation is None and df is not None and hasattr(self, "observador") and self.observador:
            try:
                observation = self.observador.observar_mercado(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    tech_features=tech_features,
                    spread_price=spread_price,
                    htf_trend=htf_trend,
                    h4_trend=htf_trend,
                )
            except Exception as _oe:
                logger.debug(f"[FATI] Erreur observation automatique pour {symbol}: {_oe}")

        # 2. Consultation automatique d'El Economista si non injecté préalablement
        if economic_context is None and hasattr(self, "economista") and self.economista:
            try:
                economic_context = self.economista.analizar_contexto_economico(symbol)
            except Exception as _ece:
                logger.error(f"[FATI] Erreur consultation El Economista pour {symbol}: {_ece}")
                # FAIL-CLOSED : En cas d'erreur ou d'exception, verrouiller immédiatement l'ouverture d'ordres
                economic_context = {
                    "cerebro_origen": "El Economista",
                    "symbol": symbol,
                    "status": "ERROR",
                    "bias": "NEUTRAL",
                    "confidence": 0.0,
                    "event_risk": "CRITICAL",
                    "high_impact_window": True,
                    "minutes_to_next_event": 0.0,
                    "next_event_title": "EXCEPTION_IN_ECONOMIC_BRAIN",
                    "calendar_confidence": "UNCERTAIN",
                    "currency_strength": {"available": False, "base": 50.0, "quote": 50.0, "differential": 0.0},
                    "interest_rate_differential": 0.0,
                    "is_rate_stale": True,
                    "news_sentiment": "UNAVAILABLE",
                    "reason": f"Erreur critique El Economista : {_ece}. Blocage de sécurité activé (Fail-Closed).",
                    "recommendation": "NO_TRADE_NEWS",
                    "data_freshness": "STALE_OR_ERROR",
                }

        decision = self.estratega.decidir(
            symbol=symbol,
            timeframe=timeframe,
            tech_features=tech_features,
            amd_context=amd_context,
            boomerang_data=boomerang_data,
            order_blocks=order_blocks,
            crt_setup=crt_setup,
            htf_trend=htf_trend,
            spread_price=spread_price,
            direction_filter=direction_filter,
            observation=observation,
            ai_advice=ai_advice,
            economic_context=economic_context,
        )

        # Enregistrement systématique de la décision dans le journal (El Sabio)
        decision_id = self.sabio.registrar_decision(decision, is_executed=(decision.get("action") in ("BUY", "SELL")))
        decision["fati_decision_id"] = decision_id
        return decision

    def learn_from_trade(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        spread_cost: float = 0.0,
        mfe: float = 0.0,
        mae: float = 0.0,
        post_exit_candles: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Cycle Pilier 3 & 5 : El Sabio apprend du trade clôturé.
        Si une dérive critique est détectée, El Centinela active immédiatement le rollback d'urgence.
        """
        report = self.sabio.aprender_de_trade(
            ticket=ticket,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            exit_reason=exit_reason,
            spread_cost=spread_cost,
            mfe=mfe,
            mae=mae,
            post_exit_candles=post_exit_candles,
        )

        drift = report.get("drift", {})
        if drift.get("status") == "CRITICAL_QUARANTINE":
            logger.critical("🚨 [FATI] Dérive critique détectée en production. El Centinela déclenche le rollback...")
            self.centinela.activar_rollback(razon_incidente=drift.get("reason", "Dérive d'espérance critique"))

        return report

    def _scheduled_drift_audit(self):
        """Tâche d'arrière-plan : surveillance de dérive de performance via El Sabio."""
        recent = self.sabio.journal.get_closed_trades(limit=25)
        self.sabio.drift_detector.evaluate_drift(recent)

    def _scheduled_pattern_discovery(self):
        """Tâche d'arrière-plan : détection de patterns toxiques (El Sabio) et hypothèses (El Investigador)."""
        toxic = self.sabio.detectar_patrones_toxicos(min_sample=5, max_win_rate=0.30)
        if toxic:
            hypotheses = self.investigador.generar_hipotesis(toxic)
            logger.info(f"💡 [El Investigador] {len(hypotheses)} hypothèses d'amélioration générées.")

    def _scheduled_research_cycle(self):
        """Tâche d'arrière-plan : cycle complet de recherche, sandbox et validation Walk-Forward."""
        try:
            report = self.investigador.ejecutar_ciclo_investigacion(sabio=self.sabio, centinela=self.centinela)
            logger.info(f"🔬 [El Investigador] Rapport de cycle autonome : {report}")
        except Exception as e:
            logger.error(f"[FATI Research] Erreur cycle recherche planifié : {e}")

    def _scheduled_economic_refresh(self):
        """Tâche d'arrière-plan : rafraîchissement des flux économiques gratuits (El Economista)."""
        try:
            res = self.economista.actualizar_datos_economicos()
            logger.info(f"📰 [El Economista] Flux économiques rafraîchis : {res}")
        except Exception as e:
            logger.debug(f"[FATI Economic] Erreur rafraîchissement économique : {e}")

