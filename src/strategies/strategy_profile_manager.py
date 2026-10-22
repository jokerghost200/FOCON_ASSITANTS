"""
Gestionnaire des Profils de Stratégies des Traders Professionnels (FOCON Strategy Copy).
Gère la configuration, le profil actif, les modes (Simulation, Copie Démo, Personnalisé),
l'évaluation des signaux et la persistance des événements dans les journaux JSONL.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal
from src.strategies.corentin_strategy import CorentinStrategy
from src.strategies.benjamin_strategy import BenjaminStrategy
from src.strategies.romain_strategy import RomainStrategy
from src.strategies.nada_strategy import NadaStrategy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "trader_profiles.yaml"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class StrategyProfileManager:
    """Orchestrateur central des 4 stratégies de traders professionnels."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None, log_dir: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.signals_file = self.log_dir / "pro_trader_signals.jsonl"
        self.outcomes_file = self.log_dir / "pro_trader_outcomes.jsonl"

        self.config: Dict[str, Any] = {}
        self.active_profile: Optional[str] = None
        self.mode: str = "simulation"  # "simulation", "copy", "custom"
        self.custom_overrides: Dict[str, Dict[str, Any]] = {}

        # Initialisation des instances de stratégies
        self.strategies: Dict[str, BaseTraderStrategy] = {}
        self.load_config()

        # Statistiques d'exécution
        self.stats: Dict[str, Dict[str, Any]] = {
            "corentin": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "signals_count": 0},
            "benjamin": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "signals_count": 0},
            "romain": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "signals_count": 0},
            "nada": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "signals_count": 0},
        }
        self._load_persisted_records()

    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis config/trader_profiles.yaml."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"[StrategyProfileManager] Erreur lecture config {self.config_path} : {e}")
                self.config = {}
        else:
            self.config = {}

        pro_cfg = self.config.get("pro_trader_profiles", {})
        self.active_profile = pro_cfg.get("active_profile", None)
        self.mode = pro_cfg.get("mode", "simulation")

        # Initialiser ou mettre à jour les stratégies
        corentin_cfg = pro_cfg.get("corentin", {})
        benjamin_cfg = pro_cfg.get("benjamin", {})
        romain_cfg = pro_cfg.get("romain", {})
        nada_cfg = pro_cfg.get("nada", {})

        self.strategies = {
            "corentin": CorentinStrategy(corentin_cfg),
            "benjamin": BenjaminStrategy(benjamin_cfg),
            "romain": RomainStrategy(romain_cfg),
            "nada": NadaStrategy(nada_cfg),
        }
        return self.config

    def save_config(self) -> bool:
        """Sauvegarde l'état actuel dans le fichier config/trader_profiles.yaml."""
        try:
            if "pro_trader_profiles" not in self.config:
                self.config["pro_trader_profiles"] = {}

            self.config["pro_trader_profiles"]["active_profile"] = self.active_profile
            self.config["pro_trader_profiles"]["mode"] = self.mode

            # Si des overrides personnalisés existent, les intégrer
            for p_id, overrides in self.custom_overrides.items():
                if p_id in self.config["pro_trader_profiles"]:
                    self.config["pro_trader_profiles"][p_id].update(overrides)

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"[StrategyProfileManager] Configuration sauvegardée dans {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"[StrategyProfileManager] Erreur sauvegarde config : {e}")
            return False

    def is_active(self) -> bool:
        """Indique si un profil de trader pro est actuellement activé."""
        return self.active_profile in self.strategies

    def get_active_profile_id(self) -> Optional[str]:
        return self.active_profile

    def get_active_strategy(self) -> Optional[BaseTraderStrategy]:
        if self.is_active():
            return self.strategies.get(self.active_profile)
        return None

    def get_active_mode(self) -> str:
        return self.mode

    def set_active_profile(
        self, profile_id: Optional[str], mode: str = "simulation", custom_cfg: Optional[Dict[str, Any]] = None
    ):
        """Active ou désactive un profil de trader professionnel."""
        valid_profiles = list(self.strategies.keys())
        if profile_id is not None and profile_id not in valid_profiles:
            raise ValueError(f"Profil invalide: '{profile_id}'. Profils disponibles: {valid_profiles}")

        self.active_profile = profile_id
        if mode in ("simulation", "copy", "custom"):
            self.mode = mode
        else:
            self.mode = "simulation"

        if custom_cfg and profile_id:
            self.custom_overrides[profile_id] = custom_cfg
            if profile_id in self.strategies:
                self.strategies[profile_id].config.update(custom_cfg)

        self.save_config()
        status_msg = f"{self.active_profile.upper()} en mode {self.mode.upper()}" if self.active_profile else "DÉSACTIVÉ"
        logger.info(f"👑 [StrategyProfileManager] Statut Copie Pro Trader : {status_msg}")

    def get_all_profiles_info(self) -> List[Dict[str, Any]]:
        """Retourne la fiche détaillée de chaque profil pour l'affichage dans l'UI ou Web."""
        profiles_info = []
        pro_cfg = self.config.get("pro_trader_profiles", {})

        for pid, strat in self.strategies.items():
            p_data = pro_cfg.get(pid, {})
            p_stats = self.stats.get(pid, {})
            trades = p_stats.get("trades", 0)
            wins = p_stats.get("wins", 0)
            win_rate = round((wins / trades * 100.0), 1) if trades > 0 else 0.0

            profiles_info.append({
                "id": pid,
                "name": strat.name,
                "style": p_data.get("style", ""),
                "description": p_data.get("description", ""),
                "symbols": p_data.get("symbols", []),
                "risk_pct": p_data.get("risk_pct", 1.0),
                "is_active": (self.active_profile == pid),
                "active_mode": self.mode if self.active_profile == pid else None,
                "stats": {
                    "trades": trades,
                    "wins": wins,
                    "losses": p_stats.get("losses", 0),
                    "win_rate_pct": win_rate,
                    "total_pnl": round(p_stats.get("total_pnl", 0.0), 2),
                    "signals_count": p_stats.get("signals_count", 0),
                },
                "config": dict(strat.config),
            })
        return profiles_info

    def evaluate_symbol(
        self,
        symbol: str,
        df_primary: Any,
        df_htf: Optional[Any] = None,
        df_ltf: Optional[Any] = None,
        spread: float = 0.0,
        min_distance: float = 0.0,
        tech_features: Optional[Dict[str, Any]] = None,
        amd_context: Optional[Dict[str, Any]] = None,
        order_flow_confluence: Optional[Any] = None,
        current_dt: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Optional[ProSignal]:
        """
        Évalue le symbole selon le profil pro actuellement actif.
        Si aucun profil n'est actif, retourne None.
        """
        if not self.is_active():
            return None

        strat = self.get_active_strategy()
        if not strat or not strat.is_symbol_supported(symbol):
            return None

        signal = strat.evaluate(
            symbol=symbol,
            df_primary=df_primary,
            df_htf=df_htf,
            df_ltf=df_ltf,
            spread=spread,
            min_distance=min_distance,
            tech_features=tech_features,
            amd_context=amd_context,
            order_flow_confluence=order_flow_confluence,
            current_dt=current_dt,
            **kwargs,
        )

        if signal:
            self.record_signal(signal, mode=self.mode, symbol=symbol)

        return signal

    def evaluate_all(
        self,
        symbol: str,
        df_primary: Any,
        df_htf: Optional[Any] = None,
        df_ltf: Optional[Any] = None,
        spread: float = 0.0,
        min_distance: float = 0.0,
        tech_features: Optional[Dict[str, Any]] = None,
        amd_context: Optional[Dict[str, Any]] = None,
        order_flow_confluence: Optional[Any] = None,
        current_dt: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[ProSignal]]:
        """
        Évalue simultanément les 4 stratégies de traders sur un symbole
        (idéal pour le mode comparatif / scanner pro).
        """
        results = {}
        for pid, strat in self.strategies.items():
            if strat.is_symbol_supported(symbol):
                res = strat.evaluate(
                    symbol=symbol,
                    df_primary=df_primary,
                    df_htf=df_htf,
                    df_ltf=df_ltf,
                    spread=spread,
                    min_distance=min_distance,
                    tech_features=tech_features,
                    amd_context=amd_context,
                    order_flow_confluence=order_flow_confluence,
                    current_dt=current_dt,
                    **kwargs,
                )
                results[pid] = res
            else:
                results[pid] = None
        return results

    def record_signal(self, signal: ProSignal, mode: str, symbol: str):
        """Persiste le signal émis dans le journal JSONL."""
        now_iso = datetime.now().isoformat()
        sig_data = {
            "timestamp": now_iso,
            "trader": signal.trader,
            "profile_id": signal.profile_id,
            "mode": mode,
            "symbol": symbol,
            "direction": signal.direction,
            "entry": signal.entry,
            "sl": signal.sl,
            "tp": signal.tp,
            "rr": signal.rr,
            "is_pending": signal.is_pending,
            "order_type": signal.order_type,
            "reason": signal.reason,
            "confluences": signal.confluences,
            "decay_risk_mult": signal.decay_risk_mult,
            "cluster_orders_count": len(signal.cluster_orders) if signal.cluster_orders else 0,
        }

        # Mise à jour des compteurs en mémoire
        if signal.profile_id in self.stats:
            self.stats[signal.profile_id]["signals_count"] += 1

        try:
            with open(self.signals_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sig_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[StrategyProfileManager] Erreur écriture signal JSONL : {e}")

    def record_trade_outcome(
        self,
        ticket: int,
        profile_id: str,
        symbol: str,
        profit: float,
        is_win: bool,
        notes: str = "",
    ):
        """Persiste le résultat d'un trade exécuté par un profil pro dans le journal JSONL."""
        now_iso = datetime.now().isoformat()
        outcome_data = {
            "timestamp": now_iso,
            "ticket": int(ticket),
            "profile_id": str(profile_id).lower(),
            "symbol": str(symbol),
            "profit": float(profit),
            "is_win": bool(is_win),
            "notes": str(notes),
        }

        # Mise à jour des compteurs en mémoire
        pid = str(profile_id).lower()
        if pid in self.stats:
            self.stats[pid]["trades"] += 1
            if is_win:
                self.stats[pid]["wins"] += 1
            else:
                self.stats[pid]["losses"] += 1
            self.stats[pid]["total_pnl"] += float(profit)

        # Si Benjamin Mauger est le profil, mettre à jour son compteur de trades journaliers
        if pid == "benjamin" and "benjamin" in self.strategies:
            b_strat = self.strategies["benjamin"]
            if hasattr(b_strat, "set_executed_trades_count"):
                b_strat.set_executed_trades_count(self.stats[pid]["trades"])

        try:
            with open(self.outcomes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[StrategyProfileManager] Erreur écriture issue JSONL : {e}")

    def _load_persisted_records(self):
        """Recharge les historiques de signaux et d'issues pour la persistance cross-redémarrage."""
        # 1. Signaux
        if self.signals_file.exists():
            try:
                with open(self.signals_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        pid = str(record.get("profile_id", "")).lower()
                        if pid in self.stats:
                            self.stats[pid]["signals_count"] += 1
            except Exception as e:
                logger.debug(f"[StrategyProfileManager] Note lecture signaux passés : {e}")

        # 2. Issues / Résultats de trades
        if self.outcomes_file.exists():
            try:
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        pid = str(record.get("profile_id", "")).lower()
                        if pid in self.stats:
                            self.stats[pid]["trades"] += 1
                            if record.get("is_win", False):
                                self.stats[pid]["wins"] += 1
                            else:
                                self.stats[pid]["losses"] += 1
                            self.stats[pid]["total_pnl"] += float(record.get("profit", 0.0))

                # Ajuster le compteur de trades de Benjamin Mauger
                if "benjamin" in self.strategies:
                    self.strategies["benjamin"].set_executed_trades_count(self.stats["benjamin"]["trades"])
            except Exception as e:
                logger.debug(f"[StrategyProfileManager] Note lecture issues passées : {e}")
