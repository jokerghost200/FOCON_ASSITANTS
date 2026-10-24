import logging
import json
import time
from src.core.mt5_connector import MT5Connector
from src.core.market_data import MarketDataEngine
from src.core.technical_indicators import TechnicalIndicators
from src.core.boomerang_detector import BoomerangDetector
from src.core.order_block_detector import OrderBlockDetector
from src.core.market_context import MarketContextEngine
from src.core.crt_detector import CRTDetector
from src.core.order_flow_engine import OrderFlowEngine, OrderFlowConfluence
from src.core.order_flow_journal import OrderFlowJournal
from src.trading.strategy_engine import StrategySelector
from src.ai.local_quant_brain import LocalQuantBrain
from src.ai.ai_router import AIRouter
from src.trading.risk_manager import RiskManager
from src.strategies.strategy_profile_manager import StrategyProfileManager

logger = logging.getLogger(__name__)


class AIOrchestrator:
    def __init__(
        self,
        mt5_connector: MT5Connector,
        market_data: MarketDataEngine,
        tech_indicators: TechnicalIndicators,
        ai_router: AIRouter,
        risk_manager: RiskManager,
        live_news_enabled: bool = True,
        order_block_scan_enabled: bool = True,
        allowed_trade_direction: str = "BOTH",
        trade_memory=None,
        weights_engine=None,
        config: dict = None,
    ):
        self.mt5 = mt5_connector
        self.market_data = market_data
        self.tech = tech_indicators
        self.config = config or {}
        self.router = ai_router
        self.live_news_enabled = live_news_enabled
        self.order_block_scan_enabled = order_block_scan_enabled
        self.allowed_trade_direction = str(allowed_trade_direction or "BOTH").upper()
        self.trade_memory = trade_memory
        self.weights_engine = weights_engine
        # Compat : l'ancien attribut `self.ai` / `self.news` reste utilisable.
        self.ai = ai_router
        self.news = ai_router
        self.risk = risk_manager
        # Cerveau quantitatif local de secours (100% hors-ligne) avec poids adaptatifs
        self.local_brain = LocalQuantBrain(min_confidence=60.0, min_rr=1.5, weights_engine=self.weights_engine)
        # Boomerang setup detector
        self.boomerang = BoomerangDetector(
            impulse_atr_mult=2.0,
            breakout_confirm=1,
            retest_tolerance=0.5,
            rejection_wick_min=0.3,
            min_rr=1.5,
        )
        self.ob_detector = OrderBlockDetector(
            impulse_atr_mult=1.8,
            lookback=300,
            retest_tolerance=0.6,
            sl_buffer_mult=0.2,
            tp_rr_mult=2.5,
        )
        self.strategy_selector = StrategySelector()
        self.crt_detector = CRTDetector(
            min_sweep_atr_mult=0.10,
            min_range_atr_mult=0.50,
            min_rr_tp1=1.5,
        )
        # Moteur Order Flow & Microstructure (DOM L2, Heatmap, CVD, Volume Profile, Absorption)
        of_cfg = self.config.get('order_flow', {}) if isinstance(self.config, dict) else {}
        self.order_flow_engine = OrderFlowEngine(
            vp_bins=int(of_cfg.get('volume_profile_bins', 50)),
            imbalance_ratio=float(of_cfg.get('imbalance_ratio', 3.0)),
            dom_wall_multiplier=float(of_cfg.get('dom_wall_multiplier', 2.0)),
            min_confluence_score=float(of_cfg.get('min_confluence_score', 60.0)),
        )
        self.order_flow_journal = OrderFlowJournal()
        self.strategy_profile_manager = StrategyProfileManager()
        self._symbol_cooldowns = {}

    def set_symbol_cooldown(self, symbol: str, duration_seconds: float = 1800.0, reason: str = "", direction: str = None):
        """
        Active un cooldown sur un symbole après une sortie de trade (perte ou clôture)
        pour éviter le revenge trading et l'ouverture répétée de positions en sens contraire du marché.
        """
        import time
        sym = str(symbol).upper()
        now = time.time()
        self._symbol_cooldowns[sym] = {
            "until": now + float(duration_seconds),
            "reason": str(reason),
            "direction": str(direction).upper() if direction else None,
            "set_at": now,
        }
        logger.info(f"⏳ [Cooldown] Symbole {sym} placé en cooldown pour {duration_seconds/60:.1f} min ({reason}).")

    def is_symbol_in_cooldown(self, symbol: str, direction: str = None) -> tuple:
        """
        Vérifie si le symbole est sous cooldown.
        Retourne (is_in_cooldown, reason).
        """
        import time
        sym = str(symbol).upper()
        cd = self._symbol_cooldowns.get(sym)
        if not cd:
            return False, ""
        now = time.time()
        if now < cd.get("until", 0.0):
            cd_dir = cd.get("direction")
            if cd_dir is None or direction is None or cd_dir == str(direction).upper():
                rem_m = (cd["until"] - now) / 60.0
                return True, f"{cd.get('reason', 'Cooldown actif')} (encore {rem_m:.1f} min)"
        else:
            self._symbol_cooldowns.pop(sym, None)
        return False, ""

    def scan_markets(self, symbols: list, timeframe: str, progress_cb=None, stop_event=None, direction_filter: str = None, heartbeat_cb=None, ai_queue=None) -> list:
        """
        Scans a list of markets, filters closed ones, pre-analyzes technicals,
        asks AI to pick the top candidates, and generates full signals for them.
        """
        if direction_filter is None:
            direction_filter = getattr(self, 'allowed_trade_direction', 'BOTH')
        dir_filter_upper = str(direction_filter or 'BOTH').upper()

        def _emit(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        _emit(f"Scanning {len(symbols)} markets on {timeframe} [Direction: {dir_filter_upper}]")
        
        # 1. Filter closed markets
        open_markets = []
        closed_markets = []
        for sym in symbols:
            if heartbeat_cb:
                heartbeat_cb()
            if self.mt5.is_market_open(sym):
                open_markets.append(sym)
            else:
                closed_markets.append(sym)
                
        _emit(f"Open: {len(open_markets)} marchés | Fermés: {len(closed_markets)}")
        
        if not open_markets:
            return [{"error": f"Aucun marché n'est ouvert actuellement. Marchés fermés: {', '.join(closed_markets)}"}]
            
        # 2. Pre-analyze technicals — yield GIL between each market via sleep(0)
        import time
        _emit(f"Calcul des indicateurs pour {len(open_markets)} marchés...")
        tech_summaries = {}
        for i, sym in enumerate(open_markets):
            if heartbeat_cb:
                heartbeat_cb()
            df = self.market_data.get_dataframe(sym, timeframe)
            if df is not None and not df.empty:
                df = self.tech.add_all_indicators(df)
                features = self.tech.extract_latest_features(df)
                amd = self.market_data.analyze_amd(df)
                features['amd_phase'] = amd.get('Phase', 'Unknown')
                features['fvg'] = amd.get('FVG', 'None')
                features['bos_mss'] = amd.get('BOS_MSS', 'None')
                features['liquidity_grab'] = amd.get('Liquidity_Grab', 'None')
                tech_summaries[sym] = features
            # Yield GIL explicitly so Qt main thread can process events and avoid freezing
            time.sleep(0.02)
            if (i + 1) % 5 == 0:
                _emit(f"Indicateurs: {i+1}/{len(open_markets)} marchés traités...")
                
        if not tech_summaries:
            return [{"error": "Impossible de récupérer les données techniques pour les marchés ouverts."}]

        _emit(f"Données techniques: {len(tech_summaries)} marchés OK.")

        # 2b. Pre-filter :
        #  - Calcule le SPREAD en % de l'ATR (rapport coût/volatilité)
        #    Un spread > MAX_SPREAD_ATR_RATIO signifie que le coût d'entrée
        #    annule l'essentiel de l'ATR disponible → trade perdant dès l'ouverture.
        #  - Trie par ATR/Price (volatilité RELATIVE) pour ne pas favoriser les
        #    paires exotiques à grande valeur nominale (USDRUB, USDIDR…).
        MAX_SPREAD_ATR_RATIO = 0.20  # le spread ne doit pas dépasser 20 % de l'ATR
        MIN_VOLATILITY_ATR_REL = 0.0006 # Écarte les marchés morts (< 0.06% de volatilité relative)
        MAX_MARKETS_FOR_AI = 15

        filtered_summaries = {}
        spread_rejected = []
        low_vol_rejected = []
        for sym, f in tech_summaries.items():
            atr = float(f.get('atr_14') or 0)
            close_price = float(f.get('close') or 1)
            atr_rel = (atr / close_price) if close_price > 0 else 0
            f['_atr_rel'] = atr_rel

            # Récupère le spread en points depuis MT5 (plus précis que l'ATR)
            sinfo = self.mt5.get_symbol_info(sym)
            spread_pts  = float(sinfo.get('spread', 0)) if sinfo else 0
            point       = float(sinfo.get('point', 0.00001)) if sinfo else 0.00001
            spread_price = spread_pts * point   # spread en unité de prix
            spread_ratio = (spread_price / atr) if atr > 0 else 1.0
            f['_spread_price'] = spread_price
            f['_spread_ratio'] = spread_ratio

            if spread_ratio > MAX_SPREAD_ATR_RATIO:
                spread_rejected.append(
                    f"{sym} (spread={spread_price:.5f}, ATR={atr:.5f}, ratio={spread_ratio:.1%})"
                )
            elif atr_rel < MIN_VOLATILITY_ATR_REL:
                low_vol_rejected.append(f"{sym} ({atr_rel*100:.3f}%)")
            else:
                filtered_summaries[sym] = f

        if spread_rejected:
            _emit(f"Spread trop élevé (> {MAX_SPREAD_ATR_RATIO:.0%} ATR) → exclus: {', '.join(spread_rejected)}")
        if low_vol_rejected:
            _emit(f"Volatilité trop faible (marchés plats) → exclus: {', '.join(low_vol_rejected)}")

        if not filtered_summaries:
            _emit("⚠️ Tous les marchés ont été filtrés. Passage avec les marchés disponibles.")
            filtered_summaries = tech_summaries   # fallback sans filtre

        # Tri strict par volatilité relative (les actifs qui bougent le plus en premier : Gold, JPY, Indices...)
        sorted_by_atr_rel = sorted(
            filtered_summaries.items(),
            key=lambda x: float(x[1].get('_atr_rel') or 0),
            reverse=True
        )
        tech_summaries = dict(sorted_by_atr_rel[:MAX_MARKETS_FOR_AI])
        _emit(f"Marchés triés par forte volatilité : {', '.join(tech_summaries.keys())}")

            
        # 3. Ask AI to select top candidates (compact summary avec confluences SMC)
        compact_summary = {
            sym: {
                "trend": f.get("trend"),
                "rsi_14": f.get("rsi_14"),
                "macd_hist": f.get("macd_hist"),
                "atr_14": f.get("atr_14"),
                "volatility_pct": f"{float(f.get('_atr_rel') or 0)*100:.2f}%",
                "amd_phase": f.get("amd_phase"),
                "fvg": f.get("fvg"),
                "bos_mss": f.get("bos_mss"),
                "liquidity_grab": f.get("liquidity_grab"),
            }
            for sym, f in tech_summaries.items()
        }

        if dir_filter_upper == "BUY_ONLY":
            dir_clause = "\nCONTRAINTE STRICTE: L'utilisateur exige de trader UNIQUEMENT À LA HAUSSE (BUY_ONLY). Ne sélectionne que des configurations d'ACHAT optimales (supports, pullbacks haussiers, FVG bull, OB bull) !"
        elif dir_filter_upper == "SELL_ONLY":
            dir_clause = "\nCONTRAINTE STRICTE: L'utilisateur exige de trader UNIQUEMENT À LA BAISSE (SELL_ONLY). Ne sélectionne que des configurations de VENTE optimales (résistances, pullbacks baissiers, FVG bear, OB bear) !"
        else:
            dir_clause = "\nCONTRAINTE: Mode MIXTE (Achat et Vente autorisés). Analyse et sélectionne les meilleures opportunités haussières et baissières."

        selection_prompt = f"""
        Tu es un système institutionnel d'élite de sélection de marchés financiers (Smart Money Concepts & Breakout).
        Voici un résumé technique des marchés ouverts en {timeframe} avec détection institutionnelle (volatilité, AMD, FVG, BOS, Liquidity Grab) :
        {json.dumps(compact_summary)}
        {dir_clause}
        
        RÈGLES STRICTES DE SÉLECTION (QUALITÉ AVANT TOUT) :
        1. ÉVITE ABSOLUMENT les marchés en sur-extension (RSI > 70 ou RSI < 30) ou en consolidation sans direction claire.
        2. NE SÉLECTIONNE QUE les marchés avec un SETUP INSTITUTIONNEL ÉVIDENT :
           - Une Phase d'Expansion avec FVG ou BOS/MSS clairs.
           - Un Liquidity Grab validé (balayage des stops).
        3. Si AUCUN marché ne présente ces configurations parfaites, tu DOIS retourner une liste vide [].
        4. Sélectionne au maximum 3 marchés présentant la plus haute probabilité de gain immédiat.
        
        Réponds UNIQUEMENT avec un objet JSON au format suivant, sans aucun autre texte ou markdown :
        {{"selected": ["SYMBOLE1", "SYMBOLE2"]}} (ou {{"selected": []}} si rien n'est bon).
        """
        has_ai = False
        if hasattr(self.router, 'has_available_provider'):
            has_ai = self.router.has_available_provider()
        elif hasattr(self.router, 'status'):
            st = self.router.status()
            has_ai = any(v.get('available') for k, v in st.items() if k in ('openrouter', 'gemini'))

        if has_ai:
            _emit(f"Sélection des meilleurs marchés parmi {len(tech_summaries)}...")
            response = self.router.ask_text(selection_prompt, stop_event=stop_event)
        else:
            _emit(f"⚡ [Mode Rapide Cerveau Local] Sélection instantanée par scoring SMC...")
            response = None
        
        def _score_setup(s):
            feat = tech_summaries[s]
            s_val = float(feat.get('_atr_rel') or 0) * 1000
            if feat.get('amd_phase') in ('Manipulation', 'Expansion', 'Distribution'): s_val += 25
            if feat.get('fvg') and feat.get('fvg') != 'None': s_val += 20
            if feat.get('bos_mss') and feat.get('bos_mss') != 'None': s_val += 20
            if feat.get('liquidity_grab') and feat.get('liquidity_grab') != 'None': s_val += 30
            rsi = float(feat.get('rsi_14') or 50)
            if rsi >= 70 or rsi <= 30: s_val -= 45 # Anti-mouton : sanctionne les mouvements sur-étendus
            tr = str(feat.get('trend') or '').upper()
            if dir_filter_upper == 'BUY_ONLY':
                if 'BULL' in tr: s_val += 30
                elif 'BEAR' in tr: s_val -= 50
            elif dir_filter_upper == 'SELL_ONLY':
                if 'BEAR' in tr: s_val += 30
                elif 'BULL' in tr: s_val -= 50
            return s_val

        selected_symbols = []
        try:
            if not response:
                raise ValueError("Aucune réponse du routeur IA")
            cleaned = response.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            
            selection_data = json.loads(cleaned.strip())
            selected_symbols = [
                sym for sym in selection_data.get("selected", [])
                if sym in tech_summaries
            ]
        except Exception as e:
            logger.warning(f"Sélection IA indisponible ({e}) -> Cerveau Local : scoring par confluences SMC et volatilité.")
            # Fallback Cerveau Local : trie par confluences institutionnelles et volatilité
            sorted_setups = sorted(tech_summaries.keys(), key=_score_setup, reverse=True)
            # Ne prendre que ceux avec un score décent (>= 30) pour éviter les marchés moyens
            selected_symbols = [s for s in sorted_setups if _score_setup(s) >= 30][:3]
            
            if selected_symbols:
                _emit(f"🧠 [Cerveau Local] Marchés de haute qualité retenus : {', '.join(selected_symbols)}")
            else:
                _emit("🧠 [Cerveau Local] Aucun marché ne remplit les critères d'exigence minimaux.")
            
        if not selected_symbols:
            _emit("Aucune opportunité évidente trouvée lors du scan de surface. Soyez patient.")

        _emit(f"Marchés sélectionnés (A+ seulement) : {selected_symbols}")

        # 4. Deep analysis for selected markets
        results = []
        if ai_queue is not None:
            _emit("Mise en file d'attente de l'analyse IA (Asynchrone)...")
            for sym in selected_symbols:
                ai_queue.put((sym, timeframe, dir_filter_upper))
        else:
            for i, sym in enumerate(selected_symbols):
                # Abort if stop requested
                if stop_event and stop_event.is_set():
                    _emit("Scan annulé.")
                    break
                if i > 0 and has_ai:
                    _emit(f"Pause 1s (rate limit)... ({i}/{len(selected_symbols)})")
                    for _ in range(1):
                        if stop_event and stop_event.is_set():
                            break
                        time.sleep(1)
                    
                _emit(f"Analyse approfondie de {sym} ({i+1}/{len(selected_symbols)})...")
                res = self.analyze_symbol(sym, timeframe, stop_event=stop_event, direction_filter=dir_filter_upper)
                results.append(res)

        if self.order_block_scan_enabled:
            _emit("Scan Order Blocks actifs...")
            try:
                ob_symbols = open_markets if open_markets else list(tech_summaries.keys())
                ob_results = self.scan_for_order_blocks(
                    ob_symbols,
                    timeframe,
                    direction_filter=dir_filter_upper,
                    stop_event=stop_event,
                )
                results.extend(ob_results)
                _emit(f"Order Blocks valides: {len(ob_results)}")
            except Exception:
                logger.exception("Order Block scan failed")

            # Scan for PENDING Order Block zones (non-retested, for limit orders)
            _emit("Scan zones OB en attente (Pending Orders)...")
            try:
                pending_ob_results = self.scan_for_pending_ob_orders(
                    ob_symbols,
                    timeframe,
                    direction_filter=dir_filter_upper,
                    stop_event=stop_event,
                )
                results.extend(pending_ob_results)
                _emit(f"📌 Zones OB pour ordres limites: {len(pending_ob_results)}")
            except Exception:
                logger.exception("Pending Order Block scan failed")

            # Scan for PENDING CRT (Candle Range Theory + SMC Limit Orders)
            _emit("Scan zones CRT universel en attente (Pending Limits)...")
            try:
                crt_symbols = open_markets if open_markets else list(tech_summaries.keys())
                pending_crt_results = self.scan_for_crt_pending_orders(
                    crt_symbols,
                    timeframe,
                    direction_filter=dir_filter_upper,
                    stop_event=stop_event,
                )
                results.extend(pending_crt_results)
                _emit(f"🎯 Setups CRT Pending Limits validés: {len(pending_crt_results)}")
            except Exception:
                logger.exception("Pending CRT scan failed")
            
        if closed_markets:
            results.append({
                "status": "CLOSED",
                "symbols": closed_markets
            })
            
        return results

    def analyze_symbol(self, symbol: str, timeframe: str, tech_features_cache: dict = None, stop_event=None, direction_filter: str = None) -> dict:
        """
        Performs a full analysis on a symbol.
        1. Fetches technical data
        2. Fetches news context
        3. Asks OpenRouter for a trade decision
        4. Calculates risk and lot size
        """
        if direction_filter is None:
            direction_filter = getattr(self, 'allowed_trade_direction', 'BOTH')
        dir_filter_upper = str(direction_filter or 'BOTH').upper()

        logger.info(f"Starting deep analysis for {symbol} on {timeframe} [Direction: {dir_filter_upper}]")
        
        # 1. Technical Data
        amd_context = {}
        if tech_features_cache:
            tech_features = tech_features_cache
            # For simplicity, if we have cache, we might miss detailed AMD context, 
            # but we can re-fetch or assume it's part of a broader cache.
            # Re-fetch just df to get full AMD context.
            df = self.market_data.get_dataframe(symbol, timeframe)
            if df is not None and not df.empty:
                amd_context = self.market_data.analyze_amd(df)
        else:
            df = self.market_data.get_dataframe(symbol, timeframe)
            if df is None:
                return {"error": f"Failed to fetch market data for {symbol}"}
                
            df = self.tech.add_all_indicators(df)
            tech_features = self.tech.extract_latest_features(df)
            amd_context = self.market_data.analyze_amd(df)

        # Get broker constraints early to pass to detectors
        symbol_info = self.mt5.get_symbol_info(symbol)
        min_distance = 0.0001
        spread_price = 0.0
        if symbol_info:
            point = float(symbol_info.get('point', 0.00001))
            trade_stops_level = symbol_info.get('trade_stops_level', 0)
            spread_pts = float(symbol_info.get('spread', 0))
            spread_price = spread_pts * point
            min_distance = max(trade_stops_level * point, spread_price) * 1.2
            if min_distance == 0:
                min_distance = point * 10

        # 1b. Higher Timeframe (HTF) Data for Context & CRT
        htf = "H1"
        if timeframe.upper() == "M15": htf = "H1"
        elif timeframe.upper() in ["H1", "H4"]: htf = "D1"

        htf_df = self.market_data.get_dataframe(symbol, htf, num_bars=700)
        htf_trend = "Unknown"
        htf_ema50 = "Unknown"
        htf_atr = float(tech_features.get('atr_14') or 0.001)
        if htf_df is not None and not htf_df.empty:
            htf_df = self.tech.add_all_indicators(htf_df)
            htf_feat = self.tech.extract_latest_features(htf_df)
            htf_trend = htf_feat.get('trend', 'Unknown')
            htf_ema50 = htf_feat.get('ema_50', 'Unknown')
            htf_atr = float(htf_feat.get('atr_14') or htf_atr)

        # 1c. Candle Range Theory (CRT) Setup Detection on HTF (H1, H4 ou D1)
        crt_setup = None
        crt_context = "Aucun setup CRT actif sur l'unité supérieure."
        if htf_df is not None and len(htf_df) >= 3:
            try:
                raw_cfg = getattr(self, 'config', {}) or {}
                crt_cfg = raw_cfg.get('crt', {}) if isinstance(raw_cfg, dict) else {}
                order_mode = str(crt_cfg.get('order_mode', 'both')).lower()
                use_pending = crt_cfg.get('use_pending_orders', True) and order_mode != "market"
                allow_market = order_mode in ("both", "market") or not use_pending

                # 1. Modèle Pending Limit CRT + ICT/SMC (si autorisé par la configuration)
                if use_pending:
                    crt_setup = self.crt_detector.detect_smc_pending(
                        df_htf=htf_df,
                        df_ltf=df,
                        atr_ltf=float(tech_features.get('atr_14') or 0.0006),
                        spread=spread_price,
                        min_rr=float(crt_cfg.get('min_rr', 1.5)),
                        expiration_bars=int(crt_cfg.get('expiration_bars', 4)),
                    )

                # 2. Modèle Ordre Simple Direct au Marché (si pas de pending ou si mode marché autorisé)
                if not crt_setup and allow_market:
                    crt_setup = self.crt_detector.detect(htf_df, atr=htf_atr, spread=spread_price)

                if crt_setup:
                    kz_tag = f" [Killzone {crt_setup.killzone_name}]" if crt_setup.in_killzone else ""
                    if getattr(crt_setup, 'is_pending', False):
                        crt_context = (
                            f"SETUP CRT + ICT/SMC UNIVERSEL ({crt_setup.direction}) VALIDÉ {timeframe}/{htf}{kz_tag} :\n"
                            f"- Modèle : {crt_setup.order_type} en attente (Pending Limit)\n"
                            f"- Range C1 ({htf}) : Bas={crt_setup.c1_low:.5f}, Haut={crt_setup.c1_high:.5f}, Équilibre 50%={crt_setup.c1_mid:.5f}\n"
                            f"- Sweep {timeframe} : Extrême={crt_setup.sweep_price:.5f} (prof. {crt_setup.sweep_depth:.5f})\n"
                            f"- Confirmation MSS : Cassure structurelle à {crt_setup.mss_price:.5f}\n"
                            f"- FVG : Zone [{crt_setup.fvg_low:.5f} - {crt_setup.fvg_high:.5f}] -> Entrée LIMIT au milieu = {crt_setup.entry_price:.5f}\n"
                            f"- Niveaux : SL={crt_setup.sl:.5f}, TP={crt_setup.tp:.5f} (R:R {crt_setup.rr:.2f}), Expiration={crt_setup.expiration_bars} bougies"
                        )
                    else:
                        crt_context = (
                            f"SETUP CRT ({crt_setup.direction}) VALIDÉ SUR {htf}{kz_tag} :\n"
                            f"- Range C1 : Bas={crt_setup.c1_low:.5f}, Haut={crt_setup.c1_high:.5f}, Équilibre 50%={crt_setup.c1_mid:.5f}\n"
                            f"- Manipulation C2 : Sweep={crt_setup.sweep_price:.5f} (prof. {crt_setup.sweep_depth:.5f}), Clôture dans range={crt_setup.c2_close:.5f}\n"
                            f"- Niveaux : SL={crt_setup.sl_raw:.5f}, TP1 (50% EQ)={crt_setup.tp1:.5f}, TP2 (Borne opposée)={crt_setup.tp2:.5f}"
                        )
                    logger.info(f"[CRT] {symbol}: {crt_context}")
            except Exception as _ce:
                logger.warning(f"CRT detector error for {symbol}: {_ce}")

        # 1d. Boomerang Setup Detection
        try:
            boom = self.boomerang.detect(
                df, symbol, timeframe,
                htf_trend=htf_trend,
                amd_context=amd_context,
            )
            boomerang_context = boom.summary()
            boomerang_decision = boom.decision    # 'VALID'|'WEAK'|'INVALID'
            boomerang_score   = boom.score
            boomerang_dir     = boom.direction
        except Exception as _be:
            logger.warning(f"Boomerang detector error for {symbol}: {_be}")
            boomerang_context = "Boomerang: detection error"
            boomerang_decision = "INVALID"
            boomerang_score    = 0
            boomerang_dir      = "NONE"

        # 1e. Order Block (SMC) Setup Detection
        try:
            valid_obs = self.ob_detector.detect(
                df, symbol, timeframe,
                htf_trend=htf_trend,
                direction_filter=dir_filter_upper,
            )
            if valid_obs:
                top_ob = valid_obs[0]
                ob_context = top_ob.summary()
                top_ob_dir = top_ob.direction
                top_ob_score = top_ob.score
                has_active_ob = True
            else:
                ob_context = "Aucun Order Block actif/retesté actuellement sur ce timeframe."
                top_ob_dir = "NONE"
                top_ob_score = 0
                has_active_ob = False
        except Exception as _oe:
            logger.warning(f"OrderBlock detector error for {symbol}: {_oe}")
            ob_context = "OrderBlock detector: erreur ou données insuffisantes."
            top_ob_dir = "NONE"
            top_ob_score = 0
            has_active_ob = False

        # Calcul des métriques de sur-extension et de repli (Anti-FOMO / Anti-Chasing)
        close_p = float(tech_features.get('close') or 0.0)
        ema50_p = float(tech_features.get('ema_50') or close_p)
        atr_p   = float(tech_features.get('atr_14') or 0.0001)
        rsi_val = float(tech_features.get('rsi_14') or 50.0)
        ema_dist_atr = ((close_p - ema50_p) / atr_p) if atr_p > 0 else 0.0

        if rsi_val >= 70:
            rsi_status = f"SURACHAT SÉVÈRE (RSI {rsi_val:.1f} >= 70). ACHAT INTERDIT."
        elif rsi_val <= 30:
            rsi_status = f"SURVENTE SÉVÈRE (RSI {rsi_val:.1f} <= 30). VENTE INTERDITE."
        else:
            rsi_status = f"Neutre / Favorable (RSI {rsi_val:.1f})"
        
        # 2. News Context (optional: Gemini+Search can be slow when quotas are exhausted)
        if self.live_news_enabled:
            news_context = self.router.analyze_news(symbol, stop_event=stop_event)
        else:
            news_context = {
                "market_impact": "NEUTRAL",
                "direction": "NEUTRAL",
                "summary": "Live news disabled in fast mode.",
                "events": [],
            }
                
        # 3. Order Flow & Microstructure (DOM L2, Heatmap Walls, CVD, Volume Profile, Absorption)
        ticks_df = None
        dom_snapshot = None
        try:
            ticks_df = self.market_data.get_ticks_dataframe(symbol, count=1000)
            dom_snapshot = self.market_data.get_order_book_snapshot(symbol)
        except Exception as _ofe:
            logger.debug(f"[OF] Erreur lecture ticks/DOM pour {symbol}: {_ofe}")

        of_confluence = self.order_flow_engine.evaluate_setup_confluence(
            symbol=symbol,
            crt_setup=crt_setup,
            ticks_df=ticks_df,
            dom_data=dom_snapshot,
            candles_df=df,
            atr=atr_p,
        )
        if crt_setup:
            crt_setup.order_flow_confluence = of_confluence
            crt_setup.order_flow_confirmed = of_confluence.is_confirmed
            crt_setup.order_flow_score = of_confluence.confluence_score

        dom_context = "Depth of Market & Order Flow: Not available for this broker/symbol."
        if dom_snapshot and isinstance(dom_snapshot, dict) and dom_snapshot.get("available"):
            try:
                bids_v = float(dom_snapshot.get('bids_volume', 0.0))
                asks_v = float(dom_snapshot.get('asks_volume', 0.0))
                bp = float(dom_snapshot.get('buy_pressure', 50.0))
                sp = float(dom_snapshot.get('sell_pressure', 50.0))
                imb = float(dom_snapshot.get('imbalance_ratio', 0.0))
                score_str = f"{float(of_confluence.confluence_score):.1f}/100" if of_confluence else "N/A"
                badge_str = f" [{of_confluence.summary_badge}]" if of_confluence and getattr(of_confluence, 'summary_badge', '') else ""
                dom_context = (
                    f"Order Flow & DOM Microstructure:\n"
                    f"  - DOM Bids: {bids_v:.1f} ({bp:.1f}% Buy Pressure)\n"
                    f"  - DOM Asks: {asks_v:.1f} ({sp:.1f}% Sell Pressure)\n"
                    f"  - Imbalance Ratio: {imb:+.2f}\n"
                    f"  - Order Flow Score: {score_str}{badge_str}\n"
                    f"  - Details: {', '.join(of_confluence.details) if of_confluence and of_confluence.details else 'Standard flow'}"
                )
            except Exception:
                dom_context = "Order Flow & DOM: Available."

        # ─── ÉVALUATION DU PROFIL DE TRADER PROFESSIONNEL (SI ACTIF) ───
        if hasattr(self, 'strategy_profile_manager') and self.strategy_profile_manager.is_active():
            active_strat = self.strategy_profile_manager.get_active_strategy()
            if active_strat and active_strat.is_symbol_supported(symbol):
                ltf_df = self.market_data.get_dataframe(symbol, "M5", num_bars=100)
                pro_signal = self.strategy_profile_manager.evaluate_symbol(
                    symbol=symbol,
                    df_primary=df,
                    df_htf=htf_df,
                    df_ltf=ltf_df,
                    spread=spread_price,
                    min_distance=min_distance,
                    tech_features=tech_features,
                    amd_context=amd_context,
                    order_flow_confluence=of_confluence,
                    news_context=news_context,
                )

                active_mode = self.strategy_profile_manager.get_active_mode()

                # Cas 1 : Le trader pro valide un setup A+
                if pro_signal:
                    logger.info(
                        f"👑 [Copie {pro_signal.trader}] Signal {pro_signal.direction} validé sur {symbol} "
                        f"(Mode {active_mode.upper()}). Entry={pro_signal.entry}, SL={pro_signal.sl}, TP={pro_signal.tp}"
                    )
                    pro_decision = {
                        "action": pro_signal.direction,
                        "confidence": 95,
                        "entry": pro_signal.entry,
                        "sl": pro_signal.sl,
                        "tp": pro_signal.tp,
                        "rr": pro_signal.rr,
                        "reason": f"[{pro_signal.trader} · {active_mode.upper()}] {pro_signal.reason}",
                        "is_pending": pro_signal.is_pending,
                        "order_type": pro_signal.order_type,
                        "pro_trader": pro_signal.trader,
                        "profile_id": pro_signal.profile_id,
                        "cluster_orders": pro_signal.cluster_orders,
                        "be_at_r": pro_signal.be_at_r,
                        "trail": pro_signal.trail,
                        "decay_risk_mult": pro_signal.decay_risk_mult,
                        "simulation_only": (active_mode == "simulation"),
                    }

                    account_info = self.mt5.get_account_info()
                    val = self.risk.validate_proposal(pro_decision, account_info, symbol_info)
                    if not val['valid']:
                        logger.warning(f"👑 [Copie {pro_signal.trader}] Risque MT5 rejeté : {val['reason']}")
                        return {"error": f"Risk validation failed: {val['reason']}"}

                    return {
                        "status": "SIGNAL",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "tech_features": tech_features,
                        "amd_context": amd_context,
                        "news_context": news_context,
                        "decision": val['proposal']
                    }

                # Cas 2 : Le trader pro n'a pas de setup (Discipline du Trader Pro = WAIT)
                else:
                    logger.info(
                        f"👑 [Copie {active_strat.name}] Aucun setup selon les règles du trader pour {symbol}. "
                        f"Discipline respectée = WAIT."
                    )
                    wait_decision = {
                        "action": "WAIT",
                        "confidence": 0,
                        "entry": close_p,
                        "sl": 0.0,
                        "tp": 0.0,
                        "reason": f"Discipline {active_strat.name} : Conditions de trade non réunies pour {symbol} (Attente de setup A+).",
                        "pro_trader": active_strat.name,
                    }
                    return {
                        "status": "WAIT",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "tech_features": tech_features,
                        "amd_context": amd_context,
                        "news_context": news_context,
                        "decision": wait_decision
                    }

        # Determine Market Regime and query Strategy Engine (incluant CRT, Sweeps, Order Blocks)
        market_regime = MarketContextEngine.detect_market_regime(tech_features, htf_trend, amd_context)
        proposals = self.strategy_selector.select_and_evaluate(
            market_regime=market_regime,
            tech_features=tech_features, 
            amd_context=amd_context, 
            htf_trend=htf_trend, 
            price=close_p, 
            spread=spread_price, 
            min_distance=min_distance,
            valid_obs=valid_obs if has_active_ob else None,
            crt_setup=crt_setup,
            order_flow_confluence=of_confluence,
        )
        
        strategy_context = "Aucune stratégie spécifique activée."
        if proposals:
            strategy_context = "Stratégies activées par le Rule Engine :\n"
            for p in proposals:
                strategy_context += f"- {p['strategy']}: {p['direction']} ({p['reason']})\n"

        if dir_filter_upper == "BUY_ONLY":
            dir_clause_ai = "7. DIRECTIONAL CONSTRAINT (MANDATORY): The trader is strictly restricted to BUY_ONLY (Hausse seule). You must NEVER output SELL. If no valid BUY setup is available, output 'WAIT'."
        elif dir_filter_upper == "SELL_ONLY":
            dir_clause_ai = "7. DIRECTIONAL CONSTRAINT (MANDATORY): The trader is strictly restricted to SELL_ONLY (Baisse seule). You must NEVER output BUY. If no valid SELL setup is available, output 'WAIT'."
        else:
            dir_clause_ai = "7. DIRECTION: Both BUY and SELL setups are allowed."

        memory_clause = ""
        if self.trade_memory:
            recent_lessons = self.trade_memory.get_recent_lessons(symbol=symbol, limit=3)
            if not recent_lessons:
                recent_lessons = self.trade_memory.get_recent_lessons(limit=3)
            if recent_lessons:
                memory_clause = "\n        8. HISTORICAL LESSONS & SELF-CRITIQUE (AVOID REPEATING THESE PAST MISTAKES):\n" + "\n".join(f"           - {l}" for l in recent_lessons)

        sniper_clause = ""
        if getattr(self.risk, 'risk_profile', '') == "SNIPER":
            sniper_clause = "\n        9. 🎯 SNIPER MODE IS ACTIVE (MANDATORY): You are operating in SNIPER mode. You must be ULTRA-SELECTIVE. If there is NO 'Order Block', NO 'Boomerang', and NO 'Candle Range Theory (CRT)' setup active, you MUST output 'WAIT'. Do not attempt to predict tops/bottoms. Only trade A+ textbook setups."

        system_prompt = f"""
        You are an elite quantitative institutional trading AI specializing in Smart Money Concepts (SMC) and Breakout-Retest.
        You analyze technical and fundamental data to identify high-probability institutional setups.
        
        CRITICAL RULES YOU MUST OBEY:
        1. For BUY: sl MUST BE STRICTLY LESS THAN entry. tp MUST BE STRICTLY GREATER THAN entry.
        2. For SELL: sl MUST BE STRICTLY GREATER THAN entry. tp MUST BE STRICTLY LESS THAN entry.
        3. MINIMUM STOP DISTANCE: Your sl and tp MUST be at least {min_distance:.5f} away from your entry price to respect the broker's minimum distance limits. Do NOT place tight stops.
        4. SPREAD AWARENESS: The current bid/ask spread for this symbol is {spread_price:.5f}. Your TP must be large enough to overcome the spread cost — at minimum 3× the spread. Do NOT trade if your expected move is less than 5× the spread.
        5. ANTI-FOMO / NO MARKET CHASING (MANDATORY):
           - NEVER buy green breakout candles at the top of a rally or sell red candles at the bottom of a drop!
           - Do NOT enter blindly just because the trend is Bullish or Bearish.
           - You MUST require price to pull back or retest a key institutional level (Order Block retest, FVG mitigation, CRT liquidity sweep, or EMA 50 dynamic support) before entering.
           - If RSI >= 70 or price is extended > 2.5x ATR above EMA 50, BUY is STRICTLY FORBIDDEN (output "WAIT").
           - If RSI <= 30 or price is extended > 2.5x ATR below EMA 50, SELL is STRICTLY FORBIDDEN (output "WAIT").
        6. DEFAULT TO WAIT (CAPITAL PRESERVATION):
           - Professional traders wait 90% of the time for the perfect setup.
           - If there is NO confirmed institutional setup (no active Order Block retest, no valid Boomerang, no valid CRT setup, no clear pullback), your decision MUST BE "WAIT".
           - Only recommend BUY or SELL if at least 2 institutional confluences align and R:R is >= 1.5.
        {dir_clause_ai}
        {memory_clause}
        {sniper_clause}
        
        You must output ONLY valid JSON matching this schema:
        {{
            "action": "BUY" | "SELL" | "WAIT",
            "confidence": 0-100,
            "entry": float,
            "sl": float,
            "tp": float,
            "reason": "short explanation describing the setup and institutional confluences"
        }}
        """
        
        prompt = f"""
        Analyze {symbol} on {timeframe} timeframe (Deep Scan: 700 candles history for robust structural understanding).
        Current Price: {tech_features.get('close')}
        Spread (bid/ask cost): {spread_price:.5f} — your TP must cover at least 5x this.
        Trend Bias: {tech_features.get('trend')}
        EMA 50: {tech_features.get('ema_50')}
        EMA 200: {tech_features.get('ema_200')}
        Price Distance to EMA 50: {ema_dist_atr:+.2f} ATR (Values > +2.5 or < -2.5 mean overextended!)
        RSI (14): {tech_features.get('rsi_14')} — Status: {rsi_status}
        MACD: {tech_features.get('macd')} (Hist: {tech_features.get('macd_hist')})
        ATR (14): {tech_features.get('atr_14')}
        Last Swing High: {tech_features.get('last_swing_high')}
        Last Swing Low: {tech_features.get('last_swing_low')}
        Williams Fractal UP (last bar): {tech_features.get('fractal_up')} | Level: {tech_features.get('last_fractal_high')}
        Williams Fractal DOWN (last bar): {tech_features.get('fractal_down')} | Level: {tech_features.get('last_fractal_low')}
        ZigZag Trend: {'UP (+1)' if tech_features.get('zigzag_trend') == 1 else 'DOWN (-1)' if tech_features.get('zigzag_trend') == -1 else 'NEUTRAL'}
        ZigZag Last Pivot: {tech_features.get('zigzag_pivot')} @ {tech_features.get('zigzag_value')}

        {dom_context}

        ─── ORDER BLOCK (SMC) SETUP ANALYSIS ───
        {ob_context}
        ────────────────────────────────────────

        ─── BOOMERANG SETUP ANALYSIS ───
        {boomerang_context}
        ────────────────────────────────

        ─── CANDLE RANGE THEORY (CRT) SETUP ───
        {crt_context}
        ───────────────────────────────────────
        
        ─── MARKET REGIME & STRATEGY ENGINE ───
        Market Regime Detected: {market_regime}
        {strategy_context}
        ───────────────────────────────────────

        AMD / Smart Money Concepts Context:
        Phase: {amd_context.get('Phase', 'Unknown')}
        FVG Detected: {amd_context.get('FVG', 'None')}
        BOS/MSS Detected: {amd_context.get('BOS_MSS', 'None')}
        Liquidity Grab (Fakeout): {amd_context.get('Liquidity_Grab', 'None')}

        Higher Timeframe ({htf}) Context:
        HTF Trend Bias: {htf_trend}
        HTF EMA 50: {htf_ema50}
        CRITICAL: Never BUY if HTF Trend is strongly Bearish. Never SELL if HTF Trend is strongly Bullish.

        News Context:
        Impact: {news_context.get('market_impact')}
        Direction: {news_context.get('direction')}
        Summary: {news_context.get('summary')}

        CRT pre-analysis: {crt_context.splitlines()[0] if crt_setup else 'Aucun setup'}.
        Boomerang pre-analysis: {boomerang_dir} (score {boomerang_score:.0f}/100, decision: {boomerang_decision}).
        Order Block pre-analysis: {top_ob_dir} (score {top_ob_score:.0f}/100, active: {has_active_ob}).
        If any institutional setup (CRT, Order Block, Boomerang) is confirmed, prioritize entering on the retest.
        If NO setup is active and price is chasing market direction, reply "WAIT".

        Provide your trade recommendation.
        """
        
        has_ai = False
        if hasattr(self.router, 'has_available_provider'):
            has_ai = self.router.has_available_provider()
        elif hasattr(self.router, 'status'):
            st = self.router.status()
            has_ai = any(v.get('available') for k, v in st.items() if k in ('openrouter', 'gemini'))

        response = None
        if has_ai:
            logger.info("Asking AI router for trade decision...")
            response = self.router.ask_text(prompt, system_prompt=system_prompt, stop_event=stop_event)
        else:
            logger.info(f"⚡ [Cerveau Local Rapide] Bypassement IA externe pour {symbol} -> Analyse quantitative instantanée.")
        
        decision = None
        if response:
            try:
                # Clean up potential markdown formatting
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                    
                decision = json.loads(cleaned.strip())
            except Exception as _je:
                logger.warning(f"Failed to parse AI decision JSON: {_je}")
                decision = None

        if not decision or not isinstance(decision, dict) or "action" not in decision:
            logger.warning(
                f"🧠 [Cerveau Local] API externe indisponible ou quota dépassé pour {symbol} "
                f"-> Bascule automatique et immédiate sur le Cerveau Quantitatif Local !"
            )
            decision = self.local_brain.decide(
                symbol=symbol,
                timeframe=timeframe,
                tech_features=tech_features,
                amd_context=amd_context,
                boomerang_data={"decision": boomerang_decision, "direction": boomerang_dir, "score": boomerang_score},
                order_blocks=valid_obs,
                htf_trend=htf_trend,
                spread_price=spread_price,
                min_distance=min_distance,
                direction_filter=dir_filter_upper,
                crt_setup=crt_setup,
            )

        # ─── PHASE 2: OVERRIDE RISK WITH STRATEGY ENGINE ───
        if decision and decision.get("action") in ("BUY", "SELL") and proposals:
            # Find the first proposal that matches the AI direction
            matching_prop = next((p for p in proposals if p["direction"] == decision["action"]), None)
            if matching_prop:
                logger.info(f"Phase 2: Overwriting entry/sl/tp with Strategy Engine values for {matching_prop['strategy']}")
                decision["entry"] = matching_prop["entry"]
                decision["sl"] = matching_prop["sl"]
                decision["tp"] = matching_prop["tp"]
                decision["reason"] += f" [Strategy: {matching_prop['strategy']}]"
                if "tp1" in matching_prop:
                    decision["tp1"] = matching_prop["tp1"]
                if "tp2" in matching_prop:
                    decision["tp2"] = matching_prop["tp2"]

                # Prise en charge explicite des deux types d'ordres : Simples (DEAL) vs En attente (PENDING)
                if matching_prop.get("is_pending"):
                    decision["is_pending"] = True
                    decision["order_type"] = matching_prop.get("order_type", "BUY_LIMIT" if decision["action"] == "BUY" else "SELL_LIMIT")
                    decision["expiration_bars"] = matching_prop.get("expiration_bars", 4)
                    decision["expiration_seconds"] = matching_prop.get("expiration_seconds", 3600)
                else:
                    decision["is_pending"] = False
                    decision["order_type"] = "MARKET"

                # ─── PHASE 3: MEMORY & PERFORMANCE ENGINE (EXPECTANCY) ───
                if self.trade_memory:
                    import datetime
                    current_hour = datetime.datetime.now(datetime.timezone.utc).hour
                    if 0 <= current_hour < 8:
                        session = "ASIA"
                    elif 8 <= current_hour < 13:
                        session = "LONDON"
                    elif 13 <= current_hour < 17:
                        session = "NEW_YORK_LONDON_OVERLAP"
                    else:
                        session = "NEW_YORK"
                        
                    strat_name = matching_prop["strategy"]
                    perf = self.trade_memory.check_expectancy(symbol, session, strat_name)
                    
                    if perf["status"] == "DISABLED":
                        logger.warning(
                            f"[Phase 3 Expectancy] {symbol} {session}: Stratégie {strat_name} "
                            f"désactivée (Expectancy = {perf['expectancy_r']}R < 0 sur {perf['trades_count']} trades)."
                        )
                        decision["action"] = "WAIT"
                        decision["reason"] = f"Phase 3 Expectancy: Stratégie {strat_name} non rentable en session {session} ({perf['expectancy_r']}R)."

        # ─── GARDE-FOU ALGORITHMIQUE DÉTERMINISTE (ANTI-FOMO / QUALITÉ D'ENTRÉE) ───
        proposed_action = str(decision.get('action', 'WAIT')).upper()
        if proposed_action in ('BUY', 'SELL'):
            # 0. Filtre Directionnel configuré par l'utilisateur (BUY_ONLY / SELL_ONLY)
            if dir_filter_upper == "BUY_ONLY" and proposed_action == "SELL":
                logger.info(f"[Direction Filter] {symbol}: Rejet SELL car politique configurée sur BUY_ONLY.")
                decision['action'] = 'WAIT'
                decision['reason'] = "Rejet SELL : la politique de trading est configurée sur BUY_ONLY (Hausse seule)."
            elif dir_filter_upper == "SELL_ONLY" and proposed_action == "BUY":
                logger.info(f"[Direction Filter] {symbol}: Rejet BUY car politique configurée sur SELL_ONLY.")
                decision['action'] = 'WAIT'
                decision['reason'] = "Rejet BUY : la politique de trading est configurée sur SELL_ONLY (Baisse seule)."

            # 1. Filtre Anti-FOMO RSI (interdit d'acheter au sommet ou de vendre au creux)
            elif proposed_action == 'BUY' and rsi_val >= 70:
                logger.warning(f"[Anti-FOMO] {symbol}: Rejet BUY car RSI ({rsi_val:.1f}) >= 70 (Surachat). Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Anti-FOMO : Rejet d'achat sur sommet (RSI {rsi_val:.1f} >= 70). Attente de repli."
            elif proposed_action == 'SELL' and rsi_val <= 30:
                logger.warning(f"[Anti-FOMO] {symbol}: Rejet SELL car RSI ({rsi_val:.1f}) <= 30 (Survente). Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Anti-FOMO : Rejet de vente sur creux (RSI {rsi_val:.1f} <= 30). Attente de rebond."

            # 2. Filtre Anti-Chasing de sur-extension EMA 50 (> 2.8 ATR)
            elif proposed_action == 'BUY' and ema_dist_atr > 2.8:
                logger.warning(f"[Anti-Chasing] {symbol}: Rejet BUY car cours sur-étendu (+{ema_dist_atr:.1f} ATR au-dessus EMA 50). Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Anti-Chasing : Cours sur-étendu (+{ema_dist_atr:.1f} ATR au-dessus EMA 50). Attente de pullback."
            elif proposed_action == 'SELL' and ema_dist_atr < -2.8:
                logger.warning(f"[Anti-Chasing] {symbol}: Rejet SELL car cours sur-étendu ({ema_dist_atr:.1f} ATR sous EMA 50). Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Anti-Chasing : Cours sur-étendu ({ema_dist_atr:.1f} ATR sous EMA 50). Attente de pullback."

            # 3. Filtre d'alignement avec la tendance Higher Timeframe (HTF)
            elif proposed_action == 'BUY' and str(htf_trend).lower() in ('bearish', 'strong bearish'):
                logger.warning(f"[HTF Filter] {symbol}: Rejet BUY contre tendance HTF Bearish. Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Tendance HTF contraire ({htf_trend}). Achat contre-tendance rejeté."
            elif proposed_action == 'SELL' and str(htf_trend).lower() in ('bullish', 'strong bullish'):
                logger.warning(f"[HTF Filter] {symbol}: Rejet SELL contre tendance HTF Bullish. Forcé en WAIT.")
                decision['action'] = 'WAIT'
                decision['reason'] = f"Tendance HTF contraire ({htf_trend}). Vente contre-tendance rejetée."

            # 4. Filtre de confluence institutionnelle (exige un vrai setup avant d'entrer)
            else:
                has_strong_confluence = (
                    (boomerang_decision == 'VALID' and boomerang_dir == proposed_action) or
                    (has_active_ob and top_ob_dir == proposed_action)
                )
                has_weak_confluence = (
                    (amd_context.get('Phase') in ('Expansion', 'Distribution', 'Manipulation') and amd_context.get('FVG') != 'None')
                )
                conf = float(decision.get('confidence', 0))
                is_sniper = getattr(self.risk, 'risk_profile', '') == "SNIPER"

                if is_sniper and not has_strong_confluence:
                    logger.warning(f"[SNIPER Gate] {symbol}: Rejet {proposed_action}. Mode Sniper exige un Boomerang ou OB valide.")
                    decision['action'] = 'WAIT'
                    decision['reason'] = "Rejet Sniper : Confluence majeure (Order Block ou Boomerang) obligatoire manquante."
                elif not has_strong_confluence and not has_weak_confluence:
                    # Plus aucune confiance aveugle dans l'IA si la technique ne suit pas !
                    logger.warning(
                        f"[Quality Gate] {symbol}: Aucun setup institutionnel valide (ni OB, ni Boomerang, ni FVG). Forcé en WAIT."
                    )
                    decision['action'] = 'WAIT'
                    decision['reason'] = (
                        "Rejet Anti-Chasing : Aucune confluence institutionnelle (ni Order Block retesté, "
                        "ni Boomerang validé, ni FVG). Interdiction d'entrer directement au marché."
                    )

            
        if decision.get('action') == 'WAIT':
            return {"status": "WAIT", "decision": decision}
            
        # 4. Risk and Lot Size
        account_info = self.mt5.get_account_info()
        symbol_info = self.mt5.get_symbol_info(symbol)
        
        if not account_info or not symbol_info:
            return {"error": "Failed to get account or symbol info for risk calculation"}
            
        validation = self.risk.validate_proposal(decision, account_info, symbol_info)
        
        if not validation['valid']:
            return {"error": f"Risk validation failed: {validation['reason']}"}
            
        return {
            "status": "SIGNAL",
            "symbol": symbol,
            "timeframe": timeframe,
            "tech_features": tech_features,
            "amd_context": amd_context,
            "news_context": news_context,
            "decision": validation['proposal']
        }

    def analyze_open_position(self, pos: dict, timeframe: str) -> dict:
        """
        Asks the AI to analyze an currently open position to decide whether to CLOSE, MODIFY (SL/TP), or HOLD.
        pos: dictionary from mt5_connector.get_open_positions()
        """
        symbol = pos['symbol']
        pos_type = pos['type']
        profit = pos['profit']
        price_open = pos['price_open']
        price_current = pos['price_current']
        sl = pos['sl']
        tp = pos['tp']
        
        logger.info(f"Analyzing open position for {symbol} ({pos_type}) - Profit: ${profit:.2f}")
        
        # Technical Data
        df = self.market_data.get_dataframe(symbol, timeframe)
        if df is None or df.empty:
            return {"action": "HOLD", "reason": "No technical data available"}
            
        df = self.tech.add_all_indicators(df)
        tech_features = self.tech.extract_latest_features(df)
        
        system_prompt = f"""
        You are an elite quantitative trading AI managing an OPEN position.
        You must evaluate if we should let profits run, cut losses early, or take profits now if the market is reversing.
        
        CRITICAL RULES:
        1. If action is MODIFY, you MUST provide 'new_sl' and 'new_tp'. 
        2. For BUY: sl MUST BE STRICTLY LESS THAN price_current. tp MUST BE STRICTLY GREATER THAN price_current.
        3. For SELL: sl MUST BE STRICTLY GREATER THAN price_current. tp MUST BE STRICTLY LESS THAN price_current.
        
        You must output ONLY valid JSON matching this schema:
        {{
            "action": "CLOSE" | "MODIFY" | "HOLD",
            "new_sl": float | null,
            "new_tp": float | null,
            "reason": "short explanation"
        }}
        """
        
        prompt = f"""
        POSITION DETAILS:
        Symbol: {symbol}
        Type: {pos_type}
        Open Price: {price_open}
        Current Price: {price_current}
        Current SL: {sl}
        Current TP: {tp}
        Current Profit/Loss: ${profit:.2f}
        
        TECHNICAL CONTEXT ({timeframe}):
        Trend Bias: {tech_features.get('trend')}
        EMA 50: {tech_features.get('ema_50')}
        EMA 200: {tech_features.get('ema_200')}
        RSI (14): {tech_features.get('rsi_14')}
        MACD: {tech_features.get('macd')} (Hist: {tech_features.get('macd_hist')})
        
        Evaluate the open position. If we are in good profit but technicals show a reversal against us, reply with "CLOSE".
        If the trend is strong in our favor, reply with "MODIFY" to trail the Stop Loss or extend Take Profit.
        Otherwise, reply "HOLD".
        """

        has_ai = False
        if hasattr(self.router, 'has_available_provider'):
            has_ai = self.router.has_available_provider()
        elif hasattr(self.router, 'status'):
            st = self.router.status()
            has_ai = any(v.get('available') for k, v in st.items() if k in ('openrouter', 'gemini'))

        response = None
        if has_ai:
            response = self.router.ask_text(prompt, system_prompt=system_prompt)
        else:
            logger.info(f"⚡ [Cerveau Local] Évaluation directe et instantanée de la position {symbol}.")
        
        decision = None
        if response:
            try:
                cleaned = response.strip()
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                if cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                    
                decision = json.loads(cleaned.strip())
            except Exception as _e:
                logger.warning(f"Failed to parse AI position management JSON: {_e}")
                decision = None

        if not decision or not isinstance(decision, dict) or "action" not in decision:
            logger.info(f"🧠 [Cerveau Local] Évaluation locale de la position {symbol}...")
            return self.local_brain.evaluate_position(pos, tech_features)

        return decision

    def scan_for_order_blocks(self, symbols: list, timeframe: str, direction_filter: str = "BOTH", stop_event=None) -> list:
        """
        Scanne une liste de symboles a la recherche d'Order Blocks actifs (retestes).
        Utilise pour le rescanning post-trade en mode autopilot.

        Returns
        -------
        list[dict]  — signaux prets a etre executes (meme format que analyze_symbol)
        """
        results = []
        for symbol in symbols:
            time.sleep(0.02) # Yield GIL
            if stop_event and stop_event.is_set():
                break
            try:
                if not self.mt5.is_market_open(symbol):
                    continue
                df = self.market_data.get_dataframe(symbol, timeframe)
                if df is None or df.empty:
                    continue
                df = self.tech.add_all_indicators(df)

                # HTF trend for alignment scoring
                htf_map = {"M5": "M15", "M15": "H1", "H1": "H4", "H4": "D1", "D1": "D1"}
                htf = htf_map.get(timeframe.upper(), "H1")
                htf_trend = "Unknown"
                htf_df = self.market_data.get_dataframe(symbol, htf)
                if htf_df is not None and not htf_df.empty:
                    htf_df = self.tech.add_all_indicators(htf_df)
                    htf_feat = self.tech.extract_latest_features(htf_df)
                    htf_trend = htf_feat.get('trend', 'Unknown')

                obs = self.ob_detector.detect(
                    df, symbol, timeframe,
                    htf_trend=htf_trend,
                    direction_filter=direction_filter,
                )

                if not obs:
                    continue

                best_ob = obs[0]   # meilleur OB classe par score
                if best_ob.score < 65:
                    continue

                # 0. Filtre Cooldown anti-revenge trading / après perte
                is_cd, cd_reason = self.is_symbol_in_cooldown(symbol, best_ob.direction)
                if is_cd:
                    logger.info(f"[OB Rescan] {symbol}: Rejet {best_ob.direction} — Symbole en cooldown: {cd_reason}")
                    continue

                # 1. Filtre anti-spam : Pas de trade si position déjà ouverte sur le symbole
                open_pos = self.mt5.get_open_positions() or []
                if any(p.get('symbol') == symbol for p in open_pos):
                    continue

                # 2. Filtre tendance HTF : Pas de trade contre la tendance HTF
                htf_upper = htf_trend.upper()
                if best_ob.direction == 'BUY' and 'BEAR' in htf_upper:
                    logger.info(f"[OB Rescan] {symbol}: Rejet BUY contre tendance HTF ({htf_trend})")
                    continue
                elif best_ob.direction == 'SELL' and 'BULL' in htf_upper:
                    logger.info(f"[OB Rescan] {symbol}: Rejet SELL contre tendance HTF ({htf_trend})")
                    continue

                # 2b. Filtre Anti-Chute Libre / Anti-Couteau qui tombe (Anti-Falling Knife)
                # Si le marché s'effondre avec des bougies baissières successives sans réaction haussière,
                # ne jamais exécuter d'ordre marché BUY aveugle !
                if len(df) >= 3:
                    last_c = df.iloc[-1]
                    prev_c = df.iloc[-2]
                    c_last, o_last = float(last_c['close']), float(last_c['open'])
                    c_prev, o_prev = float(prev_c['close']), float(prev_c['open'])
                    h_last, l_last = float(last_c['high']), float(last_c['low'])
                    rng_last = h_last - l_last

                    if best_ob.direction == 'BUY':
                        is_bear_run = (c_last < o_last) and (c_prev < o_prev)
                        lower_wick = (c_last - l_last) if c_last < o_last else (o_last - l_last)
                        has_wick_rejection = rng_last > 0 and (lower_wick / rng_last) >= 0.30
                        has_reaction = (c_last > o_last) or has_wick_rejection or getattr(best_ob, 'has_rejection', False)
                        if is_bear_run and not has_reaction:
                            logger.info(f"[OB Rescan] {symbol}: Rejet BUY — Marché en chute libre (bougies rouges consécutives sans rejet haussier). Attente de stabilisation.")
                            continue
                    elif best_ob.direction == 'SELL':
                        is_bull_run = (c_last > o_last) and (c_prev > o_prev)
                        upper_wick = (h_last - c_last) if c_last > o_last else (h_last - o_last)
                        has_wick_rejection = rng_last > 0 and (upper_wick / rng_last) >= 0.30
                        has_reaction = (c_last < o_last) or has_wick_rejection or getattr(best_ob, 'has_rejection', False)
                        if is_bull_run and not has_reaction:
                            logger.info(f"[OB Rescan] {symbol}: Rejet SELL — Marché en forte hausse sans rejet baissier. Attente de stabilisation.")
                            continue

                # 3. Filtre RSI anti-surachat / survente
                if 'RSI_14' in df.columns:
                    rsi = float(df['RSI_14'].iloc[-1] or 50.0)
                    if best_ob.direction == 'BUY' and rsi >= 68.0:
                        logger.info(f"[OB Rescan] {symbol}: Rejet BUY sur RSI suracheté ({rsi:.1f})")
                        continue
                    elif best_ob.direction == 'SELL' and rsi <= 32.0:
                        logger.info(f"[OB Rescan] {symbol}: Rejet SELL sur RSI survendu ({rsi:.1f})")
                        continue

                # 4. Filtre de spread : Le SL doit être suffisamment éloigné du spread pour ne pas être fauché
                symbol_info = self.mt5.get_symbol_info(symbol)
                if symbol_info:
                    s_pts = float(symbol_info.get('spread', 0) or 0)
                    p_pt = float(symbol_info.get('point', 0.00001) or 0.00001)
                    spread_val = s_pts * p_pt
                    sl_dist = abs(best_ob.entry - best_ob.sl)
                    if spread_val > 0 and sl_dist < (spread_val * 2.5):
                        logger.info(f"[OB Rescan] {symbol}: Rejet car SL trop serré par rapport au spread ({sl_dist:.5f} < 2.5x {spread_val:.5f})")
                        continue

                logger.info(f"[OB Rescan] {symbol}: {best_ob.ob_type} validé (score {best_ob.score:.0f}, HTF {htf_trend})")

                # Construire un 'decision' compatible avec le risk manager
                decision = {
                    "action": best_ob.direction,
                    "confidence": int(best_ob.score),
                    "entry": best_ob.entry,
                    "sl": best_ob.sl,
                    "tp": best_ob.tp,
                    "reason": f"ORDER BLOCK {best_ob.ob_type} (Score {best_ob.score:.0f}/100) | {best_ob.bars_ago} bougies | Impulsion {best_ob.impulse_strength:.1f}x ATR | R:R {best_ob.rr}",
                    "ob_type": best_ob.ob_type,
                    "ob_zone_low": best_ob.ob_low,
                    "ob_zone_high": best_ob.ob_high,
                }

                # Validation risk manager
                account_info = self.mt5.get_account_info()
                symbol_info = self.mt5.get_symbol_info(symbol)
                if not account_info or not symbol_info:
                    continue
                validation = self.risk.validate_proposal(decision, account_info, symbol_info)
                if not validation['valid']:
                    logger.warning(f"[OB Rescan] {symbol}: Risk validation failed — {validation['reason']}")
                    continue

                results.append({
                    "status": "OB_SIGNAL",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "decision": validation['proposal'],
                    "ob_summary": best_ob.summary(),
                })

            except Exception as _e:
                logger.warning(f"[OB Rescan] {symbol}: erreur — {_e}")

        return results

    def scan_for_pending_ob_orders(
        self,
        symbols: list,
        timeframe: str,
        direction_filter: str = "BOTH",
        stop_event=None,
    ) -> list:
        """
        Scans for high-quality Order Blocks that are NOT yet retested.
        These are candidates for PENDING (limit) orders.
        
        Returns list of dicts with status='PENDING_OB' containing:
        - symbol, timeframe, ob_data (OrderBlock), decision (validated proposal)
        """
        results = []
        for symbol in symbols:
            time.sleep(0.02) # Yield GIL
            if stop_event and stop_event.is_set():
                break
            try:
                if not self.mt5.is_market_open(symbol):
                    continue
                
                df = self.market_data.get_dataframe(symbol, timeframe)
                if df is None or df.empty:
                    continue
                df = self.tech.add_all_indicators(df)

                # HTF trend for alignment
                htf_map = {"M5": "M15", "M15": "H1", "H1": "H4", "H4": "D1", "D1": "D1"}
                htf = htf_map.get(timeframe.upper(), "H1")
                htf_trend = "Unknown"
                htf_df = self.market_data.get_dataframe(symbol, htf)
                if htf_df is not None and not htf_df.empty:
                    htf_df = self.tech.add_all_indicators(htf_df)
                    htf_feat = self.tech.extract_latest_features(htf_df)
                    htf_trend = htf_feat.get('trend', 'Unknown')

                # Detect PENDING (non-retested) OB zones
                pending_obs = self.ob_detector.detect_pending_zones(
                    df, symbol, timeframe,
                    htf_trend=htf_trend,
                    direction_filter=direction_filter,
                )

                if not pending_obs:
                    continue

                best_ob = pending_obs[0]
                is_cd, cd_reason = self.is_symbol_in_cooldown(symbol, best_ob.direction)
                if is_cd:
                    logger.info(f"[OB PENDING] {symbol}: Rejet {best_ob.direction} — Symbole en cooldown: {cd_reason}")
                    continue

                logger.info(
                    f"[OB PENDING] {symbol}: {best_ob.ob_type} zone détectée "
                    f"(score {best_ob.score:.0f}, entry={best_ob.entry:.5f})"
                )

                # Build a decision and validate with risk manager
                order_type_str = "BUY_LIMIT" if best_ob.direction.upper() == "BUY" else "SELL_LIMIT"
                decision = {
                    "action": best_ob.direction,
                    "confidence": int(best_ob.score),
                    "is_pending": True,
                    "order_type": order_type_str,
                    "expiration_bars": 12,
                    "entry": best_ob.entry,
                    "sl": best_ob.sl,
                    "tp": best_ob.tp,
                    "reason": (
                        f"PENDING ORDER BLOCK {best_ob.ob_type} "
                        f"(Score {best_ob.score:.0f}/100) | "
                        f"Zone [{best_ob.ob_low:.5f}-{best_ob.ob_high:.5f}] | "
                        f"Impulsion {best_ob.impulse_strength:.1f}x ATR | "
                        f"R:R {best_ob.rr}"
                    ),
                }

                account_info = self.mt5.get_account_info()
                symbol_info = self.mt5.get_symbol_info(symbol)
                if not account_info or not symbol_info:
                    continue

                validation = self.risk.validate_proposal(decision, account_info, symbol_info)
                if not validation['valid']:
                    logger.info(f"[OB PENDING] {symbol}: Risk validation failed — {validation['reason']}")
                    continue

                results.append({
                    "status": "PENDING_OB",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "decision": validation['proposal'],
                    "ob_data": best_ob,
                    "htf_trend": htf_trend,
                    "ob_summary": best_ob.summary(),
                })

            except Exception as _e:
                logger.warning(f"[OB PENDING] {symbol}: erreur — {_e}")

        return results

    def scan_for_crt_pending_orders(
        self,
        symbols: list,
        timeframe: str,
        direction_filter: str = "BOTH",
        stop_event=None,
    ) -> list:
        """
        Scanne les marchés à la recherche de configurations CRT + ICT/SMC complètes
        (Sweep + Réintégration + MSS + FVG) pour placement d'ordres limites PENDING.
        Conforme au Guide Universel.
        """
        results = []
        htf_map = {"M5": "M15", "M15": "H1", "H1": "H4", "H4": "D1", "D1": "D1"}
        htf = htf_map.get(timeframe.upper(), "H1")

        for symbol in symbols:
            time.sleep(0.02)  # Yield GIL
            if stop_event and stop_event.is_set():
                break
            try:
                if not self.mt5.is_market_open(symbol):
                    continue

                df_ltf = self.market_data.get_dataframe(symbol, timeframe)
                if df_ltf is None or df_ltf.empty or len(df_ltf) < 10:
                    continue
                df_ltf = self.tech.add_all_indicators(df_ltf)
                feat = self.tech.extract_latest_features(df_ltf)
                atr_ltf = float(feat.get('atr_14') or 0.0006)

                htf_df = self.market_data.get_dataframe(symbol, htf)
                if htf_df is None or len(htf_df) < 3:
                    continue

                sym_info = self.mt5.get_symbol_info(symbol) or {}
                point = float(sym_info.get('point', 0.00001))
                spread_val = float(sym_info.get('spread', 0)) * point

                raw_cfg = getattr(self, 'config', {}) or {}
                crt_cfg = raw_cfg.get('crt', {}) if isinstance(raw_cfg, dict) else {}
                order_mode = str(crt_cfg.get('order_mode', 'both')).lower()
                use_pending = crt_cfg.get('use_pending_orders', True) and order_mode != "market"
                allow_market = order_mode in ("both", "market") or not use_pending

                crt_setup = None
                if use_pending:
                    crt_setup = self.crt_detector.detect_smc_pending(
                        df_htf=htf_df,
                        df_ltf=df_ltf,
                        atr_ltf=atr_ltf,
                        spread=spread_val,
                        min_rr=float(crt_cfg.get('min_rr', 1.5)),
                        expiration_bars=int(crt_cfg.get('expiration_bars', 4)),
                    )

                if not crt_setup and allow_market:
                    crt_setup = self.crt_detector.detect(htf_df, atr=atr_ltf, spread=spread_val)

                if not crt_setup:
                    continue

                dir_filter_upper = str(direction_filter or 'BOTH').upper()
                if dir_filter_upper == 'BUY_ONLY' and crt_setup.direction != 'BUY':
                    continue
                elif dir_filter_upper == 'SELL_ONLY' and crt_setup.direction != 'SELL':
                    continue

                is_cd, cd_reason = self.is_symbol_in_cooldown(symbol, crt_setup.direction)
                if is_cd:
                    logger.info(f"[CRT SCAN] {symbol}: Rejet {crt_setup.direction} — Symbole en cooldown: {cd_reason}")
                    continue

                # ── Analyse Order Flow de Confluence (DOM, CVD, Volume Profile, Absorption) ──
                ticks_df = None
                dom_snapshot = None
                try:
                    ticks_df = self.market_data.get_ticks_dataframe(symbol, count=1000)
                    dom_snapshot = self.market_data.get_order_book_snapshot(symbol)
                except Exception as _ofe:
                    logger.debug(f"[OF] Erreur lecture ticks/DOM pour {symbol}: {_ofe}")

                of_confluence = self.order_flow_engine.evaluate_setup_confluence(
                    symbol=symbol,
                    crt_setup=crt_setup,
                    ticks_df=ticks_df,
                    dom_data=dom_snapshot,
                    candles_df=df_ltf,
                    atr=atr_ltf,
                )
                crt_setup.order_flow_confluence = of_confluence
                crt_setup.order_flow_confirmed = of_confluence.is_confirmed
                crt_setup.order_flow_score = of_confluence.confluence_score

                # Filtre strict optionnel si exigé par la configuration
                of_cfg = raw_cfg.get('order_flow', {}) if isinstance(raw_cfg, dict) else {}
                if of_cfg.get('require_confirmation', False) and not of_confluence.is_confirmed:
                    logger.info(f"[CRT SCAN] {symbol}: Rejet {crt_setup.direction} — Confirmation Order Flow requise mais score insuffisant ({of_confluence.confluence_score:.1f}/100)")
                    continue

                is_p = getattr(crt_setup, 'is_pending', False)
                if is_p:
                    entry_p = crt_setup.entry_price
                    sl_p = crt_setup.sl
                    tp_p = crt_setup.tp
                    o_type = crt_setup.order_type
                    status_tag = "PENDING_CRT"
                else:
                    entry_p = float(sym_info.get('ask') if crt_setup.direction == 'BUY' else sym_info.get('bid'))
                    sl_p = crt_setup.sl_raw
                    tp_p = crt_setup.tp2
                    o_type = "MARKET"
                    status_tag = "MARKET_CRT"

                base_reason = (
                    crt_setup.reason
                    if hasattr(crt_setup, 'reason') and crt_setup.reason
                    else f"CRT {crt_setup.direction} direct market ({o_type}) sur {htf}"
                )
                of_badge = f" [{of_confluence.summary_badge}]" if of_confluence.is_confirmed else ""
                reason_text = f"{base_reason}{of_badge}"

                # Calcul de la confiance avec bonus / malus Order Flow calibré selon la qualité de la donnée
                base_conf = 85 if crt_setup.in_killzone else 80
                dq = getattr(of_confluence, 'data_quality', 'ESTIMATED_QUOTE_RULE')
                if of_confluence.is_confirmed:
                    if dq == 'REAL_OBSERVED':
                        bonus = 10
                    elif dq == 'ESTIMATED_QUOTE_RULE':
                        bonus = 5
                    else:  # DEGRADED_STALE ou UNAVAILABLE
                        bonus = 0
                    conf = min(95, base_conf + bonus)
                elif dq == 'DEGRADED_STALE':
                    conf = max(65, base_conf - 5)
                elif of_confluence.confluence_score < 40:
                    conf = max(65, base_conf - 10)
                else:
                    conf = base_conf

                crt_risk_pct = float(crt_cfg.get('risk_per_trade_pct', getattr(self.risk, 'default_risk_percent', 1.0)))
                decision = {
                    "action": crt_setup.direction,
                    "confidence": conf,
                    "risk_percent": crt_risk_pct,
                    "entry": entry_p,
                    "sl": sl_p,
                    "tp": tp_p,
                    "tp1": crt_setup.tp1,
                    "tp2": crt_setup.tp2,
                    "rr": getattr(crt_setup, 'rr', 2.0),
                    "order_type": o_type,
                    "is_pending": is_p,
                    "expiration_bars": getattr(crt_setup, 'expiration_bars', 4),
                    "expiration_seconds": getattr(crt_setup, 'expiration_seconds', 3600),
                    "in_killzone": crt_setup.in_killzone,
                    "killzone_name": crt_setup.killzone_name,
                    "reason": reason_text,
                    "order_flow_confirmed": of_confluence.is_confirmed,
                    "order_flow_score": of_confluence.confluence_score,
                    "order_flow_details": of_confluence.details,
                    "order_flow_data_quality": dq,
                }

                account_info = self.mt5.get_account_info()
                if not account_info or not sym_info:
                    continue

                validation = self.risk.validate_proposal(decision, account_info, sym_info)
                if not validation['valid']:
                    logger.info(f"[CRT SCAN] {symbol}: Validation de risque échouée — {validation['reason']}")
                    continue

                # Enregistrement dans le Journal Order Flow pour traçabilité et métriques
                self.order_flow_journal.log_signal(
                    symbol=symbol,
                    direction=crt_setup.direction,
                    entry_price=entry_p,
                    strategy_name="CRT_ORDER_FLOW",
                    of_confluence=of_confluence,
                    status=status_tag,
                    notes=reason_text,
                )

                results.append({
                    "status": status_tag,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "decision": validation['proposal'],
                    "crt_data": crt_setup,
                    "order_flow": of_confluence,
                    "htf": htf,
                    "reason": reason_text,
                })
                logger.info(f"✅ [{status_tag}] {symbol}: Ordre {o_type} validé @ {entry_p} (OF Score: {of_confluence.confluence_score:.1f}/100)")

            except Exception as _e:
                logger.warning(f"[CRT SCAN] {symbol}: erreur — {_e}")

        return results

