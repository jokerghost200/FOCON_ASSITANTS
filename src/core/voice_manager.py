"""
Module de gestion vocale pour Focon Assistant (TTS & STT).
- Synthèse vocale (Text-To-Speech) via PySide6.QtTextToSpeech (offline, voix native Windows fr_FR).
- Reconnaissance vocale (Speech-To-Text) via PySide6.QtMultimedia (enregistrement micro)
  et Groq Whisper API (whisper-large-v3-turbo, ultra-rapide ~200ms).
"""

import os
import re
import tempfile
import logging
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

from PySide6.QtCore import QObject, Signal, QUrl, QThread
from PySide6.QtMultimedia import QMediaCaptureSession, QAudioInput, QMediaRecorder
from PySide6.QtTextToSpeech import QTextToSpeech

logger = logging.getLogger(__name__)

# Nettoyage regex pour que la voix lise naturellement le texte
RE_MARKDOWN_URL = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
RE_HTML_TAGS = re.compile(r'<[^>]+>')
RE_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
RE_INLINE_CODE = re.compile(r'`([^`]+)`')
RE_MD_SYMBOLS = re.compile(r'[*_~#>-]+')
RE_EMOJIS = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)


def clean_text_for_speech(text: str) -> str:
    """Nettoie le texte formaté (Markdown, HTML, émojis) pour la synthèse vocale."""
    if not text:
        return ""
    # Remplacer les blocs de code par une courte mention
    cleaned = RE_CODE_BLOCK.sub(" bloc de données ", text)
    # Remplacer les liens markdown par le texte du lien
    cleaned = RE_MARKDOWN_URL.sub(r'\1', cleaned)
    # Remplacer le code inline
    cleaned = RE_INLINE_CODE.sub(r'\1', cleaned)
    # Retirer les balises HTML (<br>, <b>, etc.)
    cleaned = RE_HTML_TAGS.sub(' ', cleaned)
    # Retirer les symboles markdown
    cleaned = RE_MD_SYMBOLS.sub(' ', cleaned)
    # Retirer les émojis qui peuvent perturber la prononciation
    cleaned = RE_EMOJIS.sub('', cleaned)
    # Réduire les espaces multiples
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class TranscriptionThread(QThread):
    """Thread d'arrière-plan pour la transcription audio (ne bloque jamais l'UI)."""
    transcription_done = Signal(str)
    transcription_error = Signal(str)

    def __init__(self, voice_manager, audio_path: str, parent=None):
        super().__init__(parent)
        self.vm = voice_manager
        self.audio_path = audio_path

    def run(self):
        try:
            import time
            last_size = -1
            stable_count = 0
            for _ in range(40):  # Jusqu'à 1.6s max, vérification rapide toutes les 40ms
                if os.path.exists(self.audio_path):
                    current_size = os.path.getsize(self.audio_path)
                    if current_size > 100:
                        if current_size == last_size:
                            stable_count += 1
                            if (current_size > 500 and stable_count >= 1) or stable_count >= 2:
                                break
                        else:
                            stable_count = 0
                        last_size = current_size
                time.sleep(0.04)

            time.sleep(0.04)

            text = self.vm.transcribe_audio(self.audio_path)
            self.transcription_done.emit(text)
        except Exception as e:
            logger.exception("Erreur lors de la transcription dans TranscriptionThread")
            self.transcription_error.emit(str(e))
        finally:
            # Nettoyage du fichier audio temporaire
            try:
                if self.audio_path and os.path.exists(self.audio_path):
                    os.remove(self.audio_path)
            except Exception:
                pass


class VoiceManager(QObject):
    """
    Gestionnaire central de la voix pour Focon Assistant.
    """
    speech_started = Signal()
    speech_finished = Signal()
    recording_started = Signal()
    recording_stopped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        # ── 1. Configuration de la Synthèse Vocale (TTS) ──
        self.is_muted = False
        self.tts: Optional[QTextToSpeech] = None
        self.voice_name: str = ""
        self.rate: float = 0.0
        self.pitch: float = 0.0
        self.volume: float = 1.0
        self.auto_adapt_on_danger: bool = False
        self.voice_announcements: bool = True
        self._init_tts()

        # ── 2. Configuration de l'Enregistreur Micro (STT) ──
        self._is_recording = False
        self._temp_audio_file: Optional[str] = None
        self.session: Optional[QMediaCaptureSession] = None
        self.audio_input: Optional[QAudioInput] = None
        self.recorder: Optional[QMediaRecorder] = None
        self._init_audio_recorder()

    def _init_tts(self):
        """Initialise QTextToSpeech avec la meilleure voix française disponible."""
        try:
            self.tts = QTextToSpeech(self)
            # Tenter de sélectionner la locale française si disponible
            locales = self.tts.availableLocales()
            for loc in locales:
                if loc.name().startswith("fr"):
                    self.tts.setLocale(loc)
                    logger.info("QTextToSpeech locale configurée sur : %s", loc.name())
                    break

            self.tts.setVolume(self.volume)
            self.tts.setRate(self.rate)
            self.tts.setPitch(self.pitch)
            self.tts.stateChanged.connect(self._on_tts_state_changed)
            logger.info("QTextToSpeech initialisé avec succès.")
        except Exception as e:
            logger.warning("QTextToSpeech non disponible sur cette machine : %s", e)
            self.tts = None

    def get_available_voices(self) -> list:
        """Retourne la liste des noms des voix disponibles sur le système."""
        if not self.tts:
            return []
        try:
            return [v.name() for v in self.tts.availableVoices()]
        except Exception:
            return []

    def get_current_voice_name(self) -> str:
        """Retourne le nom de la voix actuellement sélectionnée."""
        if not self.tts:
            return self.voice_name
        try:
            v = self.tts.voice()
            return v.name() if v else self.voice_name
        except Exception:
            return self.voice_name

    def set_voice_by_name(self, name: str) -> bool:
        """Sélectionne une voix par son nom."""
        if not self.tts or not name:
            return False
        try:
            for v in self.tts.availableVoices():
                if v.name() == name or name.lower() in v.name().lower():
                    self.tts.setVoice(v)
                    self.voice_name = v.name()
                    logger.info("Voix configurée sur : %s", self.voice_name)
                    return True
        except Exception as e:
            logger.warning("Impossible d'appliquer la voix %s : %s", name, e)
        return False

    def set_rate(self, rate: float):
        """Ajuste la vitesse de parole (-1.0 à 1.0, 0.0 = normale)."""
        self.rate = max(-1.0, min(1.0, float(rate)))
        if self.tts:
            try:
                self.tts.setRate(self.rate)
            except Exception as e:
                logger.warning("Erreur configuration vitesse vocale : %s", e)

    def set_pitch(self, pitch: float):
        """Ajuste la hauteur de ton (-1.0 à 1.0, 0.0 = standard)."""
        self.pitch = max(-1.0, min(1.0, float(pitch)))
        if self.tts:
            try:
                self.tts.setPitch(self.pitch)
            except Exception as e:
                logger.warning("Erreur configuration pitch vocal : %s", e)

    def set_volume(self, volume: float):
        """Ajuste le volume sonore (0.0 à 1.0)."""
        self.volume = max(0.0, min(1.0, float(volume)))
        if self.tts:
            try:
                self.tts.setVolume(self.volume)
            except Exception as e:
                logger.warning("Erreur configuration volume vocal : %s", e)

    def load_config(self, cfg: dict):
        """Charge les paramètres de voix depuis un dictionnaire de configuration."""
        if not cfg or not isinstance(cfg, dict):
            return
        if "voice_name" in cfg and cfg["voice_name"]:
            self.set_voice_by_name(str(cfg["voice_name"]))
        if "rate" in cfg:
            self.set_rate(float(cfg["rate"]))
        if "pitch" in cfg:
            self.set_pitch(float(cfg["pitch"]))
        if "volume" in cfg:
            self.set_volume(float(cfg["volume"]))
        if "auto_adapt_on_danger" in cfg:
            self.auto_adapt_on_danger = bool(cfg["auto_adapt_on_danger"])
        if "voice_announcements" in cfg:
            self.voice_announcements = bool(cfg["voice_announcements"])

    def get_config(self) -> dict:
        """Retourne la configuration actuelle sous forme de dictionnaire."""
        return {
            "enabled": not self.is_muted,
            "voice_name": self.get_current_voice_name(),
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "auto_adapt_on_danger": self.auto_adapt_on_danger,
            "voice_announcements": self.voice_announcements,
        }

    def _on_tts_state_changed(self, state):
        if state == QTextToSpeech.State.Speaking:
            self.speech_started.emit()
        else:
            self.speech_finished.emit()

    def speak(self, text: str):
        """Lit un texte à voix haute en français (si non muté)."""
        if self.is_muted or not self.tts:
            return

        cleaned = clean_text_for_speech(text)
        if not cleaned:
            return

        try:
            # Si le TTS était déjà en train de parler, on l'interrompt pour la nouvelle réponse
            if self.tts.state() == QTextToSpeech.State.Speaking:
                self.tts.stop()
            self.tts.say(cleaned)
        except Exception as e:
            logger.warning("Erreur lors de la lecture vocale : %s", e)

    def stop_speaking(self):
        """Arrête immédiatement la parole."""
        if self.tts and self.tts.state() == QTextToSpeech.State.Speaking:
            try:
                self.tts.stop()
            except Exception:
                pass

    def toggle_mute(self) -> bool:
        """Bascule entre Muet et Actif. Retourne True si maintenant muet."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.stop_speaking()
        return self.is_muted

    def _init_audio_recorder(self):
        """Configure la capture audio QtMultimedia."""
        try:
            self.session = QMediaCaptureSession(self)
            self.audio_input = QAudioInput(self)
            try:
                self.audio_input.setVolume(1.0)
            except Exception:
                pass
            self.recorder = QMediaRecorder(self)
            self.session.setAudioInput(self.audio_input)
            self.session.setRecorder(self.recorder)
            logger.info("Enregistreur audio Qt initialisé avec succès (volume 100%).")
        except Exception as e:
            logger.warning("Enregistreur audio Qt non disponible : %s", e)
            self.session = None

    def start_recording(self) -> bool:
        """Démarre l'enregistrement du microphone."""
        if self._is_recording or not self.recorder:
            return False

        try:
            import time
            temp_dir = tempfile.gettempdir()
            ts = int(time.time() * 1000)
            self._temp_audio_file = os.path.join(temp_dir, f"focon_voice_{os.getpid()}_{ts}.m4a")

            self.recorder.setOutputLocation(QUrl.fromLocalFile(self._temp_audio_file))
            self.recorder.record()
            self._is_recording = True
            self.recording_started.emit()
            logger.info("Enregistrement micro démarré dans %s", self._temp_audio_file)
            return True
        except Exception as e:
            logger.exception("Échec du démarrage de l'enregistrement micro : %s", e)
            self._is_recording = False
            return False

    def stop_recording(self) -> Optional[str]:
        """Arrête l'enregistrement et retourne le chemin du fichier enregistré."""
        if not self._is_recording or not self.recorder:
            return None

        try:
            self.recorder.stop()
            self._is_recording = False
            
            # Récupérer l'emplacement réel du fichier ou le fallback temporaire
            audio_path = self._temp_audio_file
            try:
                actual_loc = self.recorder.actualLocation()
                if actual_loc and not actual_loc.isEmpty():
                    loc_path = actual_loc.toLocalFile()
                    if loc_path and os.path.exists(loc_path):
                        audio_path = loc_path
            except Exception:
                pass

            self.recording_stopped.emit(audio_path or "")
            logger.info("Enregistrement micro terminé : %s", audio_path)
            return audio_path
        except Exception as e:
            logger.exception("Échec de l'arrêt de l'enregistrement micro : %s", e)
            self._is_recording = False
            return None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcrit un fichier audio en texte.
        1. Utilise prioritairement Groq Whisper API (whisper-large-v3-turbo, ~200ms).
        2. Si la clé est absente ou si Groq échoue, tente un fallback SpeechRecognition (Google STT) si installé.
        """
        if not audio_path or not os.path.exists(audio_path):
            return ""

        file_size = os.path.getsize(audio_path)
        if file_size < 100:  # Enregistrement vide ou trop court
            logger.warning("Fichier audio trop court ou vide (%d octets)", file_size)
            return ""

        def _fallback_speech_recognition() -> str:
            try:
                import importlib
                sr = importlib.import_module("speech_recognition")
                recognizer = getattr(sr, "Recognizer")()
                audio_file_cls = getattr(sr, "AudioFile")
                with audio_file_cls(audio_path) as source:
                    audio_data = recognizer.record(source)
                res = recognizer.recognize_google(audio_data, language="fr-FR")
                logger.info("Transcription réussie via SpeechRecognition (Google) : '%s'", res)
                return str(res).strip()
            except (ImportError, ModuleNotFoundError):
                logger.debug("SpeechRecognition n'est pas installé.")
                return ""
            except Exception as e_sr:
                logger.warning("Fallback SpeechRecognition a échoué : %s", e_sr)
                return ""

        # Déterminer le type MIME selon l'extension
        ext = os.path.splitext(audio_path)[1].lower().replace(".", "")
        mime_map = {
            "m4a": "audio/m4a",
            "mp4": "audio/mp4",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "aac": "audio/aac",
        }
        mime_type = mime_map.get(ext, "audio/m4a")

        # 1. Tentative avec Groq Whisper si clé disponible
        key = self.groq_api_key or os.getenv("GROQ_API_KEY", "").strip()
        if key:
            try:
                filename = os.path.basename(audio_path)
                with open(audio_path, "rb") as f:
                    file_bytes = f.read()

                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {key}"}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "language": "fr",
                    "response_format": "json",
                    "temperature": "0.0",
                    "prompt": "Trading, Forex, EURUSD, GBPUSD, Ordre, Autopilot, Risque, Position, Achat, Vente, Solde, Marché.",
                }
                files = {
                    "file": (filename, file_bytes, mime_type),
                }

                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers, data=data, files=files)
                    resp.raise_for_status()
                    res_json = resp.json()
                    transcription = str(res_json.get("text", "")).strip()
                    # Nettoyer les faux positifs de silence Whisper ("...", ".", etc.)
                    if transcription in ("...", "..", ".", "!", "?", "Merci.", "Merci d'avoir regardé.", "Sous-titres réalisés par la communauté d'Amara.org"):
                        transcription = ""
                    logger.info("Transcription Groq Whisper réussie (%d car.) : '%s'", len(transcription), transcription)
                    return transcription
            except Exception as e_groq:
                logger.warning("Groq Whisper a échoué (%s), tentative fallback...", e_groq)
                fallback_res = _fallback_speech_recognition()
                if fallback_res:
                    return fallback_res
                raise RuntimeError(f"Erreur de transcription audio : {e_groq}")
        else:
            # Clé Groq absente -> fallback direct
            fallback_res = _fallback_speech_recognition()
            if fallback_res:
                return fallback_res
            raise ValueError("Clé API Groq manquante dans .env (GROQ_API_KEY) et SpeechRecognition non disponible.")

