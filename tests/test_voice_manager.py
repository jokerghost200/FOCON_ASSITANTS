import unittest
from unittest.mock import patch, MagicMock

from src.core.voice_manager import clean_text_for_speech, VoiceManager


class TestVoiceManager(unittest.TestCase):
    def test_clean_text_for_speech_markdown_and_emojis(self):
        raw = "🚀 **Achat EURUSD** validé ! Objectif: 1.0950 (`R:R` = 2.0). [Détails](http://example.com) <br>"
        cleaned = clean_text_for_speech(raw)
        self.assertNotIn("🚀", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("<br>", cleaned)
        self.assertNotIn("http://example.com", cleaned)
        self.assertIn("Achat EURUSD", cleaned)
        self.assertIn("Détails", cleaned)

    def test_clean_text_for_speech_code_block(self):
        raw = "Voici le plan :\n```json\n{\"action\": \"BUY\"}\n```\nFin du plan."
        cleaned = clean_text_for_speech(raw)
        self.assertNotIn("{\"action\": \"BUY\"}", cleaned)
        self.assertIn("bloc de données", cleaned)
        self.assertIn("Fin du plan", cleaned)

    def test_clean_text_empty_and_spaces(self):
        self.assertEqual(clean_text_for_speech(""), "")
        self.assertEqual(clean_text_for_speech("   "), "")
        self.assertEqual(clean_text_for_speech("mot1    mot2"), "mot1 mot2")

    @patch("src.core.voice_manager.QTextToSpeech")
    @patch("src.core.voice_manager.QMediaCaptureSession")
    @patch("src.core.voice_manager.QAudioInput")
    @patch("src.core.voice_manager.QMediaRecorder")
    def test_voice_manager_toggle_mute(self, mock_rec, mock_inp, mock_sess, mock_tts):
        vm = VoiceManager()
        self.assertFalse(vm.is_muted)

        # Toggle mute ON
        now_muted = vm.toggle_mute()
        self.assertTrue(now_muted)
        self.assertTrue(vm.is_muted)

        # Toggle mute OFF
        now_muted = vm.toggle_mute()
        self.assertFalse(now_muted)
        self.assertFalse(vm.is_muted)

    @patch("src.core.voice_manager.httpx.Client")
    def test_transcribe_audio_groq_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Bonjour robot, analyse le compte"}
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        vm = VoiceManager()
        vm.groq_api_key = "gsk_test123"

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000), \
             patch("builtins.open", unittest.mock.mock_open(read_data=b"fake_audio")):
            text = vm.transcribe_audio("dummy.m4a")

        self.assertEqual(text, "Bonjour robot, analyse le compte")


if __name__ == "__main__":
    unittest.main()
