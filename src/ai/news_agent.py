import logging
import json
from src.ai.gemini_agent import GeminiAgent

logger = logging.getLogger(__name__)

class NewsAgent:
    def __init__(self, gemini_agent: GeminiAgent):
        self.gemini_agent = gemini_agent

    def analyze_news(self, symbol: str, stop_event=None) -> dict:
        """
        Fetches and analyzes recent news for a given symbol (e.g. EURUSD).
        Returns a structured assessment of the news impact.
        """
        logger.info(f"Fetching news for {symbol} via Gemini + Google Search...")
        
        prompt = f"""
        Act as an expert financial news analyst.
        Search for the most recent and impactful news events today related to {symbol} (e.g., central bank decisions, economic data, geopolitics).
        Return ONLY a valid JSON object matching this schema:
        {{
            "market_impact": "HIGH" | "MEDIUM" | "LOW" | "NEUTRAL",
            "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
            "summary": "Brief summary of the most impactful news",
            "events": [
                {{
                    "event": "Event name",
                    "impact": "HIGH" | "MEDIUM" | "LOW",
                    "source": "Source if available"
                }}
            ]
        }}
        """
        
        # We don't use response_schema here because Google Search tool is not fully supported 
        # with Structured Outputs yet in some API versions. We'll ask it to format as JSON.
        
        try:
            response = self.gemini_agent.ask(prompt, use_search=True, stop_event=stop_event)
            if not response:
                return {"market_impact": "NEUTRAL", "direction": "NEUTRAL", "summary": "Failed to fetch news."}
                
            text = response.get("text", "")
            
            # Clean up markdown JSON block if present
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            data = json.loads(text.strip())
            return data
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse news JSON. Raw response: {text}")
            return {"market_impact": "NEUTRAL", "direction": "NEUTRAL", "summary": "Error parsing news."}
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {"market_impact": "NEUTRAL", "direction": "NEUTRAL", "summary": "Analysis failed."}
