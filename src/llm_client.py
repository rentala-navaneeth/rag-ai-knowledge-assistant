import requests
import json
from config import LLM_API_URL
from src.logger import logger


class LLMClient:
    def __init__(self):
        self.url = LLM_API_URL

    def generate(self, query, context, temperature=0.3):
        payload = {
            "query": query,
            "context": context,
            "temperature": temperature
        }

        # 🔥 REQUIRED for ngrok (prevents HTML warning page)
        headers = {
            "ngrok-skip-browser-warning": "true"
        }

        logger.info(f"Calling LLM API: {self.url}")
        logger.info(f"Query: {query}")

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=30
            )

            # Check HTTP status
            if response.status_code != 200:
                logger.error(f"LLM API returned status {response.status_code}")
                return {
                    "answer": "LLM API error",
                    "confidence": "low"
                }

            # Try parsing JSON response
            try:
                data = response.json()
            except Exception:
                logger.error("Response is not valid JSON (likely ngrok page or API down)")
                return {
                    "answer": "Invalid response from LLM",
                    "confidence": "low"
                }

            raw_output = data.get("output", "")

            # Parse structured output
            return self.parse_output(raw_output)

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return {
                "answer": "LLM timeout",
                "confidence": "low"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return {
                "answer": "Error contacting LLM",
                "confidence": "low"
            }

    def parse_output(self, text):
        """
        Parses LLM output which should be JSON:
        {
          "answer": "...",
          "confidence": "low/medium/high"
        }
        """
        try:
            return json.loads(text)

        except Exception:
            logger.warning("Failed to parse structured JSON output")

            return {
                "answer": text if text else "Unstructured response from LLM",
                "confidence": "low"
            }