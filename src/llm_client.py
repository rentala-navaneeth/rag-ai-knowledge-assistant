import requests
import json
from config import LLM_API_URL
from src.logger import logger


class LLMClient:
    def __init__(self):
        self.url = LLM_API_URL

    def generate(self, query, context):
        payload = {
            "query": query,
            "context": context
        }

        logger.info(f"Calling LLM API for query: {query}")

        try:
            response = requests.post(self.url, json=payload, timeout=30)

            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code}")
                return {
                    "answer": "LLM API error",
                    "confidence": "low"
                }

            data = response.json()
            raw_output = data.get("output", "")

            return self.parse_output(raw_output)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "answer": "Error contacting LLM",
                "confidence": "low"
            }

    def parse_output(self, text):
        try:
            return json.loads(text)
        except:
            logger.warning("Failed to parse JSON output")
            return {
                "answer": text,
                "confidence": "low"
            }