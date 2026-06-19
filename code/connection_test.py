"""
code/connection_test.py — OpenAI API Connectivity & Edge-Case Validation Suite
"""
import os
import sys
import json
import logging
from openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

class PingSchema(BaseModel):
    status: str
    message: str

def main():
    log.info("Starting OpenAI Infrastructure Check...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("CRITICAL: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)
    log.info("Stage 1: OPENAI_API_KEY environment detected successfully.")

    # 2. Client Initialization
    try:
        client = OpenAI(api_key=api_key)
        log.info("Stage 2: OpenAI client initialized.")
    except Exception as e:
        log.error(f"Stage 2 Failed: Client creation error: {e}")
        sys.exit(1)

    # 3. Text Structured Ping Check
    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input="Ping check. Respond with status='OK' and message='Success' in strict JSON format.",
            temperature=0.0,
            max_output_tokens=50,
            store=False,
        )
        data = json.loads(response.output_text or "")
        if data.get("status") != "OK" or data.get("message") != "Success":
            raise ValueError(f"Unexpected ping response: {data}")
        log.info(f"Stage 3: Text structured verification successful. Response: {data}")
    except Exception as e:
        log.error(f"Stage 3 Failed: Structured text generation error: {e}")
        sys.exit(1)

    log.info("✅ All core connection tests passed cleanly. System ready for main production runs.")

if __name__ == "__main__":
    main()