"""
run_rag3_key2.py
================
Runs RAG3 evaluate routing all Groq calls through KEY2.
Must set GROQ_API_KEY before any imports touch the Groq client.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_key6 = os.getenv("GROQ_API_KEY6")
if not _key6:
    print("ERROR: GROQ_API_KEY6 not found in .env")
    sys.exit(1)

# Override before GroqLLM.__init__ reads GROQ_API_KEY
os.environ["GROQ_API_KEY"] = _key6

import yaml
from loguru import logger

with open("config/config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from src.pipelines.rag3_curation_only import RAG3Pipeline
RAG3Pipeline(config).evaluate()
