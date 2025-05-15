#!/usr/bin/env python3

# config.py — Centralized config for model backend

import os
from dotenv import load_dotenv

load_dotenv()

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
