"""Pytest config — sets dummy Gemini key so imports don't crash."""
import os
os.environ.setdefault("GEMINI_API_KEY", "AIza-test-key-for-unit-tests")
