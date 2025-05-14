#!/usr/bin/env python3

# test_red_flags.py — Unit tests for safety phrase detection

from src.red_flags import check_red_flags

def test_detects_suicide_phrase():
    text = "I want to kill myself tonight."
    assert "suicide" in check_red_flags(text)

def test_detects_self_harm_phrase():
    text = "Sometimes I cut myself when I’m upset."
    assert "self_harm" in check_red_flags(text)

def test_no_red_flags_in_neutral_text():
    text = "I had a tough week but I’m managing."
    assert check_red_flags(text) == []
