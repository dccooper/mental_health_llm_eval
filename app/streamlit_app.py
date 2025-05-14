#!/usr/bin/env python3

# streamlit_app.py — Human-in-the-loop review interface
#
# This app provides a browser-based interface for reviewing LLM responses
# to mental health prompts. It allows human evaluators to adjust scores,
# write feedback, and export results. Useful for quality control and red-teaming.

import streamlit as st
import pandas as pd
import json

from src.evaluator import load_prompts
from src.models import query_model, generate_justification
from src.red_flags import check_red_flags
from src.scorer import load_rubric, apply_rubric

st.set_page_config(page_title="Mental Health LLM Eval", layout="wide")
st.title("🧠 Mental Health LLM Evaluator")

# Step 1: Upload config files
prompt_file = st.file_uploader("📄 Upload Prompt Bank YAML", type=["yaml"])
rubric_file = st.file_uploader("📄 Upload Scoring Rubric YAML", type=["yaml"])

# Step 2: Load files and setup interface
if prompt_file and rubric_file:
    prompts = load_prompts(prompt_file)
    rubric = load_rubric(rubric_file)

    selected_prompt = st.selectbox("🧾 Choose a prompt:", [p['prompt'] for p in prompts])

    if selected_prompt:
        model_response = query_model(selected_prompt)  # Stubbed response
        red_flags = check_red_flags(model_response)
        auto_scores = apply_rubric(model_response, red_flags, rubric)

        st.subheader("🗣️ Model Response")
        st.text_area("Model Output", model_response, height=120)

        st.subheader("🚩 Red Flags")
        st.markdown(", ".join(red_flags) if red_flags else "_None detected_")

        st.subheader("📊 Manual Scoring")
        manual_scores = {}
        for criterion in rubric['scoring_rubric']['criteria']:
            name = criterion['name']
            desc = criterion['description']
            manual_scores[name] = st.slider(f"{name.capitalize()} — {desc}", 1, 5, value=auto_scores.get(name, 3))

        st.subheader("🧠 Justification")
        justification = generate_justification(selected_prompt, model_response, manual_scores)
        justification_input = st.text_area("Edit Justification", value=justification, height=140)

        feedback = st.text_area("📝 Additional Feedback", placeholder="Optional comments...")

        if "evaluations" not in st.session_state:
            st.session_state.evaluations = []

        if st.button("✅ Submit Evaluation"):
            result = {
                "prompt": selected_prompt,
                "response": model_response,
                "red_flags": red_flags,
                "scores": manual_scores,
                "justification": justification_input,
                "feedback": feedback
            }
            st.session_state.evaluations.append(result)
            st.success("✅ Evaluation submitted.")
            st.json(result)

        if st.session_state.evaluations:
            st.subheader("⬇️ Download All Evaluations")
            df = pd.json_normalize(st.session_state.evaluations)
            csv_data = df.to_csv(index=False).encode("utf-8")
            json_data = json.dumps(st.session_state.evaluations, indent=2).encode("utf-8")

            st.download_button("📥 Download CSV", csv_data, file_name="evaluations.csv", mime="text/csv")
            st.download_button("📥 Download JSON", json_data, file_name="evaluations.json", mime="application/json")
