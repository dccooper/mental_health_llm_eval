#!/usr/bin/env python3

# streamlit_app.py — Human-in-the-loop review interface
#
# This app provides a browser-based interface for reviewing LLM responses
# to mental health prompts. It allows human evaluators to adjust scores,
# write feedback, and export results. Useful for quality control and red-teaming.

import streamlit as st
import pandas as pd
import json
from typing import Dict, List
import plotly.express as px
from datetime import datetime

from src.evaluator import (
    PromptEntry,
    EvaluationResult,
    load_prompts,
    evaluate_response,
    generate_report
)
from src.models import query_model
from src.red_flags import check_red_flags, get_severity

# Custom color palette
COLORS = {
    'burgundy': '#780000',
    'bright_red': '#c1121f',
    'cream': '#fdf0d5',
    'dark_blue': '#003049',
    'light_blue': '#669bbc'
}

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {COLORS['cream']};
    }}
    .stButton > button {{
        background-color: {COLORS['dark_blue']};
        color: white;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['light_blue']};
    }}
    .stTextInput > div > div > input {{
        background-color: white;
    }}
    .stTextArea > div > div > textarea {{
        background-color: white;
    }}
    .stSelectbox > div > div > select {{
        background-color: white;
    }}
    h1, h2, h3 {{
        color: {COLORS['dark_blue']};
    }}
    .stAlert {{
        background-color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Mental Health LLM Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None
if "auto_evaluation" not in st.session_state:
    st.session_state.auto_evaluation = None

# Sidebar for configuration and navigation
with st.sidebar:
    st.title("🧠 Configuration")
    
    # File upload
    prompt_file = st.file_uploader("📄 Upload Prompt Bank", type=["yaml"])
    
    if prompt_file:
        prompts = load_prompts(prompt_file)
        categories = sorted(set(p.category for p in prompts))
        
        # Category filter
        selected_category = st.selectbox(
            "🏷️ Filter by Category",
            ["All"] + categories
        )
        
        # Prompt selection
        filtered_prompts = [
            p for p in prompts
            if selected_category == "All" or p.category == selected_category
        ]
        
        selected_prompt = st.selectbox(
            "📝 Select Prompt",
            filtered_prompts,
            format_func=lambda p: f"{p.id}: {p.prompt[:50]}..."
        )
        
        if selected_prompt != st.session_state.current_prompt:
            st.session_state.current_prompt = selected_prompt
            # Get model response and evaluation
            model_response = query_model(selected_prompt.prompt)
            st.session_state.auto_evaluation = evaluate_response(
                selected_prompt, model_response
            )
    
    # Stats summary
    if st.session_state.evaluations:
        st.divider()
        st.subheader("📊 Session Stats")
        st.metric("Prompts Evaluated", len(st.session_state.evaluations))
        
        avg_score = sum(e.total_score for e in st.session_state.evaluations) / len(st.session_state.evaluations)
        st.metric("Average Score", f"{avg_score:.2f}")
        
        critical_count = sum(
            1 for e in st.session_state.evaluations
            if any(f for f in e.detected_red_flags if get_severity([f]) == "critical")
        )
        st.metric("Critical Issues", critical_count, delta=-critical_count, delta_color="inverse")

# Main content
st.title("Mental Health LLM Evaluator")

if not prompt_file:
    st.info("👈 Please upload a prompt bank YAML file to begin evaluation.")
    st.stop()

if not st.session_state.current_prompt or not st.session_state.auto_evaluation:
    st.warning("⚠️ Please select a prompt to evaluate.")
    st.stop()

# Display current prompt context
prompt = st.session_state.current_prompt
eval_result = st.session_state.auto_evaluation

st.header("Prompt Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Context")
    st.info(prompt.context)
    
    st.subheader("Expected Behaviors")
    for behavior in prompt.expected_behaviors:
        st.checkbox(behavior, value=behavior in eval_result.meets_expected_behaviors, disabled=True)

with col2:
    st.subheader("Red Flags to Watch")
    for flag in prompt.red_flags:
        st.checkbox(flag, value=flag in eval_result.detected_red_flags, disabled=True)

# Model response and evaluation
st.header("Model Response")
response_col, flags_col = st.columns([2, 1])

with response_col:
    st.text_area(
        "Response Text",
        eval_result.model_response,
        height=200,
        disabled=True
    )

with flags_col:
    st.subheader("🚩 Detected Issues")
    if eval_result.detected_red_flags:
        for flag in eval_result.detected_red_flags:
            severity = get_severity([flag])
            color = "red" if severity == "critical" else "orange"
            st.markdown(f":{color}[⚠️ {flag}]")
    else:
        st.success("No red flags detected")

# Scoring interface
st.header("Evaluation")
score_cols = st.columns(len(eval_result.scores))

manual_scores = {}
for col, (dimension, score) in zip(score_cols, eval_result.scores.items()):
    with col:
        st.subheader(dimension.capitalize())
        manual_scores[dimension] = st.slider(
            f"{dimension} score",
            min_value=0.0,
            max_value=1.0,
            value=float(score),
            step=0.1,
            help=f"Adjust the {dimension} score if needed"
        )

# Justification and feedback
st.header("Review")
justification = st.text_area(
    "Justification",
    eval_result.justification,
    height=100,
    help="Edit the automated justification if needed"
)

feedback = st.text_area(
    "Additional Feedback",
    "",
    height=100,
    placeholder="Add any additional observations, concerns, or suggestions..."
)

# Submit evaluation
if st.button("✅ Submit Evaluation", type="primary"):
    # Create modified evaluation result
    modified_result = EvaluationResult(
        prompt_entry=eval_result.prompt_entry,
        model_response=eval_result.model_response,
        detected_red_flags=eval_result.detected_red_flags,
        scores=manual_scores,
        dimension_scores=eval_result.dimension_scores,
        justification=justification,
        meets_expected_behaviors=eval_result.meets_expected_behaviors,
        total_score=sum(score * weight for score, weight in zip(
            manual_scores.values(),
            [0.3, 0.25, 0.2, 0.15, 0.1]  # weights for safety, clinical, empathy, ethics, cultural
        ))
    )
    
    st.session_state.evaluations.append(modified_result)
    st.success("✅ Evaluation submitted successfully!")

# Update the plotly theme to match our color scheme
def create_score_plot(df):
    fig = px.box(
        df,
        x="Dimension",
        y="Score",
        title="Score Distribution by Dimension",
        points="all"
    )
    fig.update_layout(
        plot_bgcolor=COLORS['cream'],
        paper_bgcolor=COLORS['cream'],
        font_color=COLORS['dark_blue'],
        title_font_color=COLORS['dark_blue']
    )
    fig.update_traces(
        marker_color=COLORS['light_blue'],
        boxpoints='all'
    )
    return fig

# Results visualization
if st.session_state.evaluations:
    st.header("Session Results")
    
    # Prepare data for visualization
    eval_data = []
    for eval_result in st.session_state.evaluations:
        for dimension, score in eval_result.scores.items():
            eval_data.append({
                "Prompt ID": eval_result.prompt_entry.id,
                "Dimension": dimension.capitalize(),
                "Score": score
            })
    
    df = pd.DataFrame(eval_data)
    
    # Plot scores using our custom function
    fig = create_score_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as CSV
        df_export = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "prompt_id": e.prompt_entry.id,
            "category": e.prompt_entry.category,
            "prompt": e.prompt_entry.prompt,
            "response": e.model_response,
            "total_score": e.total_score,
            **e.scores,
            "red_flags": ",".join(e.detected_red_flags),
            "justification": e.justification,
            "feedback": feedback
        } for e in st.session_state.evaluations])
        
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            "mental_health_llm_evaluations.csv",
            "text/csv"
        )
    
    with col2:
        # Export as JSON
        json_data = json.dumps([{
            "timestamp": datetime.now().isoformat(),
            "prompt": asdict(e.prompt_entry),
            "evaluation": {
                "response": e.model_response,
                "scores": e.scores,
                "dimension_scores": e.dimension_scores,
                "red_flags": e.detected_red_flags,
                "justification": e.justification,
                "feedback": feedback,
                "total_score": e.total_score
            }
        } for e in st.session_state.evaluations], indent=2).encode("utf-8")
        
        st.download_button(
            "📥 Download JSON",
            json_data,
            "mental_health_llm_evaluations.json",
            "application/json"
        )
