#!/usr/bin/env python3

"""
Mental Health LLM Evaluator UI
=============================

This module provides a Streamlit-based web interface for:
1. Loading and managing prompt banks
2. Running real-time LLM evaluations
3. Manual review and score adjustment
4. Results visualization and export
5. Session statistics tracking

The interface is designed for mental health professionals and researchers
to evaluate LLM responses for safety, clinical appropriateness, and empathy.
"""

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

# Custom color palette for consistent UI theming
COLORS = {
    'burgundy': '#780000',  # Used for critical alerts
    'bright_red': '#c1121f',  # Used for warnings
    'cream': '#fdf0d5',    # Primary background
    'dark_blue': '#003049', # Primary text and headers
    'light_blue': '#669bbc' # Interactive elements
}

# Apply custom styling
st.markdown(f"""
    <style>
    /* Main app styling */
    .stApp {{
        background-color: {COLORS['cream']};
    }}
    
    /* Interactive elements */
    .stButton > button {{
        background-color: {COLORS['dark_blue']};
        color: white;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['light_blue']};
    }}
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        background-color: white;
    }}
    
    /* Typography */
    h1, h2, h3 {{
        color: {COLORS['dark_blue']};
    }}
    
    /* Alerts and notifications */
    .stAlert {{
        background-color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Mental Health LLM Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []  # Stores all completed evaluations
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None  # Currently selected prompt
if "auto_evaluation" not in st.session_state:
    st.session_state.auto_evaluation = None  # Current automated evaluation result

# Sidebar configuration
with st.sidebar:
    st.title("🧠 Configuration")
    
    # Prompt bank management
    prompt_file = st.file_uploader(
        "📄 Upload Prompt Bank",
        type=["yaml"],
        help="Upload a YAML file containing evaluation prompts"
    )
    
    if prompt_file:
        # Load and organize prompts
        loaded_prompts = load_prompts(prompt_file)
        available_categories = sorted(set(p.category for p in loaded_prompts))
        
        # Category filtering
        selected_category = st.selectbox(
            "🏷️ Filter by Category",
            ["All"] + available_categories,
            help="Filter prompts by their category"
        )
        
        # Prompt selection
        filtered_prompts = [
            p for p in loaded_prompts
            if selected_category == "All" or p.category == selected_category
        ]
        
        selected_prompt = st.selectbox(
            "📝 Select Prompt",
            filtered_prompts,
            format_func=lambda p: f"{p.id}: {p.prompt[:50]}...",
            help="Choose a prompt to evaluate"
        )
        
        # Update current evaluation state
        if selected_prompt != st.session_state.current_prompt:
            st.session_state.current_prompt = selected_prompt
            # Generate and evaluate model response
            model_response = query_model(selected_prompt.prompt)
            st.session_state.auto_evaluation = evaluate_response(
                selected_prompt, model_response
            )
    
    # Session statistics display
    if st.session_state.evaluations:
        st.divider()
        st.subheader("📊 Session Stats")
        
        # Calculate key metrics
        total_evaluations = len(st.session_state.evaluations)
        average_score = sum(e.total_score for e in st.session_state.evaluations) / total_evaluations
        critical_issues = sum(
            1 for e in st.session_state.evaluations
            if any(f for f in e.detected_red_flags if get_severity([f]) == "critical")
        )
        
        # Display metrics
        st.metric("Prompts Evaluated", total_evaluations)
        st.metric("Average Score", f"{average_score:.2f}")
        st.metric(
            "Critical Issues",
            critical_issues,
            delta=-critical_issues,
            delta_color="inverse",
            help="Number of responses with critical safety concerns"
        )

# Main content area
st.title("Mental Health LLM Evaluator")

if not prompt_file:
    st.info("👈 Please upload a prompt bank YAML file to begin evaluation.")
    st.stop()

if not st.session_state.current_prompt or not st.session_state.auto_evaluation:
    st.warning("⚠️ Please select a prompt to evaluate.")
    st.stop()

# Display current evaluation context
current_prompt = st.session_state.current_prompt
current_evaluation = st.session_state.auto_evaluation

st.header("Prompt Analysis")
context_col, behavior_col = st.columns(2)

with context_col:
    st.subheader("Context")
    st.info(current_prompt.context)
    
    st.subheader("Expected Behaviors")
    for behavior in current_prompt.expected_behaviors:
        st.checkbox(
            behavior,
            value=behavior in current_evaluation.meets_expected_behaviors,
            disabled=True,
            help="Behaviors the model response should exhibit"
        )

with behavior_col:
    st.subheader("Red Flags to Watch")
    for flag in current_prompt.red_flags:
        st.checkbox(
            flag,
            value=flag in current_evaluation.detected_red_flags,
            disabled=True,
            help="Potential safety concerns to monitor"
        )

# Model response analysis
st.header("Model Response")
response_col, flags_col = st.columns([2, 1])

with response_col:
    st.text_area(
        "Response Text",
        current_evaluation.model_response,
        height=200,
        disabled=True,
        help="The LLM's response to the prompt"
    )

with flags_col:
    st.subheader("🚩 Detected Issues")
    if current_evaluation.detected_red_flags:
        for flag in current_evaluation.detected_red_flags:
            severity = get_severity([flag])
            flag_color = "red" if severity == "critical" else "orange"
            st.markdown(f":{flag_color}[⚠️ {flag}]")
    else:
        st.success("No red flags detected")

# Scoring interface
st.header("Evaluation")
score_cols = st.columns(len(current_evaluation.scores))

manual_scores = {}
for col, (dimension, score) in zip(score_cols, current_evaluation.scores.items()):
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

# Review and feedback
st.header("Review")
justification = st.text_area(
    "Justification",
    current_evaluation.justification,
    height=100,
    help="Explanation for the evaluation scores"
)

feedback = st.text_area(
    "Additional Feedback",
    "",
    height=100,
    placeholder="Add any additional observations, concerns, or suggestions..."
)

# Submit evaluation
if st.button("✅ Submit Evaluation", type="primary"):
    # Create modified evaluation result with manual adjustments
    modified_evaluation = EvaluationResult(
        prompt_entry=current_evaluation.prompt_entry,
        model_response=current_evaluation.model_response,
        detected_red_flags=current_evaluation.detected_red_flags,
        scores=manual_scores,
        dimension_scores=current_evaluation.dimension_scores,
        justification=justification,
        meets_expected_behaviors=current_evaluation.meets_expected_behaviors,
        total_score=sum(
            score * weight for score, weight in zip(
                manual_scores.values(),
                [0.3, 0.25, 0.2, 0.15, 0.1]  # Dimension weights
            )
        )
    )
    
    st.session_state.evaluations.append(modified_evaluation)
    st.success("✅ Evaluation submitted successfully!")

# Results visualization
if st.session_state.evaluations:
    st.header("Session Results")
    
    # Prepare visualization data
    evaluation_data = []
    for eval_result in st.session_state.evaluations:
        for dimension, score in eval_result.scores.items():
            evaluation_data.append({
                "Prompt ID": eval_result.prompt_entry.id,
                "Dimension": dimension.capitalize(),
                "Score": score
            })
    
    results_df = pd.DataFrame(evaluation_data)
    
    # Create and display score distribution plot
    fig = create_score_plot(results_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("Export Results")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Prepare CSV export
        export_df = pd.DataFrame([{
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
        
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv_data,
            "mental_health_llm_evaluations.csv",
            "text/csv",
            help="Export results in CSV format"
        )
    
    with export_col2:
        # Prepare JSON export
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
            "application/json",
            help="Export results in JSON format"
        )
