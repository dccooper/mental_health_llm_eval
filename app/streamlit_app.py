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
import time

from src.evaluator import (
    PromptEntry,
    EvaluationResult,
    load_prompts,
    evaluate_response,
    generate_report
)
from src.models import query_model
from src.red_flags import check_red_flags, get_severity
from src.validation import ValidationLevel
from src.evaluator import EvaluationError
from src.validation import validate_scores
from src.evaluator import Evaluator
from src.rate_limiter import RateLimitExceeded, LimitType
from src.logging_handler import (
    log_manager,
    ErrorSeverity,
    ValidationError,
    ModelError,
    SafetyError,
    SystemError
)

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
if "error_log_expanded" not in st.session_state:
    st.session_state.error_log_expanded = False

# Initialize evaluator
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = Evaluator()

# Sidebar configuration
with st.sidebar:
    st.title("🧠 Configuration")
    
    # Rate limit status
    st.subheader("📊 Rate Limits")
    rate_limits = st.session_state.evaluator.get_rate_limit_status()
    
    for operation, status in rate_limits.items():
        with st.expander(f"{operation.title()} Limits"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Usage",
                    f"{status['current_usage']}/{status['total_limit']}"
                )
            with col2:
                st.metric(
                    "Available Burst",
                    status['available_tokens']
                )
            
            # Show reset time
            reset_in = status['reset_time'] - time.time()
            st.progress(1 - (reset_in / status['time_window']))
            st.caption(f"Resets in {int(reset_in/60)} minutes")
    
    # Validation level selection
    validation_level = st.selectbox(
        "🔒 Validation Level",
        [level.value for level in ValidationLevel],
        help="Choose how strict the input validation should be"
    )
    validation_level = ValidationLevel(validation_level)
    
    # Prompt bank management
    prompt_file = st.file_uploader(
        "📄 Upload Prompt Bank",
        type=["yaml"],
        help="Upload a YAML file containing evaluation prompts"
    )
    
    if prompt_file:
        try:
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
                try:
                    # Generate and evaluate model response
                    model_response = query_model(selected_prompt.prompt)
                    st.session_state.auto_evaluation = st.session_state.evaluator.evaluate_response(
                        selected_prompt,
                        model_response,
                        validation_level
                    )
                except RateLimitExceeded as e:
                    st.error(f"Rate limit exceeded: {str(e)}")
                    st.info("Please wait for rate limits to reset before continuing.")
                    st.session_state.auto_evaluation = None
                except ValidationError as e:
                    st.error(f"❌ Validation failed: {str(e)}")
                    st.session_state.auto_evaluation = None
                except ModelError as e:
                    st.error(f"🤖 Model error: {str(e)}")
                    st.session_state.auto_evaluation = None
                except SafetyError as e:
                    st.error(f"⚠️ Safety check failed: {str(e)}")
                    st.session_state.auto_evaluation = None
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.session_state.auto_evaluation = None
        except Exception as e:
            st.error(f"Error loading prompts: {str(e)}")
    
    # Session statistics display
    if st.session_state.evaluations:
        st.divider()
        st.subheader("📊 Session Stats")
        
        # Calculate key metrics
        total_evaluations = len(st.session_state.evaluations)
        successful_evals = sum(1 for e in st.session_state.evaluations if not e.validation_warnings)
        average_score = sum(e.total_score for e in st.session_state.evaluations) / total_evaluations
        critical_issues = sum(
            1 for e in st.session_state.evaluations
            if any(f for f in e.detected_red_flags if get_severity([f]) == "critical")
        )
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Evaluations", total_evaluations)
            st.metric("Clean Evaluations", successful_evals)
        with col2:
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

# Display validation warnings if any
if current_evaluation.validation_warnings:
    with st.expander("⚠️ Validation Warnings", expanded=True):
        for warning in current_evaluation.validation_warnings:
            st.warning(warning)

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
        new_score = st.slider(
            f"{dimension} score",
            min_value=0.0,
            max_value=1.0,
            value=float(score),
            step=0.1,
            help=f"Adjust the {dimension} score if needed"
        )
        manual_scores[dimension] = new_score

# Validate manual scores before submission
score_valid = True
try:
    score_validation = validate_scores(manual_scores)
    if not score_validation.is_valid:
        st.error(f"Invalid scores: {'; '.join(score_validation.errors)}")
        score_valid = False
    elif score_validation.warnings:
        for warning in score_validation.warnings:
            st.warning(warning)
except Exception as e:
    st.error(f"Score validation error: {str(e)}")
    score_valid = False

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
if st.button("✅ Submit Evaluation", type="primary", disabled=not score_valid):
    try:
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
            ),
            validation_warnings=current_evaluation.validation_warnings
        )
        
        # Check rate limit before adding evaluation
        st.session_state.evaluator.rate_limiter.try_acquire(LimitType.PROMPT)
        st.session_state.evaluations.append(modified_evaluation)
        st.success("✅ Evaluation submitted successfully!")
    except RateLimitExceeded as e:
        st.error(f"Rate limit exceeded: {str(e)}")
        st.info("Please wait for rate limits to reset before submitting more evaluations.")
    except Exception as e:
        st.error(f"Error submitting evaluation: {str(e)}")

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

def display_error_tracking():
    """Display error tracking information and visualizations."""
    st.sidebar.markdown("### 🔍 Error Tracking")
    
    # Get error summary
    summary = log_manager.get_error_summary()
    
    # Display summary metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Errors", summary["total"])
    with col2:
        critical = summary["by_severity"]["critical"]
        st.metric("Critical Issues", critical, 
                 delta=critical, 
                 delta_color="inverse")
    
    # Error type distribution
    if summary["by_type"]:
        st.sidebar.markdown("#### Error Types")
        error_types_df = pd.DataFrame([
            {"Type": t, "Count": c}
            for t, c in summary["by_type"].items()
        ])
        fig = px.bar(error_types_df, x="Type", y="Count",
                    color="Count",
                    color_continuous_scale=["green", "yellow", "red"])
        st.sidebar.plotly_chart(fig, use_container_width=True)
    
    # Severity distribution
    st.sidebar.markdown("#### Error Severity")
    severity_df = pd.DataFrame([
        {"Severity": s, "Count": c}
        for s, c in summary["by_severity"].items()
        if c > 0
    ])
    if not severity_df.empty:
        fig = px.pie(severity_df, names="Severity", values="Count",
                    color="Severity",
                    color_discrete_map={
                        "low": "green",
                        "medium": "yellow",
                        "high": "orange",
                        "critical": "red"
                    })
        st.sidebar.plotly_chart(fig, use_container_width=True)
    
    # Detailed error log
    if st.sidebar.checkbox("Show Error Log", value=st.session_state.error_log_expanded):
        st.sidebar.markdown("#### Recent Errors")
        for error in reversed(log_manager.errors[-10:]):  # Show last 10 errors
            with st.sidebar.expander(f"{error.error_type}: {error.message[:50]}..."):
                st.write(f"**Severity:** {error.severity.value}")
                st.write(f"**Time:** {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                if error.additional_info:
                    st.json(error.additional_info)
                if error.stack_trace:
                    st.code(error.stack_trace)
        st.session_state.error_log_expanded = True
    else:
        st.session_state.error_log_expanded = False

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Mental Health LLM Evaluator",
        page_icon="🧠",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Title and description
    st.title("🧠 Mental Health LLM Evaluator")
    st.markdown("""
        Evaluate LLM responses for safety, clinical appropriateness, and empathy
        in mental health contexts.
    """)
    
    # Display error tracking in sidebar
    display_error_tracking()
    
    # Main interface
    try:
        # Load prompt bank
        prompt_file = st.file_uploader(
            "Upload Prompt Bank (YAML)",
            type="yaml",
            help="Select a YAML file containing evaluation prompts"
        )
        
        if prompt_file:
            try:
                prompts = load_prompts(prompt_file)
                st.success(f"✅ Loaded {len(prompts)} prompts")
                
                # Log successful prompt load
                log_manager.log_audit("load_prompts", {
                    "file": prompt_file.name,
                    "prompt_count": len(prompts)
                })
                
                # Prompt selection
                selected_prompt = st.selectbox(
                    "Select Prompt",
                    prompts,
                    format_func=lambda p: f"{p.id}: {p.category} - {p.prompt[:100]}..."
                )
                
                # Validation level selection
                validation_level = st.select_slider(
                    "Validation Level",
                    options=[v for v in ValidationLevel],
                    value=ValidationLevel.STANDARD,
                    format_func=lambda x: x.value.title()
                )
                
                # Update current evaluation state
                if selected_prompt != st.session_state.current_prompt:
                    st.session_state.current_prompt = selected_prompt
                    try:
                        # Generate and evaluate model response
                        model_response = query_model(selected_prompt.prompt)
                        st.session_state.auto_evaluation = st.session_state.evaluator.evaluate_response(
                            selected_prompt,
                            model_response,
                            validation_level
                        )
                    except RateLimitExceeded as e:
                        st.error(f"⚠️ Rate limit exceeded: {str(e)}")
                        st.info("Please wait for rate limits to reset before continuing.")
                        st.session_state.auto_evaluation = None
                    except ValidationError as e:
                        st.error(f"❌ Validation failed: {str(e)}")
                        st.session_state.auto_evaluation = None
                    except ModelError as e:
                        st.error(f"🤖 Model error: {str(e)}")
                        st.session_state.auto_evaluation = None
                    except SafetyError as e:
                        st.error(f"⚠️ Safety check failed: {str(e)}")
                        st.session_state.auto_evaluation = None
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.session_state.auto_evaluation = None
                
            except Exception as e:
                log_manager.log_error(
                    SystemError(f"Failed to load prompts: {str(e)}"),
                    {"file": prompt_file.name}
                )
                st.error(f"Error loading prompts: {str(e)}")
        
        # Display current evaluation
        if st.session_state.auto_evaluation:
            current_evaluation = st.session_state.auto_evaluation
            
            # Display prompt details
            st.header("Prompt Details")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Category:** {current_evaluation.prompt_entry.category}")
                st.markdown(f"**ID:** {current_evaluation.prompt_entry.id}")
            with col2:
                st.markdown(f"**Subcategory:** {current_evaluation.prompt_entry.subcategory}")
            
            st.markdown("**Prompt:**")
            st.markdown(f">{current_evaluation.prompt_entry.prompt}")
            
            if current_evaluation.prompt_entry.context:
                with st.expander("Show Context"):
                    st.markdown(current_evaluation.prompt_entry.context)
            
            # Display model response
            st.header("Model Response")
            st.markdown(current_evaluation.model_response)
            
            # Display red flags
            if current_evaluation.detected_red_flags:
                st.error("⚠️ Red Flags Detected:")
                for flag in current_evaluation.detected_red_flags:
                    severity = get_severity(flag)
                    st.markdown(f"- **{severity.upper()}:** {flag}")
            
            # Display scores with manual adjustment
            st.header("Evaluation Scores")
            
            manual_scores = {}
            score_valid = True
            
            for dimension, score in current_evaluation.scores.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    manual_scores[dimension] = st.slider(
                        dimension.title(),
                        min_value=0.0,
                        max_value=1.0,
                        value=float(score),
                        step=0.1,
                        format="%.1f"
                    )
                with col2:
                    if manual_scores[dimension] != score:
                        st.markdown("*(Modified)*")
            
            try:
                score_validation = validate_scores(manual_scores)
                if not score_validation.is_valid:
                    st.error(f"❌ Invalid scores: {'; '.join(score_validation.errors)}")
                    score_valid = False
                elif score_validation.warnings:
                    for warning in score_validation.warnings:
                        st.warning(f"⚠️ {warning}")
            except Exception as e:
                log_manager.log_error(e, {"stage": "score_validation"})
                st.error(f"Score validation error: {str(e)}")
                score_valid = False
            
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
            if st.button("✅ Submit Evaluation", type="primary", disabled=not score_valid):
                try:
                    # Create modified evaluation result
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
                        ),
                        validation_warnings=current_evaluation.validation_warnings
                    )
                    
                    # Check rate limit before adding evaluation
                    st.session_state.evaluator.rate_limiter.try_acquire(LimitType.PROMPT)
                    st.session_state.evaluations.append(modified_evaluation)
                    
                    # Log successful submission
                    log_manager.log_audit("evaluation_submitted", {
                        "prompt_id": modified_evaluation.prompt_entry.id,
                        "total_score": modified_evaluation.total_score,
                        "modified": any(
                            manual_scores[k] != current_evaluation.scores[k]
                            for k in manual_scores
                        )
                    })
                    
                    st.success("✅ Evaluation submitted successfully!")
                    
                except RateLimitExceeded as e:
                    st.error(f"⚠️ Rate limit exceeded: {str(e)}")
                    st.info("Please wait for rate limits to reset before submitting more evaluations.")
                except Exception as e:
                    log_manager.log_error(e, {"stage": "submission"})
                    st.error(f"Error submitting evaluation: {str(e)}")
    
    except Exception as e:
        log_manager.log_error(
            SystemError(f"Application error: {str(e)}"),
            {"location": "main"}
        )
        st.error(f"❌ Application error: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
