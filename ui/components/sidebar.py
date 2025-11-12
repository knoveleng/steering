"""
Sidebar component for settings
"""

import streamlit as st
from pathlib import Path
import glob


def render_sidebar():
    """Render sidebar with all settings"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Config file selection
        config_path = st.text_input(
            "Config File",
            value="configs/default.yaml",
            help="Path to configuration file"
        )
        
        # Session selection
        artifacts_dir = st.text_input(
            "Artifacts Directory",
            value="artifacts",
            help="Directory containing calibration sessions"
        )
        
        # Find available sessions
        session_dirs = []
        if Path(artifacts_dir).exists():
            session_dirs = sorted(
                glob.glob(str(Path(artifacts_dir) / "calibration_*")),
                key=lambda x: Path(x).stat().st_mtime,
                reverse=True
            )
        
        if session_dirs:
            session_names = [Path(s).name for s in session_dirs]
            selected_session = st.selectbox(
                "Calibration Session",
                options=session_names,
                help="Select a pre-calibrated session"
            )
            session_path = str(Path(artifacts_dir) / selected_session)
        else:
            st.warning("No calibration sessions found. Run calibration first.")
            session_path = None
        
        # Load button
        if st.button("🚀 Load Model & Calibration", type="primary", disabled=not session_path):
            if session_path:
                from ui.app import load_model_and_calibration
                success = load_model_and_calibration(config_path, session_path)
                if success:
                    st.success("✅ Loaded successfully!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Steering Control")
        
        # Initialize theta value if not exists
        if 'theta_value' not in st.session_state:
            st.session_state.theta_value = 100
            st.session_state.theta_slider = 100
            st.session_state.theta_input = 100

        # Sync logic BEFORE creating widgets
        # Check if slider changed from stored value
        if 'theta_slider' in st.session_state and st.session_state.theta_slider != st.session_state.theta_value:
            # Slider was moved, update input to match
            st.session_state.theta_value = st.session_state.theta_slider
            st.session_state.theta_input = st.session_state.theta_slider
        # Check if input changed from stored value
        elif 'theta_input' in st.session_state and st.session_state.theta_input != st.session_state.theta_value:
            # Input was changed, update slider to match
            st.session_state.theta_value = st.session_state.theta_input
            st.session_state.theta_slider = st.session_state.theta_input

        # Theta controls - both slider and number input
        col1, col2 = st.columns([3, 1])

        with col1:
            st.slider(
                "Steering Angle (θ)",
                min_value=0,
                max_value=360,
                step=10,
                help="Rotation angle in the steering plane",
                key="theta_slider"
            )

        with col2:
            st.number_input(
                "θ°",
                min_value=0,
                max_value=360,
                step=1,
                help="Type exact value",
                key="theta_input"
            )

        # Use the synced value
        theta = st.session_state.theta_value
        
        # Show theta visually
        st.markdown(f'<div class="theta-display">{theta}°</div>', unsafe_allow_html=True)
        
        # Chat template settings
        with st.expander("💬 Chat Template"):
            use_chat_template = st.checkbox(
                "Enable Chat Template",
                value=True,
                help="Use model's chat template for formatting"
            )
            
            if use_chat_template:
                system_prompt = st.text_area(
                    "System Prompt",
                    value="You are a helpful and safe AI assistant.",
                    height=100,
                    help="System message for the model"
                )
            else:
                system_prompt = None
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Angular Steering** allows fine-grained control over model behavior
            by rotating activations in a 2D plane learned from harmful/harmless examples.
            
            **Developed by:** Your Team
            **Version:** 1.0.0
            
            [Documentation](#) | [GitHub](#)
            """)
    
    return {
        'config_path': config_path,
        'session_path': session_path,
        'theta': theta,
        'use_chat_template': use_chat_template if 'use_chat_template' in locals() else True,
        'system_prompt': system_prompt if 'system_prompt' in locals() else None
    }