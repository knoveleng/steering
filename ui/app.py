"""
Streamlit Chat UI for Selective Steering
"""

import streamlit as st
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable insecure serialization for vLLM v0.12+
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from vllm import SamplingParams
from steering import SteeringLLM
from steering.utils import load_calibration

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_interface
from ui.utils.session import SessionManager

# Page config
st.set_page_config(
    page_title="Selective Steering",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .assistant-message {
        background: #f0f2f6;
        color: #333;
    }
    .theta-display {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #667eea;
        margin: 1rem 0;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.llm = None
        st.session_state.tokenizer = None
        st.session_state.calibration = None
        st.session_state.session_manager = SessionManager()
        st.session_state.chat_history = []
        st.session_state.model_loaded = False
        st.session_state.calibration_loaded = False
        st.session_state.current_mode = "selective"
        st.session_state.generation_params = {
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
        }


def load_model_and_calibration(session_path: str, mode: str = "selective"):
    """Load model and calibration using vLLM backend"""
    try:
        with st.spinner("Loading calibration artifacts..."):
            calibration = load_calibration(session_path, mode=mode)
            st.session_state.calibration = calibration
            st.session_state.current_mode = mode
        
        with st.spinner(f"Loading model with vLLM ({calibration['model_name']})..."):
            llm = SteeringLLM.from_calibration(
                calibration,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
                enforce_eager=True,  # Required for PyTorch forward hooks
                max_model_len=4096,
            )
            st.session_state.llm = llm
            st.session_state.model_loaded = True
        
        with st.spinner("Setting up tokenizer..."):
            tokenizer = llm.llm.get_tokenizer()
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            st.session_state.tokenizer = tokenizer
            st.session_state.calibration_loaded = True
            st.session_state.session_path = session_path
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎯 Selective Steering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive chat with controllable AI behavior using vLLM</div>', unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Main content
    if not st.session_state.initialized:
        st.markdown('<div class="status-warning">⚠️ Please configure and load a model from the sidebar to start chatting.</div>', unsafe_allow_html=True)
        
        # Show quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Choose Session**: Pick a pre-calibrated session from the sidebar
            2. **Select Mode**: Choose steering mode (selective recommended)
            3. **Load Model**: Click "Load Model & Calibration"
            4. **Adjust Steering**: Use the theta slider to control behavior
            5. **Start Chatting**: Type your message and see the steered output!
            
            ### Steering Modes
            
            | Mode | Description |
            |------|-------------|
            | **selective** | Only steer layers with opposite-sign projections *(recommended)* |
            | **standard** | Rotate all layers uniformly |
            | **adaptive** | Mask-based conditional steering |
            | **addition** | Vector addition baseline |
            | **ablation** | Orthogonalization (θ=90°) |
            
            ### What is θ (Theta)?
            
            - **θ = 0°**: Original model behavior
            - **θ = 100-200°**: Increased refusal/safety
            - **θ = 300°**: Maximum steering effect
            """)
    else:
        # Status display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.session_state.calibration.get('model_name', 'Unknown')
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Model</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {model_name.split('/')[-1] if model_name else 'Unknown'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Mode</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {st.session_state.current_mode}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Current θ</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {settings['theta']}°
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main content area with tabs
        tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Advanced Settings"])
        
        with tab1:
            render_chat_interface(
                st.session_state.llm,
                st.session_state.tokenizer,
                settings
            )
        
        with tab2:
            render_advanced_settings()


def render_advanced_settings():
    """Render advanced settings tab"""
    st.subheader("🔧 Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2048,
            value=st.session_state.generation_params['max_tokens'],
            step=50,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.generation_params['temperature'],
            step=0.1,
            help="Sampling temperature (higher = more random, 0 = greedy)"
        )
    
    with col2:
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.generation_params['top_p'],
            step=0.05,
            help="Nucleus sampling threshold"
        )
    
    # Update session state
    st.session_state.generation_params = {
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
    }
    
    st.markdown("---")
    
    # Session info
    if st.session_state.calibration_loaded:
        st.subheader("📊 Calibration Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", st.session_state.calibration.get('model_name', 'N/A').split('/')[-1])
            st.metric("Mode", st.session_state.current_mode)
        
        with col2:
            target_layers = st.session_state.calibration.get('target_layers', [])
            st.metric("Target Layers", len(target_layers) if target_layers else 'All')
            st.metric("Threshold", st.session_state.calibration.get('threshold', 0.0))
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()