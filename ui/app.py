"""
Streamlit Chat UI for Angular Steering
"""

import streamlit as st
import torch
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader

from components.sidebar import render_sidebar
from components.chat import render_chat_interface
from utils.session import SessionManager

# Page config
st.set_page_config(
    page_title="Angular Steering Chat",
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
        st.session_state.pipeline = None
        st.session_state.config = None
        st.session_state.session_manager = SessionManager()
        st.session_state.chat_history = []
        st.session_state.model_loaded = False
        st.session_state.calibration_loaded = False
        st.session_state.generation_params = {
            'max_length': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }


def load_model_and_calibration(config_path: str, session_path: str, backend: str):
    """Load model and calibration"""
    try:
        with st.spinner("Loading configuration..."):
            config = ConfigLoader.load(config_path)
            st.session_state.config = config
        
        with st.spinner("Loading model and tokenizer..."):
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
            st.session_state.model_loaded = True
        
        with st.spinner("Initializing pipeline..."):
            pipeline = AngularSteeringPipeline(
                model,
                tokenizer,
                config,
                backend=backend
            )
            st.session_state.pipeline = pipeline
        
        with st.spinner("Loading calibration..."):
            pipeline.load_calibration(session_path)
            st.session_state.calibration_loaded = True
            st.session_state.session_path = session_path
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        return False


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎯 Angular Steering Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive chat with controllable AI behavior</div>', unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Main content
    if not st.session_state.initialized:
        st.markdown('<div class="status-warning">⚠️ Please configure and load a model from the sidebar to start chatting.</div>', unsafe_allow_html=True)
        
        # Show quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Configure Model**: Select your config file in the sidebar
            2. **Choose Session**: Pick a pre-calibrated session
            3. **Load Model**: Click "Load Model & Calibration"
            4. **Adjust Steering**: Use the theta slider or type a value to control behavior
            5. **Start Chatting**: Type your message and see the steered output!
            
            ### What is Angular Steering?
            
            Angular steering allows you to control AI behavior by rotating activations in a learned 2D plane:
            - **θ = 0°**: Original model behavior
            - **θ = 100-200°**: Increased refusal/safety
            - **θ = 300°**: Maximum steering effect
            
            The steering plane is learned from harmful and harmless examples during calibration.
            """)
    else:
        # Status display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Model</div>
                <div style="font-size: 1.2rem; font-weight: 600;">
                    {st.session_state.config['model']['name'].split('/')[-1]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Backend</div>
                <div style="font-size: 1.2rem; font-weight: 600;">
                    {settings['backend'].upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Current θ</div>
                <div style="font-size: 1.2rem; font-weight: 600;">
                    {settings['theta']}°
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main content area with tabs
        tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Advanced Settings"])
        
        with tab1:
            render_chat_interface(st.session_state.pipeline, settings)
        
        with tab2:
            render_advanced_settings()


def render_advanced_settings():
    """Render advanced settings tab"""
    st.subheader("🔧 Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider(
            "Max Length",
            min_value=50,
            max_value=2048,
            value=st.session_state.generation_params['max_length'],
            step=50,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state.generation_params['temperature'],
            step=0.1,
            help="Sampling temperature (higher = more random)"
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
        
        do_sample = st.checkbox(
            "Do Sample",
            value=st.session_state.generation_params['do_sample'],
            help="Use sampling instead of greedy decoding"
        )
    
    # Update session state
    st.session_state.generation_params = {
        'max_length': max_length,
        'temperature': temperature,
        'top_p': top_p,
        'do_sample': do_sample
    }
    
    st.markdown("---")
    
    # Session info
    if st.session_state.calibration_loaded:
        st.subheader("📊 Calibration Info")
        
        try:
            session_path = Path(st.session_state.session_path)
            
            # Show session metadata if available
            metadata_file = session_path / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Layer", metadata.get('best_layer', 'N/A'))
                    st.metric("Harmful Samples", metadata.get('n_harmful', 'N/A'))
                
                with col2:
                    st.metric("Harmless Samples", metadata.get('n_harmless', 'N/A'))
                    st.metric("Steering Mode", metadata.get('steering_mode', 'N/A'))
        except Exception as e:
            st.warning(f"Could not load session metadata: {e}")
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()