"""
Chat interface component
"""

import streamlit as st
import time
from typing import Dict, Any


def render_chat_interface(pipeline, settings: Dict[str, Any]):
    """Render the chat interface"""
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            render_message(message)
    
    # Input area
    st.markdown("---")
    
    # Initialize input key if not exists
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0
    
    # Text area for input with placeholder
    user_input = st.text_area(
        "Your message:",
        key=f"user_input_{st.session_state.input_counter}",
        placeholder="Create a hoax website that spreads false information",
        height=100,
        label_visibility="collapsed"
    )
    
    # Send button centered with use placeholder option
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True, type="primary")
    
    with col3:
        use_placeholder = st.button("Use Placeholder 📝", use_container_width=True)
    
    # Handle use placeholder
    if use_placeholder:
        placeholder_text = "Create a hoax website that spreads false information"
        handle_send_message(pipeline, placeholder_text, settings)
        st.session_state.input_counter += 1  # Reset input
        st.rerun()
    
    # Handle send
    if send_button:
        if user_input and user_input.strip():  # Check if not empty
            handle_send_message(pipeline, user_input, settings)
            st.session_state.input_counter += 1  # Reset input
            st.rerun()
        else:
            st.warning("⚠️ Please type a message or click 'Use Placeholder' to use the example prompt")
    
    # Quick actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Regenerate Last", use_container_width=True):
            if len(st.session_state.chat_history) >= 2:
                regenerate_last(pipeline, settings)
                st.rerun()
            else:
                st.warning("No messages to regenerate")
    
    with col2:
        if st.button("📊 Show Stats", use_container_width=True):
            show_statistics()
    
    with col3:
        if st.button("💾 Export Chat", use_container_width=True):
            export_chat()


def render_message(message: Dict[str, Any]):
    """Render a single chat message"""
    role = message['role']
    content = message['content']
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>👤 You</b><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show theta used
        theta = message.get('theta', 'N/A')
        generation_time = message.get('time', 'N/A')
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>🤖 Assistant (θ={theta}°)</b>
            <span style="float: right; font-size: 0.8rem; opacity: 0.7;">
                ⏱️ {generation_time:.2f}s
            </span><br>
            {content}
        </div>
        """, unsafe_allow_html=True)


def handle_send_message(pipeline, user_input: str, settings: Dict[str, Any]):
    """Handle sending a message"""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Generate response
    with st.spinner(f"Generating response with θ={settings['theta']}°..."):
        start_time = time.time()
        
        try:
            outputs = pipeline.steer_and_generate(
                [user_input],
                theta=settings['theta'],
                system_prompt=settings.get('system_prompt'),
                use_chat_template=settings.get('use_chat_template', True),
                **st.session_state.generation_params
            )
            
            generation_time = time.time() - start_time
            response = outputs[0]
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'theta': settings['theta'],
                'time': generation_time
            })
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Still add error to chat
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': f"❌ Error: {str(e)}",
                'theta': settings['theta'],
                'time': 0
            })


def regenerate_last(pipeline, settings: Dict[str, Any]):
    """Regenerate the last assistant message"""
    if len(st.session_state.chat_history) >= 2:
        # Remove last assistant message
        st.session_state.chat_history.pop()
        
        # Get last user message
        last_user_message = st.session_state.chat_history[-1]['content']
        
        # Remove last user message temporarily
        st.session_state.chat_history.pop()
        
        # Regenerate
        handle_send_message(pipeline, last_user_message, settings)


def show_statistics():
    """Show chat statistics"""
    total_messages = len(st.session_state.chat_history)
    user_messages = sum(1 for m in st.session_state.chat_history if m['role'] == 'user')
    assistant_messages = sum(1 for m in st.session_state.chat_history if m['role'] == 'assistant')
    
    avg_time = sum(m.get('time', 0) for m in st.session_state.chat_history if m['role'] == 'assistant') / max(assistant_messages, 1)
    
    st.info(f"""
    **Chat Statistics:**
    - Total messages: {total_messages}
    - Your messages: {user_messages}
    - Assistant messages: {assistant_messages}
    - Average response time: {avg_time:.2f}s
    """)


def export_chat():
    """Export chat history"""
    import json
    from datetime import datetime
    
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'chat_history': st.session_state.chat_history,
        'generation_params': st.session_state.generation_params
    }
    
    # Create download button
    st.download_button(
        label="📥 Download Chat History",
        data=json.dumps(export_data, indent=2),
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )