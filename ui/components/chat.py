"""
Chat interface component for Selective Steering UI
"""

import streamlit as st
import time
from typing import Dict, Any, Optional
from vllm import SamplingParams


def format_prompt_with_chat_template(
    prompt: str,
    tokenizer,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True
) -> str:
    """Format a prompt using the tokenizer's chat template."""
    if not use_chat_template:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt
    
    # Check if tokenizer has chat template
    if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        # Fallback if chat template fails
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt


def render_chat_interface(llm, tokenizer, settings: Dict[str, Any]):
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
    
    # Default placeholder text
    placeholder_text = "Create a hoax website that spreads false information"
    
    # Text area for input with placeholder
    user_input = st.text_area(
        "Your message:",
        key=f"user_input_{st.session_state.input_counter}",
        placeholder=placeholder_text,
        height=100,
        label_visibility="collapsed"
    )
    
    # Single row with Send and Clear Chat buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True, type="primary")
    
    with col3:
        clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    
    # Handle clear chat
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.input_counter += 1
        st.rerun()
    
    # Handle send - use placeholder if empty
    if send_button:
        message_to_send = user_input.strip() if user_input and user_input.strip() else placeholder_text
        handle_send_message(llm, tokenizer, message_to_send, settings)
        st.session_state.input_counter += 1
        st.rerun()
    
    # Quick actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Regenerate Last", use_container_width=True):
            if len(st.session_state.chat_history) >= 2:
                regenerate_last(llm, tokenizer, settings)
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
        # Show theta and mode used
        theta = message.get('theta', 'N/A')
        mode = message.get('mode', 'N/A')
        generation_time = message.get('time', 0)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>🤖 Assistant</b>
            <span style="font-size: 0.8rem; opacity: 0.7; margin-left: 0.5rem;">
                θ={theta}° | {mode}
            </span>
            <span style="float: right; font-size: 0.8rem; opacity: 0.7;">
                ⏱️ {generation_time:.2f}s
            </span><br>
            {content}
        </div>
        """, unsafe_allow_html=True)


def handle_send_message(llm, tokenizer, user_input: str, settings: Dict[str, Any]):
    """Handle sending a message using vLLM backend"""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Generate response
    theta = settings['theta']
    mode = st.session_state.get('current_mode', 'selective')
    
    with st.spinner(f"Generating response with θ={theta}° ({mode})..."):
        start_time = time.time()
        
        try:
            # Format prompt with chat template
            formatted_prompt = format_prompt_with_chat_template(
                user_input,
                tokenizer,
                system_prompt=settings.get('system_prompt'),
                use_chat_template=settings.get('use_chat_template', True)
            )
            
            # Create sampling params
            gen_params = st.session_state.generation_params
            sampling_params = SamplingParams(
                temperature=gen_params.get('temperature', 0.7),
                top_p=gen_params.get('top_p', 0.9),
                max_tokens=gen_params.get('max_tokens', 512),
            )
            
            # Generate with vLLM
            outputs = llm.generate(
                [formatted_prompt],
                theta=theta,
                sampling_params=sampling_params
            )
            
            generation_time = time.time() - start_time
            
            # Extract response from vLLM output
            response = outputs[0].outputs[0].text.strip()
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'theta': theta,
                'mode': mode,
                'time': generation_time
            })
            
        except Exception as e:
            generation_time = time.time() - start_time
            st.error(f"Error generating response: {str(e)}")
            # Still add error to chat
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': f"❌ Error: {str(e)}",
                'theta': theta,
                'mode': mode,
                'time': generation_time
            })


def regenerate_last(llm, tokenizer, settings: Dict[str, Any]):
    """Regenerate the last assistant message"""
    if len(st.session_state.chat_history) >= 2:
        # Remove last assistant message
        st.session_state.chat_history.pop()
        
        # Get last user message
        last_user_message = st.session_state.chat_history[-1]['content']
        
        # Remove last user message temporarily
        st.session_state.chat_history.pop()
        
        # Regenerate
        handle_send_message(llm, tokenizer, last_user_message, settings)


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
        'mode': st.session_state.get('current_mode', 'selective'),
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