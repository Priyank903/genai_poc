"""
Streamlit Chat Interface for Multi-Agent Post-Discharge Care System
"""

import streamlit as st
import sys
import os
from datetime import datetime
import json

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_agent_system import MultiAgentSystem

# Page configuration
st.set_page_config(
    page_title="Post-Discharge Care Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .receptionist-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .clinical-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .patient-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'system' not in st.session_state:
    st.session_state.system = None
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None

@st.cache_resource
def initialize_system():
    """Initialize the multi-agent system"""
    try:
        return MultiAgentSystem()
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None

def display_message(message, is_user=False):
    """Display a chat message"""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        agent = message.get('agent', 'system')
        response = message.get('response', '')
        
        with st.chat_message("assistant"):
            # Agent badge
            if agent == 'receptionist':
                st.markdown('<div class="agent-badge receptionist-badge">👋 Receptionist Agent</div>', 
                          unsafe_allow_html=True)
            elif agent == 'clinical':
                st.markdown('<div class="agent-badge clinical-badge">🩺 Clinical AI Agent</div>', 
                          unsafe_allow_html=True)
            
            st.write(response)
            
            # Show patient info if patient was just found
            if message.get('patient_found') and message.get('patient_data'):
                patient_data = message.get('patient_data')
                st.markdown("---")
                st.markdown("### 📋 Patient Information")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Name:** {patient_data.get('patient_name', 'N/A')}")
                    st.markdown(f"**Diagnosis:** {patient_data.get('primary_diagnosis', 'N/A')}")
                    st.markdown(f"**Discharge Date:** {patient_data.get('discharge_date', 'N/A')}")
                
                with col2:
                    medications = patient_data.get('medications', [])
                    if medications:
                        st.markdown("**Medications:**")
                        for med in medications:
                            st.markdown(f"• {med}")
                
                # Dietary restrictions
                diet = patient_data.get('dietary_restrictions', '')
                if diet:
                    st.markdown(f"**Dietary:** {diet}")
                
                # Follow-up
                follow_up = patient_data.get('follow_up', '')
                if follow_up:
                    st.markdown(f"**Follow-up:** {follow_up}")
                
                # Warning signs
                warnings = patient_data.get('warning_signs', '')
                if warnings:
                    st.markdown(f"**Warning Signs:** {warnings}")
            
            # ALWAYS show RAG chunks if available
            if message.get('relevant_chunks'):
                st.markdown("---")
                st.markdown("### 📚 Retrieved RAG Chunks")
                chunks = message.get('relevant_chunks', [])[:3]  # Always show top 3
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"📖 Chunk {i} - Page {chunk.get('page', 'N/A')} (Score: {chunk.get('similarity_score', 0):.3f})"):
                        st.write(chunk.get('text', 'No text available'))
                        st.caption(f"Source: {chunk.get('source', 'Unknown')}")
            
            # Show additional info if available
            if message.get('chunks_found', 0) > 0:
                st.caption(f"📚 Found {message['chunks_found']} relevant medical references")
            
            if message.get('source'):
                source_emoji = "🌐" if "web" in message['source'] else "📖"
                st.caption(f"{source_emoji} Source: {message['source']}")

def display_patient_info(patient_data):
    """Display patient information in sidebar"""
    if patient_data:
        st.sidebar.markdown("### 👤 Current Patient")
        
        # Patient basic info
        st.sidebar.markdown(f"""
        <div class="patient-info">
            <strong>Name:</strong> {patient_data.get('patient_name', 'N/A')}<br>
            <strong>Diagnosis:</strong> {patient_data.get('primary_diagnosis', 'N/A')}<br>
            <strong>Discharge Date:</strong> {patient_data.get('discharge_date', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        # Medications
        medications = patient_data.get('medications', [])
        if medications:
            st.sidebar.markdown("**💊 Current Medications:**")
            for med in medications:
                st.sidebar.markdown(f"• {med}")
        
        # Dietary restrictions
        diet = patient_data.get('dietary_restrictions', '')
        if diet:
            st.sidebar.markdown(f"**🥗 Dietary Restrictions:**\n{diet}")
        
        # Warning signs
        warnings = patient_data.get('warning_signs', '')
        if warnings:
            st.sidebar.markdown("**⚠️ Warning Signs to Watch:**")
            st.sidebar.markdown(f"<div class='warning-box'>{warnings}</div>", 
                              unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Post-Discharge Care Assistant</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state.system is None:
        with st.spinner("Initializing AI system..."):
            st.session_state.system = initialize_system()
    
    if st.session_state.system is None:
        st.error("❌ Failed to initialize the system. Please check your setup and try again.")
        st.stop()
    
    # Add initial greeting if no messages exist
    if not st.session_state.messages:
        initial_greeting = {
            "role": "assistant", 
            "content": {
                "response": "Hello! I'm your post-discharge care assistant. What's your name?",
                "agent": "receptionist"
            }
        }
        st.session_state.messages.append(initial_greeting)
        # Set receptionist to name collection state to avoid double greeting
        st.session_state.system.receptionist.conversation_state = "name_collection"
    
    # Simple reset button in main area
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Reset", type="secondary"):
            st.session_state.system.reset_conversation()
            st.session_state.messages = []
            st.session_state.current_patient = None
            st.rerun()
        
    
    # Main chat interface
    st.markdown("### 💬 Chat with Your Care Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_message(message["content"], is_user=True)
        else:
            display_message(message["content"], is_user=False)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message(prompt, is_user=True)
        
        # Get system response
        with st.spinner("Processing..."):
            try:
                response = st.session_state.system.process_user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_message(response, is_user=False)
            except Exception as e:
                st.error(f"❌ Error processing your request: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.875rem;'>
        ⚠️ <strong>Medical Disclaimer:</strong> This is an AI assistant for educational purposes only. 
        Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
