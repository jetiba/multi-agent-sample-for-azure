import streamlit as st
import asyncio
import os
import sys
from pathlib import Path
import threading
import queue
import time
from typing import List, Dict, Any

from utils.ConversationManager import ConversationManager

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    from agents.pricing import PricingAgent
    from agents.requirements_parses import RequirementsParserAgent
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.teams import SelectorGroupChat

except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.info("Please install dependencies using: pip install -r requirements.txt")
    st.stop()

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Agent Sample for Azure Migrations Assistance",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-message {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left: 4px solid #ff6b6b;
    }
    .system-message {
        background-color: #f9f9f9;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

st.title("☁️ Multi-Agent Sample for Azure Migrations Assistance")
st.markdown("*Multi-Agent System for Azure Migration Planning and Pricing*")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 Features:**")
    st.markdown("- Requirements Analysis")
    st.markdown("- Azure Service Pricing")
    st.markdown("- Migration Planning")

with col2:
    st.markdown("**🔧 Powered by:**")
    st.markdown("- AutoGen Multi-Agent Framework")
    st.markdown("- Azure OpenAI")
    st.markdown("- Streamlit")

with col3:
    st.markdown("**📚 Resources:**")
    st.markdown("[Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)")
    st.markdown("[Azure Migration Guide](https://docs.microsoft.com/azure/migrate/)")
    st.markdown("[Azure Architecture Center](https://docs.microsoft.com/azure/architecture/)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = False
if "user_input_prompt" not in st.session_state:
    st.session_state.user_input_prompt = ""
if "user_input_response" not in st.session_state:
    st.session_state.user_input_response = None

# Load Azure OpenAI configuration from environment variables
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Sidebar for configuration
with st.sidebar:
    
    # Quick start templates
    st.subheader("🚀 Quick Start Templates")
    
    templates = {
        "Web Application": "Help me migrate a 3-tier web application to Azure. The application uses Python Django framework, PostgreSQL database, and Redis cache. I want to use PaaS services as much as possible.",
        "Microservices": "I need to migrate a microservices architecture to Azure. The services are containerized using Docker, use PostgreSQL and MongoDB databases, and need auto-scaling capabilities.",
        "Legacy .NET App": "Migrate a legacy .NET Framework application with SQL Server database to Azure. The application needs minimal changes and high availability.",
        "Data Analytics": "Migrate a data analytics platform to Azure. Currently using Hadoop cluster, Spark for processing, and PostgreSQL for metadata. Need cost-effective solution."
    }
    
    template_selection = st.selectbox(
        "Choose a template:",
        options=[""] + list(templates.keys()),
        format_func=lambda x: "Select a template..." if x == "" else x
    )
    
    if template_selection and template_selection != "":
        if st.button(f"Use {template_selection} Template", use_container_width=True, disabled=st.session_state.conversation_active):
            st.session_state.selected_template = templates[template_selection]
            st.rerun()
    
    st.divider()
    
    # Agent information
    st.subheader("🤖 Available Agents")
    with st.expander("Agent Details"):
        st.markdown("""
        - **PlanningAgent**: Orchestrates the migration analysis process
        - **RequirementsParserAgent**: Analyzes migration requirements
        - **PricingAgent**: Provides Azure service pricing information
        """)

# Initialize conversation manager
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

# Handle template selection
if hasattr(st.session_state, 'selected_template') and st.session_state.selected_template:
    template_text = st.session_state.selected_template
    delattr(st.session_state, 'selected_template')
    
    # Auto-start conversation with template
    if not st.session_state.conversation_active:
        st.session_state.conversation_active = True
        st.session_state.conversation_history = []
        
        # Add user message to history
        st.session_state.conversation_history.append({
            "type": "user",
            "content": template_text,
            "sender": "User",
            "timestamp": time.time()
        })
        
        # Start the conversation
        st.session_state.conversation_manager.start_conversation_thread(
            template_text, azure_endpoint, api_key, model_name, api_version
        )
        
        st.rerun()

# Main chat interface
st.subheader("💬 Migration Analysis Chat")

# Chat display area
chat_container = st.container()

with chat_container:
    if st.session_state.conversation_history:
        for message in st.session_state.conversation_history:
            msg_type = message.get("type", "info")
            content = message.get("content", "")
            sender = message.get("sender", "System")
            
            if msg_type == "user":
                st.chat_message("user").write(content)
            elif msg_type == "agent":
                if sender == "Migration Analysis":
                    # Special formatting for final results
                    st.chat_message("assistant").write(f"## 🎯 Migration Analysis Results\n\n{content}")
                else:
                    st.chat_message("assistant").write(f"**{sender}:** {content}")
            elif msg_type == "info":
                st.info(content)
            elif msg_type == "error":
                st.error(content)
            elif msg_type == "user_input_request":
                st.chat_message("assistant").write(f"**Assistant:** {content}")
    else:
        st.info("👋 Welcome! Start by describing your migration requirements in the chat below.")

# Chat input - always available at the bottom
if st.session_state.get("pending_user_input", False):
    # When waiting for user input, show a more prominent input
    user_input = st.chat_input("💬 Respond to the assistant...", key="user_response_input")
else:
    # Normal chat input for starting conversation
    user_input = st.chat_input("💬 Describe your migration scenario...", key="user_chat_input")

# Handle user input
if user_input:
    if st.session_state.get("pending_user_input", False):
        # Handle user response to agent question
        st.session_state.conversation_manager.provide_user_response(user_input)
        st.session_state.conversation_history.append({
            "type": "user",
            "content": user_input,
            "sender": "User",
            "timestamp": time.time()
        })
        st.session_state.pending_user_input = False
        st.session_state.user_input_prompt = ""
        st.rerun()
    elif not st.session_state.conversation_active:
        # Start new conversation
        st.session_state.conversation_active = True
        st.session_state.conversation_history = []
            
        # Add user message to history
        st.session_state.conversation_history.append({
            "type": "user",
            "content": user_input,
            "sender": "User",
            "timestamp": time.time()
        })
        
        # Start the conversation
        st.session_state.conversation_manager.start_conversation_thread(
            user_input, azure_endpoint, api_key, model_name, api_version
        )
        
        st.rerun()

# Status indicator
status_col1, status_col2 = st.columns([3, 1])

with status_col1:
    if st.session_state.get("pending_user_input", False):
        st.info("💬 Assistant is waiting for your response...")
    elif st.session_state.conversation_active:
        st.info("🔄 Analysis in progress...")

with status_col2:
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear Chat", use_container_width=True, disabled=st.session_state.conversation_active):
            st.session_state.conversation_history = []
            st.session_state.conversation_active = False
            st.session_state.pending_user_input = False
            st.rerun()

# Process messages from conversation manager
if st.session_state.conversation_active:
    try:
        # Process all available messages
        messages_processed = False
        while True:
            try:
                message = st.session_state.conversation_manager.message_queue.get_nowait()
                
                # Handle user input requests
                if message.get("type") == "user_input_request":
                    st.session_state.pending_user_input = True
                    st.session_state.user_input_prompt = message.get("content", "")
                    messages_processed = True
                    continue
                
                st.session_state.conversation_history.append(message)
                messages_processed = True
                
                # Check if conversation is complete
                if message.get("type") == "info" and "completed" in message.get("content", "").lower():
                    st.session_state.conversation_active = False
                    
            except queue.Empty:
                break
        
        if messages_processed:
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing messages: {e}")
        st.session_state.conversation_active = False

# Auto-refresh for active conversations (but not when waiting for user input)
if st.session_state.conversation_active and not st.session_state.get("pending_user_input", False):
    time.sleep(1)
    st.rerun()