import streamlit as st
import asyncio
import os
import sys
from pathlib import Path
import threading
import queue
import time
from typing import List, Dict, Any

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
    import logging
    from autogen_core import TRACE_LOGGER_NAME

except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.info("Please install dependencies using: pip install -r requirements.txt")
    st.stop()

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Streamlit page configuration
st.set_page_config(
    page_title="Azure Migration Assistant",
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

st.title("☁️ Azure Migration Assistant")
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

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Azure OpenAI configuration
    st.subheader("Azure OpenAI Settings")
    
    with st.expander("Configuration Details", expanded=True):
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint", 
            value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            help="Your Azure OpenAI service endpoint URL"
        )
        api_key = st.text_input(
            "API Key", 
            value=os.getenv("AZURE_OPENAI_API_KEY", ""),
            type="password",
            help="Your Azure OpenAI API key"
        )
        model_name = st.text_input(
            "Model Name", 
            value=os.getenv("AZURE_OPENAI_MODEL", ""),
            help="The Azure OpenAI model to use"
        )
        api_version = st.text_input(
            "API Version", 
            value=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            help="The Azure OpenAI API version to use"
        )
    
    # Configuration validation
    config_valid = all([azure_endpoint, api_key])
    if config_valid:
        st.success("✅ Configuration Valid")
    else:
        st.warning("⚠️ Please complete configuration")
    
    st.divider()
    
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

class ConversationManager:
    """Manages the multi-agent conversation"""
    
    def __init__(self):
        self.message_queue = queue.Queue()
        self.conversation_thread = None
        self.model_client = None
        self.input_event = threading.Event()
        self.user_response = None
    
    async def create_model_client(self, endpoint: str, api_key: str, model: str, api_version: str):
        """Create Azure OpenAI model client"""
        return AzureOpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    
    async def initialize_agents(self, model_client):
        """Initialize all agents"""
        # Initialize specialized agents
        pa = PricingAgent().initialize(model_client=model_client)
        rpa = RequirementsParserAgent().initialize(model_client=model_client)
        
        # Planning agent with enhanced system message
        planning_agent = AssistantAgent(
            "PlanningAgent",
            description="An agent for planning Azure migration tasks",
            model_client=model_client,
            system_message="""
            You are an Azure Migration Planning Agent. Your role is to coordinate a comprehensive migration analysis.

            Your team consists of:
            - RequirementsParserAgent: Analyzes, extracts and generate request for missing migration requirements
            - PricingAgent: Provides Azure service pricing and cost analysis
            - UserProxyAgent: Acts as a proxy for the user, providing input and feedback to the team

            Your responsibilities:
            - Manage the overall plan for solving the user's question.
            - Decide when more information is needed from the user.
            - Route any requests for user feedback or clarification through the UserProxyAgent.

            You may receive suggestions or questions from other agents. If they require user input, you must determine whether it's appropriate and, if so, send a request to the UserProxyAgent on their behalf.

            Strict rule: You are the only agent that can call or relay messages through the UserProxyAgent.
            
            Process:
            1. Start by having RequirementsParserAgent analyze the user's requirements
            2. If RequirementsParserAgent generate a request for missing information, take the request in <Request for user> and ask UserProxyAgent for user feedback.
            3. Based on the requirements, ask PricingAgent for relevant pricing information
            4. Provide a comprehensive migration recommendation including:
               - Recommended Azure services
               - Architecture overview
               - Cost estimates
               - Migration approach
               - Next steps
            
            Always end your final summary with "TERMINATE" to indicate completion.
            Be specific and provide actionable recommendations.
            """,
        )
        
        return planning_agent, pa, rpa
    
    def handle_user_input_request(self, prompt: str) -> str:
        """Handle user input requests from agents"""
        # Add a special message to trigger UI update for user input
        self.add_message_to_queue("user_input_request", prompt, "Assistant")
        
        # Wait for user response
        self.input_event.clear()
        self.input_event.wait()  # This will block until user provides input
        
        # Return the user's response
        response = self.user_response
        self.user_response = None
        
        return response or "continue"
    
    def provide_user_response(self, response: str):
        """Provide user response to the conversation"""
        self.user_response = response
        self.input_event.set()  # Signal that response is available
    
    def add_message_to_queue(self, msg_type: str, content: str, sender: str = "System"):
        """Add message to the queue for UI updates"""
        self.message_queue.put({
            "type": msg_type,
            "content": content,
            "sender": sender,
            "timestamp": time.time()
        })
    
    async def run_conversation(self, task: str, endpoint: str, api_key: str, model: str, api_version: str):
        """Run the multi-agent conversation"""
        try:
            self.add_message_to_queue("info", f"🔄 Initializing Azure Migration Assistant...")
            
            # Create model client
            self.model_client = await self.create_model_client(endpoint, api_key, model, api_version)
            
            # Initialize agents
            planning_agent, pa, rpa = await self.initialize_agents(self.model_client)
            
            self.add_message_to_queue("info", f"🤖 Agents ready. Analyzing your migration requirements...")
            
            # Create user proxy that can handle input requests
            user_proxy_agent = UserProxyAgent(
                name="UserProxyAgent",
                description="A user proxy agent for the Azure migration system",
                input_func=self.handle_user_input_request,  # Use the new input handler
            )
            
            # Setup termination condition
            text_mention_termination = TextMentionTermination("TERMINATE")
            
            # Create team
            selector_prompt = """You are selecting the next agent to speak in an Azure migration consultation.

            Available agents and their roles:
            {roles}

            Conversation history:
            {history}

            Select the most appropriate agent from {participants} to continue the conversation.
            - Start with PlanningAgent for task coordination
            - Use RequirementsParserAgent to analyze requirements and detect missing information
            - Use UserProxyAgent for user interaction if needed for missing information
            - Use PricingAgent for cost analysis after collection all the requirements

            PlanningAgent is the only agent that can call or relay messages through UserProxyAgent.
            
            Select only one agent name.
            """
            
            team = SelectorGroupChat(
                [planning_agent, rpa, pa, user_proxy_agent],
                model_client=self.model_client,
                termination_condition=text_mention_termination,
                selector_prompt=selector_prompt,
                allow_repeated_speaker=True,
                max_selector_attempts=2
            )
            
            self.add_message_to_queue("info", f"🧠 Running multi-agent analysis... This may take a moment.")
            
            # Run the conversation
            final_result = ""
            conversation_messages = []
            
            async for message in team.run_stream(task=task):
                logger.info(message)
                if hasattr(message, 'source') and hasattr(message, 'content'):
                    sender = message.source
                    content = str(message.content)
                    
                    # Store all messages for processing
                    conversation_messages.append({"sender": sender, "content": content})
                    
                    # Filter which messages to show to user
                    should_show = False
                    
                    # Show messages from PlanningAgent (orchestrator)
                    if sender == "PlanningAgent":
                        should_show = True
                    
                    # Show important messages from specialized agents
                    elif sender in ["requirements_parser_agent", "pricing_agent"]:
                        # Only show final analysis/results, not intermediate processing
                        # should_show = False
                        # if any(keyword in content.lower() for keyword in 
                        #        ["analysis", "summary", "recommendation", "estimate", "cost", "requirements"]):
                        #     should_show = True
                        # logger.debug(content)
                        if '<Request for the user>:' in content:
                        #     # This is a user input request, show it
                            should_show = True
                            # st.session_state.pending_user_input = True
                            # st.session_state.user_input_prompt = content.split('<Request for the user>:')[1].strip()
                    
                    # Never show UserProxyAgent messages (handled separately)
                    elif sender == "UserProxyAgent":
                        should_show = False
                    
                    # Show the message if it should be displayed
                    if should_show:
                        self.add_message_to_queue("agent", content, sender)
                    
                    # Check for termination
                    if "TERMINATE" in content:
                        # Find the final comprehensive response from PlanningAgent
                        for msg in reversed(conversation_messages):
                            if (msg["sender"] == "PlanningAgent" and 
                                len(msg["content"]) > 200 and  # Substantial content
                                any(keyword in msg["content"].lower() for keyword in 
                                    ["recommendation", "summary", "migration", "cost", "architecture"])):
                                final_result = msg["content"]
                                break
                        
                        if final_result:
                            # Clean up the final result by removing "TERMINATE"
                            final_result = final_result.replace("TERMINATE", "").strip()
                            self.add_message_to_queue("agent", final_result, "Migration Analysis")
                        
                        self.add_message_to_queue("info", "✅ Migration analysis completed!")
                        break
            
        except Exception as e:
            self.add_message_to_queue("error", f"Error during conversation: {str(e)}", "System")
        finally:
            if self.model_client:
                await self.model_client.close()
    
    def start_conversation_thread(self, task: str, endpoint: str, api_key: str, model: str, api_version: str):
        """Start conversation in a background thread"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.run_conversation(task, endpoint, api_key, model, api_version)
                )
            finally:
                loop.close()
        
        self.conversation_thread = threading.Thread(target=run_async, daemon=True)
        self.conversation_thread.start()

# Initialize conversation manager
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

# Handle template selection
if hasattr(st.session_state, 'selected_template') and st.session_state.selected_template:
    template_text = st.session_state.selected_template
    delattr(st.session_state, 'selected_template')
    
    # Auto-start conversation with template
    if config_valid and not st.session_state.conversation_active:
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
        if config_valid:
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
        else:
            st.error("⚠️ Please complete Azure OpenAI configuration in the sidebar first.")

# Status indicator
status_col1, status_col2 = st.columns([3, 1])

with status_col1:
    if st.session_state.get("pending_user_input", False):
        st.info("💬 Assistant is waiting for your response...")
    elif st.session_state.conversation_active:
        st.info("🔄 Analysis in progress...")
    elif not config_valid:
        st.warning("⚠️ Please complete Azure OpenAI configuration in the sidebar")

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


