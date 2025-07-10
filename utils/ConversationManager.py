import asyncio
import queue
import threading
import time
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents.pricing import PricingAgent
from agents.requirements_parses import RequirementsParserAgent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
import logging
from autogen_core import TRACE_LOGGER_NAME

class ConversationManager:
    """Manages the multi-agent conversation"""
    
    def __init__(self):
        self.message_queue = queue.Queue()
        self.conversation_thread = None
        self.model_client = None
        self.input_event = threading.Event()
        self.user_response = None
        
        # Logging for agents interaction
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(TRACE_LOGGER_NAME)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG)
    
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
            You are an Azure Migration Planning Agent. 
            Your role is to talk with the user, collect the requirements and coordinate a comprehensive migration analysis.

            Your team consists of:
            - RequirementsParserAgent: Analyzes, extracts and generate request for missing migration requirements
            - PricingAgent: Provides Azure service pricing and cost analysis
            - UserProxyAgent: Acts as a proxy for the user, providing input and feedback to the team

            Your responsibilities:
            - Manage the overall plan for solving the user's question.
            - Decide when more information is needed from the user.
            - Route any requests for user feedback or clarification through the UserProxyAgent.

            Strict rules:
            - You are the only agent that can call or relay messages through the UserProxyAgent.
            - When sending a request for user input, send **only ONE request at a time** and wait for the user response before proceeding.
            - Do not batch multiple questions into a single user request.
            - Do not add XML tags or special formatting when communicating with the user.

            Process:
            1. Start by having RequirementsParserAgent analyze the user's requirements.
            2. If RequirementsParserAgent generates requests for missing information, ask them **one by one** via UserProxyAgent, waiting for each response before proceeding to the next.
            3. Once all requirements are gathered, ask PricingAgent for relevant pricing information.
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

            Strict process guidelines:
            - Always start with PlanningAgent for task coordination and user interaction.
            - PlanningAgent manages all user interactions and decides when to involve RequirementsParserAgent or PricingAgent.
            - RequirementsParserAgent is used only when PlanningAgent requests requirement analysis.
            - If RequirementsParserAgent identifies missing information, PlanningAgent will sequentially request user input via UserProxyAgent, asking one question at a time.
            - PricingAgent is used only after PlanningAgent confirms all requirements are collected.
            - PlanningAgent is the only agent that can call or relay messages through UserProxyAgent.

            Selection rules:
            - Select only ONE agent.
            - Do NOT select multiple agents or provide explanations. Return only the agent name.
            """

            self.logger.info(selector_prompt)
            
            team = SelectorGroupChat(
                [planning_agent, rpa, pa, user_proxy_agent],
                model_client=self.model_client,
                termination_condition=text_mention_termination,
                selector_prompt=selector_prompt,
                allow_repeated_speaker=False,
                max_selector_attempts=2
            )
            
            self.add_message_to_queue("info", f"🧠 Running multi-agent analysis... This may take a moment.")
            
            # Run the conversation
            final_result = ""
            conversation_messages = []
            
            async for message in team.run_stream(task=task):
                self.logger.info(message)
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
                        should_show = False
                        # if any(keyword in content.lower() for keyword in 
                        #        ["analysis", "summary", "recommendation", "estimate", "cost", "requirements"]):
                        #     should_show = True
                        # logger.debug(content)
                        # if '<Request for the user>:' in content:
                        # #     # This is a user input request, show it
                        #     should_show = True
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
