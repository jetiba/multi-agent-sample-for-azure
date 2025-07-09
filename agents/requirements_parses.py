from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

class RequirementsParserAgent():
    """An agent specialized in parsing the key requirements from an input text."""

    def initialize(self, model_client: ChatCompletionClient) -> AssistantAgent:
        # Define an AssistantAgent with the model, tool, system message, and reflection enabled.
        # The system message instructs the agent via natural language.
        agent = AssistantAgent(
            name="requirements_parser_agent",
            model_client=model_client,
            system_message= """You are an agent specialized in understanding the requirements for migrating or implementing solutions on Azure.
            Starting from the user inout you have to extract the key requirement from the input text.
            It will return the following information:
            - Workload type (web portal, API, hpc, batch, ...)
            - Application architecture layers
            - Languages and frameworks if present
            - Database and storage types if present
            - Deployment model (IaaS, PaaS, SaaS, containers, serverless, ...)

            If some requirements are not present in the input text, ask user for feedback.
            """,
            model_client_stream=True,  # Enable streaming tokens from the model client.
        )

        return agent