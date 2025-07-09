from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents.pricing import PricingAgent
from agents.requirements_parses import RequirementsParserAgent
import asyncio
from dotenv import load_dotenv
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
import logging
from autogen_core import TRACE_LOGGER_NAME

load_dotenv()


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = AzureOpenAIChatCompletionClient(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    # wa = WeatherAgent().initialize(model_client=model_client)
    # await Console(wa.run_stream(task="What is the weather in New York?"))
    # # Close the connection to the model client.
    # await model_client.close()

    user_proxy_agent = UserProxyAgent(
        name="UserProxyAgent",
        description="An agent that acts as a proxy for the user, providing input to the team.",
        input_func=input,
    )

    pa = PricingAgent().initialize(model_client=model_client)
    rpa = RequirementsParserAgent().initialize(model_client=model_client)

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            RequirementsParserAgent: Provides requirements from user input.
            PricingAgent: Provides pricing information for Azure services.
            UserProxyAgent: Acts as a proxy for the user, providing input and feedback to the team.

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    text_mention_termination = TextMentionTermination("TERMINATE")
    termination = text_mention_termination

    team = SelectorGroupChat(
        [planning_agent, rpa, pa, user_proxy_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
        max_selector_attempts=1
    )

    task="The customer needs to migrate on Azure a web portal, developed as a 3-tier application, with a PostgreSQL database, and a Redis cache. The application is developed in Python using the Django framework. The customer wants to use PaaS services as much as possible."

    # await Console(team.run_stream(task=task))

    async for message in team.run_stream(task=task):
        logger.error(message)
        logger.error(message.content)
        if message.source in ["RequirementsParserAgent"]:
            if '<Request for the user>:' in message.content:
                logger.info(message.content)

    # Close the connection to the model client.
    await model_client.close()


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
if __name__ == "__main__":

    asyncio.run(main())