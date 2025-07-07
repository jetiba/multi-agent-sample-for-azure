from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents.pricing import PricingAgent
from agents.weather import WeatherAgent
import asyncio
from dotenv import load_dotenv
import os

from typing import List, Sequence

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

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
    pa = PricingAgent().initialize(model_client=model_client)

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            PricingAgent: Provides pricing information for Azure services

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".

        A task might require multiple runs with different parameters (e.g. due to pagination).
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
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination


    # await Console(pa.run_stream(task="What is the pricing for Standard E1208 family Virtual Machines in westeurope?"))

    team = SelectorGroupChat(
        [planning_agent, pa],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
        max_selector_attempts=1
    )

    await Console(team.run_stream(task="What is the pricing for Standard E1208 family Virtual Machines in westeurope?"))

   
    # Close the connection to the model client.
    await model_client.close()


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
if __name__ == "__main__":

    asyncio.run(main())