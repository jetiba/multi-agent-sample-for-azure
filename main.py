from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents.pricing import PricingAgent
from agents.weather import WeatherAgent
import asyncio
from dotenv import load_dotenv
import os

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
    await Console(pa.run_stream(task="What is the pricing for Container Apps in westeurope?"))
    # Close the connection to the model client.
    await model_client.close()


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
if __name__ == "__main__":

    asyncio.run(main())