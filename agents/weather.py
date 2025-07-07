from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import ChatCompletionClient



class WeatherAgent():
    """A specialized weather agent that can provide weather information."""

    # Define a simple function tool that the agent can use.
    # For this example, we use a fake weather tool for demonstration purposes.
    async def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 73 degrees and Sunny."

    def initialize(self, model_client: ChatCompletionClient) -> AssistantAgent:
        # Define an AssistantAgent with the model, tool, system message, and reflection enabled.
        # The system message instructs the agent via natural language.
        agent = AssistantAgent(
            name="weather_agent",
            model_client=model_client,
            tools=[self.get_weather],
            system_message="You are a helpful assistant.",
            reflect_on_tool_use=True,
            model_client_stream=True,  # Enable streaming tokens from the model client.
        )

        return agent