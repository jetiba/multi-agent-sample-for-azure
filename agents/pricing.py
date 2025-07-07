from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
import requests
import json


async def list_service_names() -> list[str]:
    """List the names of Azure services."""
    # curl -s "https://prices.azure.com/api/retail/prices?\$top=1000" | jq -r '.Items[].serviceName' | sort | uniq
    try:
        # Make the API request
        response = requests.get("https://prices.azure.com/api/retail/prices?$top=1000")
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract service names, sort and get unique values
        service_names = sorted(set(item['serviceName'] for item in data.get('Items', [])))
        
        return service_names
    except requests.RequestException as e:
        # Fallback to static list if API call fails
        raise Exception(f"Failed to fetch service names: {e}")

async def get_pricing(service_name: str, arm_region_name: str, currency_code: str) -> list[dict]:
    """Get the pricing for a given Azure service with optional region and currency filters.
    
    Args:
        service_name: The name of the Azure service to get pricing for
        arm_region_name: Optional ARM region name to filter by (e.g., 'eastus', 'westus2'). Default is 'westeurope'.
        currency_code: Optional currency code to filter by (e.g., 'USD', 'EUR'). Default is 'USD'.
        
    Returns:
        List of pricing information dictionaries containing skuName, unitPrice, unitOfMeasure and currency
    """
    try:
        # Build the filter query dynamically
        filters = [f"serviceName eq '{service_name}'"]
        
        if arm_region_name:
            filters.append(f"armRegionName eq '{arm_region_name}'")
        
        if currency_code:
            filters.append(f"currencyCode eq '{currency_code}'")
        
        # Combine filters with 'and' operator
        filter_query = " and ".join(filters)
        
        # Make the API request with dynamic filters
        response = requests.get(f"https://prices.azure.com/api/retail/prices?$filter={filter_query}")
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract relevant pricing information
        # pricing_info = [
        #     {
        #         "skuName": item['skuName'],
        #         "unitPrice": item['unitPrice'],
        #         "currency": item['currency'],
        #         "armRegionName": item.get('armRegionName', 'N/A'),
        #         "unitOfMeasure": item.get('unitOfMeasure', 'N/A'),
        #     }
        #     for item in data.get('Items', [])
        # ]
                    
        # return pricing_info
        
        test = data.get('Items', [])
        if not test:
            raise ValueError(f"No pricing data found for service '{service_name}' with region '{arm_region_name}' and currency '{currency_code}'")
        return test
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch pricing for {service_name}: {e}")


class PricingAgent():
    """A specialized Azure pricing agent that can provide pricing information."""

    def initialize(self, model_client: ChatCompletionClient) -> AssistantAgent:
        # Define an AssistantAgent with the model, tool, system message, and reflection enabled.
        # The system message instructs the agent via natural language.
        agent = AssistantAgent(
            name="pricing_agent",
            model_client=model_client,
            tools=[get_pricing, list_service_names],
            system_message="You are a Azure Pricing assistant.",
            reflect_on_tool_use=True,
            model_client_stream=True,  # Enable streaming tokens from the model client.
            max_tool_iterations=5,
        )

        return agent