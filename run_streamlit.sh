#!/bin/bash

# Script to run the Azure Migration Assistant Streamlit app

echo "🚀 Starting Azure Migration Assistant..."
echo "📋 Make sure you have installed the dependencies:"
echo "   uv sync"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if streamlit is available in the environment
if ! uv run streamlit --help &> /dev/null; then
    echo "❌ Streamlit is not installed. Please run:"
    echo "   uv add streamlit"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Please create one with your Azure OpenAI credentials:"
    echo "   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here"
    echo "   AZURE_OPENAI_API_KEY=your_api_key_here"
    echo "   AZURE_OPENAI_MODEL=gpt-35-turbo"
    echo "   AZURE_OPENAI_API_VERSION=2025-01-01-preview"
    echo ""
    exit 1
else
    echo "✅ Found existing .env file"
fi

echo "🌐 Launching Streamlit app on http://localhost:8501"
echo "📚 Open your browser and navigate to the URL above"
echo ""

# Run the Streamlit app with uv
uv run streamlit run streamlit_app.py --server.port 8501 --server.address localhost
