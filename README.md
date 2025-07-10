# Multi-agent samples for Azure migrations assistance 🚀

A multi-agent system built with Microsoft AutoGen for Azure cloud migration planning and pricing analysis. This tool helps organizations plan their Azure migrations by analyzing requirements and providing cost estimates.

## 🌟 Features

- **Multi-Agent Architecture**: Powered by Microsoft AutoGen with specialized agents for different tasks
- **Requirements Analysis**: Intelligent parsing of migration requirements and workload characteristics
- **Azure Pricing Integration**: Real-time pricing information from Azure Retail Prices API
- **Interactive Web Interface**: Streamlit-based UI for easy interaction with the agent system
- **Azure OpenAI Integration**: Uses Azure OpenAI for intelligent conversation and analysis

## 🏗️ Architecture

The system consists of specialized agents:

- **Requirements Parser Agent**: Extracts key requirements from user input including:
  - Workload type (web portal, API, HPC, batch, etc.)
  - Application architecture layers
  - Languages and frameworks
  - Database and storage types
  - Deployment model (IaaS, PaaS, SaaS, containers, serverless)

- **Pricing Agent**: Provides Azure service pricing information with:
  - Real-time pricing data from Azure Retail Prices API
  - Regional pricing comparisons
  - Multi-currency support

- **User Proxy Agent**: Handles user interaction and input coordination

## 📋 Prerequisites

- Python 3.13+
- Azure OpenAI API access
- UV package manager (recommended)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mas-dummy
   ```

2. **Install UV package manager** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_MODEL=gpt-35-turbo
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   ```

## 🚀 Usage

### Web Interface (Recommended)

1. **Start the Streamlit app**:
   ```bash
   ./run_streamlit.sh
   ```
   
   Or manually:
   ```bash
   uv run streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Begin your migration conversation** by describing your current infrastructure and migration goals

### Command Line Interface

Run the CLI version:
```bash
uv run python main.py
```

## 🔧 Configuration

The application uses the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_MODEL` | Model name to use | `gpt-35-turbo` |
| `AZURE_OPENAI_API_VERSION` | API version | `2025-01-01-preview` |

## 📁 Project Structure

```
mas-dummy/
├── agents/                 # Agent implementations
│   ├── pricing.py         # Azure pricing agent
│   └── requirements_parses.py  # Requirements parsing agent
├── utils/                 # Utility modules
│   └── ConversationManager.py  # Conversation management
├── logs/                  # Application logs
├── main.py               # CLI entry point
├── streamlit_app.py      # Web interface
├── run_streamlit.sh      # Startup script
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Ensure your Azure OpenAI credentials are correctly configured
2. Verify all dependencies are installed with `uv sync`
3. Open an issue in the repository for additional support

## 🎯 Example Use Cases

- **Legacy Application Migration**: Analyze existing on-premises applications for Azure migration
- **Cost Estimation**: Get accurate pricing estimates for Azure services
- **Architecture Planning**: Receive recommendations for Azure service selection
- **Multi-Cloud Strategy**: Compare Azure offerings with current cloud solutions

---

Built with ❤️ using Microsoft AutoGen and Azure OpenAI