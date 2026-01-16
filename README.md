# ShardGuard

ShardGuard is a proof-of-concept system designed to secure Model-Context Protocol (MCP) workflows in the presence
of untrusted LLMs. While LLMs offer powerful automation capabilities across sensitive user contexts—like email,
calendars, and financial services—they also pose significant risks, including prompt injection, data exfiltration,
and unauthorized tool usage. ShardGuard tackles this by isolating execution into fine-grained, privacy-preserving
steps coordinated by a trusted service. It leverages a trusted planning LLM to decompose tasks, uses opaque values
and sanitization functions to limit data exposure, and ensures least-privilege access to MCP tools. Unlike
traditional information flow control systems, ShardGuard avoids complex labeling and taint tracking, offering a more
practical yet effective path to safe LLM-driven automation.

> **⚠️ Note**: This is a basic prototype of [a MCP security system](https://docs.google.com/document/d/1fB_DedbmW5E7MQSgXd98iNj9aQfXtajNKSEtAGUzLYI/edit?tab=t.0).
> Don't use it for anything production. We're still testing things to see if this is a good approach!

## Features

- **Input Sanitization**: Automatically sanitizes user input with rich CLI logging
- **Multi-Provider LLM Integration**: Supports both local Ollama and remote LLM providers (Google Gemini)
- **Sensitive Data Masking**: Automatically identifies and masks sensitive information with reference placeholders
- **Flexible Model Support**: Works with various models including llama3.2, gemini-2.0-flash-exp, and more

## Quick Start

### Prerequisites

- Python 3.13+
- Poetry (for dependency management)
- Ollama (for local models) OR Google AI API key (for Gemini models)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ShardGuard
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:

   ```bash
   eval $(poetry env activate)
   ```

### Basic Usage

#### Using Ollama (Local Models)

1. **Start Ollama**:

   ```bash
   ollama serve
   ```

2. **Pull a model**:

   ```bash
   ollama pull llama3.2
   ```

3. **Run ShardGuard**:

   ```bash
   poetry run shardguard plan "Send an email to john@example.com about the meeting"
   ```

#### Using Google Gemini (Remote Models)

1. **Get a Gemini API key**:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create an API key

2. **Set your API key**:

   ```bash
   # Option 1: Create a .env file (recommended)
   cp .env.example .env
   # Edit .env and add your API key: GEMINI_API_KEY=your-api-key-here

   # Option 2: Set environment variable
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. **Run ShardGuard with Gemini**:

   ```bash
   poetry run shardguard plan "Send an email to john@example.com about the meeting" --provider gemini --model gemini-2.0-flash-exp
   ```

#### Advanced Usage

```bash
# List available MCP tools with Ollama
shardguard list-tools --provider ollama --model llama3.2

# List available MCP tools with Gemini
shardguard list-tools --provider gemini --model gemini-2.0-flash-exp

# Use custom Ollama URL
shardguard plan "Your prompt" --provider ollama --ollama-url http://localhost:11434

# Pass Gemini API key directly (not recommended for security)
shardguard plan "Your prompt" --provider gemini --gemini-api-key "your-key"
```

## Configuration

### Available Models

| Provider    | Models                                                                      | Notes                     |
|-------------|-----------------------------------------------------------------------------|---------------------------|
| **Gemini**  | `gemini-2.0-flash-exp` (default)<br>`gemini-1.5-pro`<br>`gemini-1.5-flash`  | Remote, requires API key  |
| **Ollama**  | `llama3.2` (default)<br>`llama3.1`<br>`codellama`<br>`mistral`              | Local, free               |

## Development

### Setup

```bash
# Install dependencies
poetry install -E dev

# Install pre-commit hooks (required for each developer)
poetry run pre-commit install

# Run tests
poetry run pytest

# Run linting and formatting
poetry run ruff check .
poetry run ruff format .
```

> **Note**: Each developer must run `poetry run pre-commit install` to enable automatic code quality checks on commit.
> The hooks will then run automatically before each commit to ensure code quality.

### Project Structure

```text
src/shardguard/
├── cli.py              # CLI interface with multi-provider support
├── core/
│   ├── coordination.py # Main service
│   ├── llm_providers.py # LLM provider implementations (Ollama, Gemini)
│   ├── mcp_integration.py # MCP integration with provider support
│   ├── models.py       # Data models
│   ├── planning.py     # Planning LLM implementations
│   └── prompts.py      # Prompt templates
└── utils/
    └── logging.py      # Logging utilities
```

### Example Output

```json
{
  "original_prompt": "Setup server with password [PASSWORD]",
  "sub_prompts": [
    {
      "id": 1,
      "content": "Configure server infrastructure",
      "opaque_values": {}
    },
    {
      "id": 2,
      "content": "Setup admin access with [PASSWORD]",
      "opaque_values": {
        "[PASSWORD]": "secret123"
      }
    }
  ]
}
```

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.

## Security

This project deals with sensitive data processing. Please review the security considerations in `SECURITY.md`
before using in any production-like environment.
