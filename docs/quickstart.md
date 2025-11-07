# Quick Start Guide

This guide will get you up and running with CellSem LLM Client in minutes.

## Basic Usage

### Creating Agents

The simplest way to get started is using the configuration utilities:

```python
from cellsem_llm_client import create_openai_agent, create_anthropic_agent

# Create agents with automatic environment configuration
openai_agent = create_openai_agent()
anthropic_agent = create_anthropic_agent()
```

### Simple Queries

```python
# Basic text generation
openai_response = openai_agent.query("Explain quantum computing in 50 words")
print(openai_response)

claude_response = anthropic_agent.query("What are the benefits of renewable energy?")
print(claude_response)
```

### Custom Configuration

```python
# Custom model and parameters
custom_agent = create_openai_agent(
    model="gpt-4",
    max_tokens=2000
)

response = custom_agent.query("Write a detailed analysis of machine learning")
```

## Working with System Messages

System messages help set the context and behavior of the LLM:

```python
# Using system messages for better control
agent = create_openai_agent()

response = agent.query(
    "What is 2+2?",
    system_message="You are a math tutor. Always show your work and explain step by step."
)
```

## Provider-Specific Features

### OpenAI Models

```python
from cellsem_llm_client import OpenAIAgent

# Using specific OpenAI models
gpt4_agent = OpenAIAgent(
    model="gpt-4",
    api_key="your-api-key",
    max_tokens=1500
)

# GPT-3.5 for faster, cheaper responses
gpt35_agent = OpenAIAgent(
    model="gpt-3.5-turbo",
    api_key="your-api-key"
)
```

### Anthropic Models

```python
from cellsem_llm_client import AnthropicAgent

# Using different Claude models
haiku_agent = AnthropicAgent(
    model="claude-3-haiku-20240307",  # Fast and economical
    api_key="your-api-key"
)

sonnet_agent = AnthropicAgent(
    model="claude-3-sonnet-20240229",  # Balanced performance
    api_key="your-api-key"
)
```

## Advanced Usage

### Direct LiteLLM Integration

For maximum flexibility, use the base LiteLLMAgent:

```python
from cellsem_llm_client import LiteLLMAgent

# Works with any LiteLLM-supported provider
agent = LiteLLMAgent(
    model="gpt-4",  # or "claude-3-sonnet-20240229", etc.
    api_key="your-api-key",
    max_tokens=1000
)

response = agent.query("Your prompt here")
```

### Environment-Based Configuration

The client automatically loads configuration from environment variables:

```python
import os
from cellsem_llm_client import create_litellm_agent

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Agent automatically uses the environment key
agent = create_litellm_agent(model="gpt-3.5-turbo")
```

## Best Practices

### Error Handling

```python
from cellsem_llm_client import create_openai_agent

try:
    agent = create_openai_agent()
    response = agent.query("Your prompt")
    print(response)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

### Cost Optimization

:::{tip}
- Use **GPT-3.5-turbo** or **Claude Haiku** for simple tasks
- Use **GPT-4** or **Claude Sonnet** for complex reasoning
- Set appropriate `max_tokens` to control costs
- Consider caching responses for repeated queries
:::

### Testing and Development

```python
# Use different models for development vs production
import os

if os.getenv("ENVIRONMENT") == "production":
    agent = create_openai_agent(model="gpt-4")
else:
    agent = create_openai_agent(model="gpt-3.5-turbo")  # Cheaper for testing
```

## Example Applications

### Content Generation

```python
agent = create_anthropic_agent()

blog_post = agent.query(
    "Write a 300-word blog post about sustainable technology",
    system_message="You are a technical writer specializing in environmental technology."
)
```

### Code Analysis

```python
agent = create_openai_agent(model="gpt-4")

code_review = agent.query(
    f"Review this Python code for best practices:\n\n{code}",
    system_message="You are a senior Python developer. Provide constructive feedback."
)
```

### Data Analysis

```python
agent = create_anthropic_agent()

analysis = agent.query(
    f"Analyze this dataset summary: {data_summary}",
    system_message="You are a data scientist. Provide insights and recommendations."
)
```

## Next Steps

- Explore the {doc}`api/index` for detailed API documentation
- Check out {doc}`contributing` for development guidelines
- See [ROADMAP.md](https://github.com/Cellular-Semantics/cellsem_llm_client/blob/main/ROADMAP.md) for upcoming features:
  - Token tracking and cost monitoring
  - JSON schema validation
  - File attachment support
  - AI-powered model recommendations