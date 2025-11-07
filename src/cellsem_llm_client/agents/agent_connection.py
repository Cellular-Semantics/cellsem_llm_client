"""Agent connection classes for LLM interactions."""

from abc import ABC, abstractmethod

from litellm import completion


class AgentConnection(ABC):
    """Abstract base class for LLM agent connections."""

    @abstractmethod
    def query(self, message: str, system_message: str | None = None) -> str:
        """Send a query to the LLM and return the response.

        Args:
            message: The user message to send
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        pass


class LiteLLMAgent(AgentConnection):
    """LLM agent using LiteLLM for multi-provider support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize the LiteLLM agent.

        Args:
            model: The model name (e.g., 'gpt-3.5-turbo', 'claude-3-haiku-20240307')
            api_key: The API key for authentication
            max_tokens: Maximum tokens for response (default: 1000)

        Raises:
            ValueError: If model or api_key is None
        """
        if model is None:
            raise ValueError("Model is required")
        if api_key is None:
            raise ValueError("API key is required")

        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def query(self, message: str, system_message: str | None = None) -> str:
        """Send a query to the LLM using LiteLLM.

        Args:
            message: The user message to send
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": message})

        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        return str(response.choices[0].message.content)


class OpenAIAgent(LiteLLMAgent):
    """Convenience class for OpenAI models with sensible defaults."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize OpenAI agent with default model.

        Args:
            model: OpenAI model name (default: 'gpt-3.5-turbo')
            api_key: OpenAI API key
            max_tokens: Maximum tokens for response
        """
        super().__init__(model=model, api_key=api_key, max_tokens=max_tokens)


class AnthropicAgent(LiteLLMAgent):
    """Convenience class for Anthropic models with sensible defaults."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize Anthropic agent with default model.

        Args:
            model: Anthropic model name (default: 'claude-3-haiku-20240307')
            api_key: Anthropic API key
            max_tokens: Maximum tokens for response
        """
        super().__init__(model=model, api_key=api_key, max_tokens=max_tokens)
