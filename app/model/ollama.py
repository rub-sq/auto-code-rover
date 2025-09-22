"""
For models other than those from OpenAI, use LiteLLM if possible.
Create all models managed by Ollama here, since they need to talk to ollama server.
"""

import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Literal, cast

import ollama
import timeout_decorator
from ollama._types import Message, Options
from openai.types.chat import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.model import common
from app.model.common import Model


class OllamaModel(Model):
    """
    Base class for creating Singleton instances of Ollama models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(self, name: str):
        if self._initialized:
            return
        # local models are free
        super().__init__(name, 0.0, 0.0)
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()
        try:
            self.send_empty_request()
            print(f"Model {self.name} is up and running.")
        except timeout_decorator.TimeoutError as e:
            print(
                "Ollama server is taking too long (more than 2 mins) to respond. Please check whether it's running.",
                e,
            )
            sys.exit(1)

    @timeout_decorator.timeout(120)  # 2 min
    def send_empty_request(self):
        """
        Send an empty request to the model, for two purposes
        (1) check whether the model is up and running
        (2) preload the model for faster response time (models will be kept in memory for 5 mins after loaded)
        (see https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-pre-load-a-model-to-get-faster-response-times)
        """
        ollama.chat(model=self.name, messages=[])

    def check_api_key(self) -> str:
        return "No key required for local models."

    def extract_resp_content(
        self, chat_completion_message: ChatCompletionMessage
    ) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_completion_message.content
        if content is None:
            return ""
        else:
            return content

    def _clean_messages(self, messages: list[dict]) -> list[dict]:
        """
        Clean messages to prevent template token issues and ensure proper formatting.
        Also enhance API guidance for local models.
        """
        cleaned = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                # Remove problematic template tokens that can cause loops
                content = content.replace("[INST:src]", "")
                content = content.replace("[/INST]", "")
                content = content.replace("<s>", "")
                content = content.replace("</s>", "")
                
                import re
                content = re.sub(r'\s+', ' ', content).strip()
                if content:  # Only add non-empty messages
                    cleaned.append({
                        "role": msg.get("role", "user"),
                        "content": content
                    })
        return cleaned

    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ):
        stop_words = ["assistant", "\n\n \n\n", "[INST", "</s>", "[/INST]"]
        json_stop_words = deepcopy(stop_words)
        json_stop_words.append("```")
        json_stop_words.append(" " * 10)
        # FIXME: ignore tools field since we don't use tools now

        if temperature is None:
            temperature = common.MODEL_TEMP

        try:
            # Record start time for timing metrics
            start_time = time.time()
            
            # Clean and prepare messages to prevent template token issues
            cleaned_messages = self._clean_messages(messages)
            
            # build up options for ollama
            options = {
                "temperature": temperature,
                "top_p": top_p,
            }
            if response_format == "json_object":
                # additional instructions for json mode - but don't modify original messages
                json_messages = cleaned_messages.copy()
                json_instruction = {
                    "role": "user",
                    "content": "Please respond with valid JSON only. Stop your response after a valid JSON object is generated.",
                }
                json_messages.append(json_instruction)
                # give more stop words and lower max_token for json mode
                options.update({"stop": json_stop_words, "num_predict": 256})
                response = ollama.chat(
                    model=self.name,
                    messages=cast(list[Message], json_messages),
                    format="json",
                    options=cast(Options, options),
                    stream=False,
                )
            else:
                options.update({"stop": stop_words, "num_predict": 2048})
                response = ollama.chat(
                    model=self.name,
                    messages=cast(list[Message], cleaned_messages),
                    options=cast(Options, options),
                    stream=False,
                )

            if not isinstance(response, Mapping):
                print(f"Warning: Unexpected response type from Ollama: {type(response)}")
                return "", 0, 0, 0
                
            resp_msg = response.get("message", None)
            if resp_msg is None:
                print(f"Warning: No message in Ollama response: {response}")
                return "", 0, 0, 0

            # Extract metrics from Ollama response
            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            total_duration_ns = response.get("total_duration", 0)
            
            # Calculate cost (0 for local models) and log metrics
            cost = self.calc_cost(input_tokens, output_tokens)  # Will be 0 for local models
            
            # Update thread cost tracking for analysis compatibility
            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            content: str = resp_msg.get("content", "")
            
            # Additional cleaning of the response to prevent token issues
            if content:
                content = content.replace("[INST:src]", "").replace("[/INST]", "")
                import re
                content = re.sub(r'\[INST:src\]+', '', content)  # Remove any repeated tokens
                content = content.strip()
            
            return content, cost, input_tokens, output_tokens

        except Exception as e:
            print(f"Error in Ollama API call: {e}")
            print(f"Model: {self.name}, Messages count: {len(messages)}")
            # Don't re-raise for now to prevent retry loops, return empty response instead
            return "", 0, 0, 0


class Llama3_8B(OllamaModel):
    def __init__(self):
        super().__init__("llama3")
        self.note = "Llama3 8B model."


class Llama3_70B(OllamaModel):
    def __init__(self):
        super().__init__("llama3:70b")
        self.note = "Llama3 70B model."


class CodeLlama13B(OllamaModel):
    def __init__(self):
        super().__init__("codellama:13b")
        self.note = "CodeLlama 13B model optimized for code generation."


class Qwen14B(OllamaModel):
    def __init__(self):
        super().__init__("qwen:14b")
        self.note = "Qwen 14B model from Alibaba Cloud."


class CodeLlama7B(OllamaModel):
    def __init__(self):
        super().__init__("codellama:7b")
        self.note = "CodeLlama 7B model optimized for code generation."


class Qwen7B(OllamaModel):
    def __init__(self):
        super().__init__("qwen:7b")
        self.note = "Qwen 7B model from Alibaba Cloud."


class DeepSeekCoder67B(OllamaModel):
    def __init__(self):
        super().__init__("deepseek-coder:6.7b")
        self.note = "DeepSeek-Coder 6.7B model specialized for code understanding and generation."


class DeepSeekR17B(OllamaModel):
    def __init__(self):
        super().__init__("deepseek-r1:7b")
        self.note = "DeepSeek-R1 7B model with advanced reasoning capabilities."
