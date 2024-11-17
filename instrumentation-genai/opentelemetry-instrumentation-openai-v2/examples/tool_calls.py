import os
from typing import cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "True"
logger_provider = LoggerProvider()
logger_provider.add_log_record_processor(
    SimpleLogRecordProcessor(ConsoleLogExporter())
)
event_logger_provider = EventLoggerProvider(logger_provider=logger_provider)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider,
    event_logger_provider=event_logger_provider,
)


if __name__ == "__main__":
    client = OpenAI()
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="What's the weather like in San Francisco?",
        )
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "finds the weather for a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to find the weather for, e.g. 'London'",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
        ],
        messages=messages,
    )
    message = response.choices[0].message
    assert (tool_calls := message.tool_calls)
    tool_call_id = tool_calls[0].id
    messages.append(
        cast(ChatCompletionAssistantMessageParam, message.to_dict())
    )
    messages.append(
        ChatCompletionToolMessageParam(
            content="sunny", role="tool", tool_call_id=tool_call_id
        ),
    )
    client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
