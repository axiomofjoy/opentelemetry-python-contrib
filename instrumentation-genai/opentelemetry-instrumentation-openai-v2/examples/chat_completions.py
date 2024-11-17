import os

from openai import OpenAI

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
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
