uv run pytest tests/test_*.py -v -n auto --tb=auto
# can do specific model provider with --model-provider=gemini
# can specify models (comma-delimited for multiple) with:
#   --openai-model=gpt-4o,gpt-5
#   --anthropic-model=claude-3-5-sonnet-20241022,claude-opus-4-20250514
#   --gemini-model=gemini-2.0-flash-exp,gemini-2.5-flash
