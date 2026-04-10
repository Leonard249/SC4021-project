#!/usr/bin/env python3
"""Quick smoke test for the stateless OpenAI-compatible prompt setup."""

from openai_headless import create_stateless_completion


PROMPT = "Summarize the instruction you received in GEMINI.md in one paragraph."


def main():
    print(create_stateless_completion(PROMPT))


if __name__ == "__main__":
    main()
