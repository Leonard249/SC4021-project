#!/usr/bin/env python3
"""Stateless OpenAI-compatible chat helper for the data cleaning pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
from openai import APITimeoutError, OpenAI


BASE = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = BASE / "GEMINI.md"
DEFAULT_BASE_URL = (
    os.environ.get("HEADLESS_OPENAI_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "http://localhost:8002/v1"
)
DEFAULT_API_KEY = (
    os.environ.get("HEADLESS_OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "dummy"
)
DEFAULT_MODEL = (
    os.environ.get("HEADLESS_OPENAI_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or os.environ.get("GEMINI_MODEL")
    or "google/gemma-4-31b-it"
)


def load_system_prompt(path: str | os.PathLike[str] = DEFAULT_SYSTEM_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def build_messages(
    user_prompt: str,
    system_prompt_path: str | os.PathLike[str] = DEFAULT_SYSTEM_PROMPT_PATH,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt(system_prompt_path)},
        {"role": "user", "content": user_prompt},
    ]


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for chunk in content:
        if isinstance(chunk, str):
            parts.append(chunk)
            continue
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def build_request_timeout(timeout_seconds: int) -> httpx.Timeout:
    # For streaming responses, read timeout acts like an inactivity timeout.
    return httpx.Timeout(
        timeout=None,
        connect=min(float(timeout_seconds), 30.0),
        read=float(timeout_seconds),
        write=float(timeout_seconds),
        pool=float(timeout_seconds),
    )


def create_stateless_completion(
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout_seconds: int = 180,
    system_prompt_path: str | os.PathLike[str] = DEFAULT_SYSTEM_PROMPT_PATH,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        response = client.with_options(
            timeout=build_request_timeout(timeout_seconds)
        ).chat.completions.create(
            model=model,
            messages=build_messages(user_prompt, system_prompt_path=system_prompt_path),
            temperature=0,
            stream=True,
        )
        content_parts = []
        total_chars = 0
        stream_started = False

        for chunk in response:
            if not stream_started:
                stream_started = True
                if on_stream_event is not None:
                    on_stream_event({"type": "stream_started"})

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta_text = flatten_message_content(choices[0].delta.content)
            if not delta_text:
                continue

            content_parts.append(delta_text)
            total_chars += len(delta_text)
            if on_stream_event is not None:
                on_stream_event({
                    "type": "delta",
                    "delta_chars": len(delta_text),
                    "total_chars": total_chars,
                })
    except (APITimeoutError, httpx.TimeoutException) as exc:
        raise TimeoutError(
            f"OpenAI-compatible endpoint timed out after {timeout_seconds}s without stream activity."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
            raise TimeoutError(
                f"OpenAI-compatible endpoint timed out after {timeout_seconds}s without stream activity."
            ) from exc
        raise

    if on_stream_event is not None:
        on_stream_event({
            "type": "stream_finished",
            "total_chars": total_chars,
        })

    content = "".join(content_parts)
    if not content.strip():
        raise ValueError("OpenAI-compatible endpoint returned an empty response.")
    return content


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--prompt", required=True, help="Single prompt to send.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the endpoint. For vLLM-style local endpoints, a placeholder like 'dummy' is fine.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Stream inactivity timeout for the request in seconds.",
    )
    parser.add_argument(
        "--system-prompt-path",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to the system instruction file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        content = create_stateless_completion(
            args.prompt,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            system_prompt_path=args.system_prompt_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(content)


if __name__ == "__main__":
    main()
