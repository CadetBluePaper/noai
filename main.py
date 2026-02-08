"""
Main file for the Gemini agent appplication.

This file sets up the argument parser, loads environment variables,
and sets up the main agent loop that interacts with the Gemini API, 
handles function calls, and manages the conversation history.

Important Note:

DO NOT use this code on your own machine, this project was meant for me to 
learn about the Gemini API and AI agents

It does not have the necessary security measures in place to be safely run
on a local machine (I almost nuked my computer lol). If you do want to run this
just make sure to specify the directory correctly in the .env file. 

Author: CadetBluePaper
"""
import os
import argparse
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from functions.functions import (
    SCHEMA_LIST_DIRECTORY,
    SCHEMA_READ_FILE,
    SCHEMA_WRITE_FILE,
    SCHEMA_EXECUTE_PYTHON,
    dispatch_tool_call,
)

MODEL_NAME = "gemini-2.5-flash"
MAX_ITERATIONS = 20


def main() -> None:
    """Entry point for the Gemini agent application."""
    load_dotenv()

    api_key: str | None = os.environ.get("GEMINI_API_KEY")
    system_instruction: str | None = os.environ.get("SYSTEM_PROMPT")

    if not api_key or not system_instruction:
        raise RuntimeError(
            "Missing required environment variables: "
            "GEMINI_API_KEY and/or SYSTEM_PROMPT"
        )

    client: genai.Client = genai.Client(api_key=api_key)

    parser = argparse.ArgumentParser(description="Hyperparameter Arguments")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to generate content for.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    cli_args = parser.parse_args()

    conversation_history: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=cli_args.prompt)],
        )
    ]

    run_agent_loop(
        client=client,
        conversation_history=conversation_history,
        verbose=cli_args.verbose,
        system_instruction=system_instruction,
    )


def run_agent_loop(
    client: genai.Client,
    conversation_history: List[types.Content],
    verbose: bool,
    system_instruction: str,
    max_iterations: int = MAX_ITERATIONS,
) -> None:
    """
    Main feedback loop for the agent.

    Args:
        client: Gemini API client.
        conversation_history: Accumulated conversation history.
        verbose: Enable verbose output.
        system_instruction: System prompt for the model.
        max_iterations: Maximum number of loop iterations.
    """
    available_tools = types.Tool(
        function_declarations=[
            SCHEMA_LIST_DIRECTORY,
            SCHEMA_READ_FILE,
            SCHEMA_WRITE_FILE,
            SCHEMA_EXECUTE_PYTHON,
        ]
    )

    for _ in range(max_iterations):
        try:
            model_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    tools=[available_tools],
                    system_instruction=system_instruction,
                ),
            )

            if verbose and model_response.usage_metadata:
                print(
                    f"Prompt tokens used: "
                    f"{model_response.usage_metadata.prompt_token_count}"
                )
                print(
                    f"Response tokens used: "
                    f"{model_response.usage_metadata.candidates_token_count}"
                )

            for candidate in model_response.candidates:
                conversation_history.append(candidate.content)

            if model_response.text:
                print(f"Final response: {model_response.text}")
                break

            if model_response.function_calls:
                for function_call in model_response.function_calls:
                    tool_response = dispatch_tool_call(
                        function_call,
                        verbose=verbose,
                    )
                    conversation_history.append(tool_response)

        except Exception as exc:
            print(f"Error during generation: {exc}")
            break
    else:
        print("Max iterations reached without completion, exiting...")

        
if __name__ == "__main__":
    main()