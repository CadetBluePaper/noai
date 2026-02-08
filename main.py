import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.functions import schema_get_files, schema_get_file_content, schema_write_file, schema_run_python_file
from functions.functions import call_function

def main():
    #load environment variables from .env file
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    system_prompt = os.environ.get("SYSTEM_PROMPT")

    client = genai.Client(api_key=api_key)

    #set up argument parser
    parser = argparse.ArgumentParser(description="Hyperparameter Arguments")

    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate content for.")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose output.")

    args = parser.parse_args()

    #create messages list
    message_history = [types.Content(role="user", parts=[types.Part(text=args.prompt)])]

    #content_loop
    feedback_loop(client, message_history, args.verbose, system_prompt)

def feedback_loop(client, message_history, verbose, system_prompt, max_iterations=5):
    
    available_functions = types.Tool(
        function_declarations=[
            schema_get_files,
            schema_get_file_content,
            schema_write_file,
            schema_run_python_file
        ]
    )

    for _ in range(max_iterations):
        try: 
            #generate content
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=message_history,
                config=types.GenerateContentConfig(
                    tools=[available_functions], 
                    system_instruction=system_prompt
                )
            )

            if verbose: 
                print(f"Prompt tokens used: {response.usage_metadata.prompt_token_count}")
                print(f"Response tokens used: {response.usage_metadata.candidates_token_count}")

            #Add the models response to the message history
            for candidate in response.candidates:
                message_history.append(candidate.content)

            if response.text: 
                print(f"Final response: {response.text}")
                break
            
            #Function calls
            if response.function_calls:
                function_responses = []
                for function in response.function_calls:
                    result = call_function(function, verbose=verbose)
                    function_responses.append(result)


                    try:
                        call_response = result.parts[0].function_response.response
                        if verbose:
                            print(f" -> {call_response}")
                    except (AttributeError, IndexError):
                        print(" -> No response from function call.")

                for tool_content in function_responses:
                    message_history.append(tool_content)

        except Exception as e:
            print(f"Error during generation: {e}")
            break

    else:
        print("Max iterations reached without completion, exiting...")

if __name__ == "__main__":
    main()
                







