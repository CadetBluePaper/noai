"""
Tool function declarations

This file provides the function declearations and schemas for
the agent to use
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# =========================
# Tool schemas
# =========================

SCHEMA_LIST_DIRECTORY = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in a directory with sizes and directory flags.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description="Target directory relative to the working directory.",
            )
        },
    ),
)

SCHEMA_READ_FILE = types.FunctionDeclaration(
    name="get_file_content",
    description="Returns the content of a file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file relative to the working directory.",
            )
        },
    ),
)

SCHEMA_WRITE_FILE = types.FunctionDeclaration(
    name="write_file",
    description="Writes content to a file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file relative to the working directory.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Content to write to the file.",
            ),
        },
    ),
)

SCHEMA_EXECUTE_PYTHON = types.FunctionDeclaration(
    name="run_python_file",
    description="Runs a Python file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Python file path relative to the working directory.",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Optional command-line arguments.",
            ),
        },
    ),
)

# =========================
# Tool implementations
# =========================

def list_directory_contents(
    working_directory: str,
    directory: str = ".",
) -> str:
    """List files and directories within a working directory."""
    base_dir = os.path.abspath(working_directory)
    target_dir = os.path.abspath(os.path.join(base_dir, directory))

    if os.path.commonpath([base_dir, target_dir]) != base_dir:
        return f"Error: {directory} is outside the working directory."

    if not os.path.isdir(target_dir):
        return f"Error: {directory} is not a valid directory."

    try:
        lines: List[str] = []
        for name in os.listdir(target_dir):
            path = os.path.join(target_dir, name)
            is_dir = os.path.isdir(path)
            size = 0 if is_dir else os.path.getsize(path)
            lines.append(
                f"{name}: file_size={size} bytes, is_dir={is_dir}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing directory: {exc}"


def read_file_content(
    working_directory: str,
    file_path: str,
) -> str:
    """Read and return file contents (truncated if needed)."""
    max_chars = int(os.environ.get("MAX_FILE_CHARS", "10000"))
    base_dir = os.path.abspath(working_directory)
    target_file = os.path.abspath(os.path.join(base_dir, file_path))

    if os.path.commonpath([base_dir, target_file]) != base_dir:
        return "Error: File is outside the working directory."

    if not os.path.isfile(target_file):
        return f"Error: File not found: {file_path}"

    try:
        with open(target_file, "r", encoding="utf-8") as file:
            content = file.read(max_chars)

        if os.path.getsize(target_file) > max_chars:
            content += f"\n...File truncated at {max_chars} characters."

        return content
    except Exception as exc:
        return f"Error reading file: {exc}"


def write_file_content(
    working_directory: str,
    file_path: str,
    content: str,
) -> str:
    """Write content to a file."""
    base_dir = os.path.abspath(working_directory)
    target_file = os.path.abspath(os.path.join(base_dir, file_path))

    if os.path.commonpath([base_dir, target_file]) != base_dir:
        return "Error: File path is outside the working directory."

    try:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}."
    except Exception as exc:
        return f"Error writing file: {exc}"


def execute_python_script(
    working_directory: str,
    file_path: str,
    args: Optional[List[str]] = None,
) -> str:
    """Execute a Python script."""
    args = args or []
    base_dir = os.path.abspath(working_directory)
    target_file = os.path.abspath(os.path.join(base_dir, file_path))

    if os.path.commonpath([base_dir, target_file]) != base_dir:
        return "Error: Script is outside the working directory."

    if not target_file.endswith(".py"):
        return "Error: Target file is not a Python script."

    try:
        result = subprocess.run(
            ["python", target_file, *args],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=base_dir,
        )

        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"Exit code: {result.returncode}")

        return "\n".join(output) or "Script executed successfully."
    except Exception as exc:
        return f"Error executing script: {exc}"


# =========================
# Dispatcher
# =========================

TOOL_FUNCTION_MAP: Dict[str, Any] = {
    "get_files_info": list_directory_contents,
    "get_file_content": read_file_content,
    "write_file": write_file_content,
    "run_python_file": execute_python_script,
}


def dispatch_tool_call(
    tool_call: Any,
    verbose: bool = False,
) -> types.Content:
    """Dispatch a tool call and return a Gemini-compatible response."""
    tool_name: str = tool_call.name
    args: Dict[str, Any] = dict(tool_call.args)
    args["working_directory"] = os.environ.get("WORKING_DIR", "")

    if verbose:
        print(f"Calling tool: {tool_name} with args: {args}")

    try:
        if tool_name not in TOOL_FUNCTION_MAP:
            raise KeyError(f"Unknown tool: {tool_name}")

        result = TOOL_FUNCTION_MAP[tool_name](**args)
        response = {"output": result}
    except Exception as exc:
        response = {"error": str(exc)}

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=tool_name,
                response=response,
            )
        ],
    )