import os
import subprocess
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
max_chars = int(os.environ.get("FILE_CHARACTER_LIMIT"))

schema_get_files = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in directory along with their sizes and whether they are directories.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description="The directory to list files from, relative to the working directory."
            )
        }
    )
)

def get_files(work_directory, directory="."):
    full_path = os.path.abspath(os.path.join(work_directory, directory))
    abs_work_directory = os.path.abspath(work_directory)

    if os.path.commonpath([full_path, abs_work_directory]) != abs_work_directory:
        return (f"Error: {directory} is outside the work directory")

    if not os.path.isdir(full_path):
        return (f"Error: {directory} is not a valid directory")
    
    try:
        items = os.listdir(full_path)
        result = ""

        for item in items:
            item_path = os.path.join(full_path, item)
            if os.path.isfile(item_path):
                result += f"{item}: file_size: {os.path.getsize(item_path)} bytes, is_dir=False\n"
            elif os.path.isdir(item_path):
                result += f"{item}: file_size: {os.path.getsize(item_path)} bytes, is_dir=True\n"
            else:
                result += f"{item}: file_size: {os.path.getsize(item_path)} bytes, is_dir=Unknown\n"
        
        return result.strip()
    except Exception as e:
        return (f"Error listing files: {e}")
    
schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description="Returns the content of a file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the file to read, relative to the working directory."
            )
        }
    )
)
    
def get_file_content(work_directory, file_path):
    abs_work_directory = os.path.abspath(work_directory)
    abs_file_path = os.path.abspath(os.path.join(work_directory, file_path))

    if os.path.commonpath([abs_file_path, abs_work_directory]) != abs_work_directory:
        return (f"Error: Cannot read {file_path} because it is outside the work directory")
    
    if not os.path.isfile(abs_file_path):
        return (f"File not found or is not a normal file: {file_path}")
    
    try:
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_chars)
            if os.path.getsize(abs_file_path) > max_chars:
                content += f"...File {file_path} truncated at {max_chars} characters."

            return content
            
    except Exception as e:
        return (f"Error reading file {file_path}: {e}")
    
schema_write_file = types.FunctionDeclaration(
    name="write_file",
    description="Writes content to a python file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the file to write, relative to the working directory."
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="The content to write to the file."
            )
        }
    )
)
    
def write_file(work_directory, file_path, content):
    abs_work_directory = os.path.abspath(work_directory)
    abs_file_path = os.path.abspath(os.path.join(work_directory, file_path))

    if os.path.commonpath([abs_file_path, abs_work_directory]) != abs_work_directory:
        return (f"Error: Cannot write to {file_path} because it is outside the work directory")
    
    if not os.path.exists(abs_file_path):
        try:
            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        except Exception as e:
            return (f"Error: creating directory {e}")
    if os.path.exists(abs_file_path) and os.path.isdir(abs_file_path):
        return (f"Error: {file_path} is a directory, not a file")

    try:
        with open(abs_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return (f"Successfully wrote to file: {file_path}, {len(content)} characters.")
    except Exception as e:
        return (f"Error writing to file {file_path}: {e}")

schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Runs a Python file.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the Python file to run, relative to the working directory."
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                description="Optional arguments to pass to the Python file.",
                items=types.Schema(
                    type=types.Type.STRING
                    )
            )
        }
    )
)

def run_python_file(work_directory, file_path, args=None):
    if args is None:
        args = []
    abs_work_directory = os.path.abspath(work_directory)
    abs_file_path = os.path.abspath(os.path.join(work_directory, file_path))

    if os.path.commonpath([abs_file_path, abs_work_directory]) != abs_work_directory:

        return (f"Error: Cannot run {file_path} because it is outside the work directory")
    
    if not os.path.exists(abs_file_path):
        return (f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)

    if ext != '.py':
        return (f"Error: {file_path} is not a Python file")
    
    try:
        command = ["python", abs_file_path]

        if args:
            command.extend(args)

        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=30, 
            cwd=abs_work_directory)
        
        output = []

        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"Process exited with code {result.returncode}")
        return "\n".join(output) if output else "Python file executed successfully with no output."
    except Exception as e:
        return (f"Error running python file {file_path}: {e}")

#call the function and return the result as a content object
def call_function(function_call_part, verbose=False):
    if verbose:
        print(f"Calling function: {function_call_part.name} with args: {function_call_part.args}")
    else:
        print(f"Calling function: {function_call_part.name}")

    function_map = {
        "get_files_info": get_files,
        "get_file_content": get_file_content,
        "write_file": write_file,
        "run_python_file": run_python_file
    }

    function_name = function_call_part.name
    if function_name not in function_map:
        return types.Content(
            role="function",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Unknown function: {function_name}"}
                )
            ]
        )
    
    args = dict(function_call_part.args)
    args["work_directory"] = os.environ.get("WORKING_DIR")
    function_result = function_map[function_name](**args)

    if isinstance(function_result, str):
        function_result = {"output": function_result}
    elif not isinstance(function_result, dict):
        function_result = {"output": str(function_result)}

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response=function_result
            )
        ]
    )
