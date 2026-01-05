
"""
File operation and system tools for PUding Agent.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from .utils import normalize_path, is_text_file, should_exclude_file
from .config import MAX_FILE_SIZE

def read_local_file(file_path: str) -> Dict[str, Any]:
    """Read content of a single file."""
    try:
        path = normalize_path(file_path)
        
        if not path.exists():
            return {"error": f"File '{file_path}' does not exist"}
        
        if not path.is_file():
            return {"error": f"'{file_path}' is not a file"}
        
        if path.stat().st_size > MAX_FILE_SIZE:
            return {"error": f"File '{file_path}' is too large (max {MAX_FILE_SIZE} bytes)"}
        
        if not is_text_file(path):
            return {"error": f"File '{file_path}' appears to be binary or non-textual"}
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "file_path": str(path),
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        return {"error": f"Failed to read '{file_path}': {str(e)}"}

def read_multiple_files(file_paths: List[str]) -> Dict[str, Any]:
    """Read contents of multiple files."""
    results = {}
    errors = []
    
    for file_path in file_paths:
        result = read_local_file(file_path)
        if "error" in result:
            errors.append(f"'{file_path}': {result['error']}")
        else:
            results[file_path] = result
    
    return {
        "success": len(errors) == 0,
        "files": results,
        "errors": errors
    }

def create_file(file_path: str, content: str) -> Dict[str, Any]:
    """Create a new file or overwrite an existing one."""
    try:
        path = normalize_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": str(path),
            "size": len(content),
            "message": f"File '{file_path}' created successfully"
        }
    except Exception as e:
        return {"error": f"Failed to create '{file_path}': {str(e)}"}

def create_multiple_files(files: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create multiple files."""
    results = {}
    errors = []
    
    for file_info in files:
        if "path" not in file_info or "content" not in file_info:
            errors.append("File info must contain 'path' and 'content' keys")
            continue
        
        result = create_file(file_info["path"], file_info["content"])
        if "error" in result:
            errors.append(f"'{file_info['path']}': {result['error']}")
        else:
            results[file_info["path"]] = result
    
    return {
        "success": len(errors) == 0,
        "files": results,
        "errors": errors
    }

def edit_file(file_path: str, original_snippet: str, new_snippet: str) -> Dict[str, Any]:
    """Replace a specific original_snippet with new_snippet in a file."""
    try:
        path = normalize_path(file_path)
        
        if not path.exists():
            return {"error": f"File '{file_path}' does not exist"}
        
        # Read current content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if original_snippet not in content:
            return {"error": f"Original snippet not found in '{file_path}'"}
        
        # Replace the snippet
        # Use replace(..., 1) to replace only the first occurrence for safety
        new_content = content.replace(original_snippet, new_snippet, 1) 
        
        # Write back to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            "success": True,
            "file_path": str(path),
            "message": f"File '{file_path}' edited successfully",
            "changes": {
                "original_length": len(content),
                "new_length": len(new_content),
                "diff": len(new_content) - len(content)
            }
        }
    except Exception as e:
        return {"error": f"Failed to edit '{file_path}': {str(e)}"}

def run_command(command: str) -> Dict[str, Any]:
    """Run a shell command."""
    try:
        if os.name == 'nt':
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True,
                text=True,
                timeout=120
            )
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
        
        return {
            "success": result.returncode == 0,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command '{command}' timed out"}
    except Exception as e:
        return {"error": f"Failed to run command '{command}': {str(e)}"}

def list_directory(dir_path: str = ".") -> Dict[str, Any]:
    """List contents of a directory."""
    try:
        path = normalize_path(dir_path)
        
        if not path.exists():
            return {"error": f"Directory '{dir_path}' does not exist"}
        
        if not path.is_dir():
            return {"error": f"'{dir_path}' is not a directory"}
        
        items = []
        for item in path.iterdir():
            # Use comprehensive exclusion logic
            if should_exclude_file(item):
                continue

            item_info = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            }
            items.append(item_info)
        
        return {
            "success": True,
            "directory": str(path),
            "items": items,
            "count": len(items)
        }
    except Exception as e:
        return {"error": f"Failed to list directory '{dir_path}': {str(e)}"}

# Tool definitions for Gemini function calling (function_declarations)
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the content of a single file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "read_multiple_files",
        "description": "Read the contents of multiple files",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read"
                }
            },
            "required": ["file_paths"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file or overwrite an existing one",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path where the file should be created"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "create_multiple_files",
        "description": "Create multiple files at once",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    },
                    "description": "List of files to create, each with path and content"
                }
            },
            "required": ["files"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace a specific snippet in a file with new content",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "original_snippet": {
                    "type": "string",
                    "description": "The exact text to be replaced"
                },
                "new_snippet": {
                    "type": "string",
                    "description": "The new text to replace the original snippet"
                }
            },
            "required": ["file_path", "original_snippet", "new_snippet"]
        }
    },
    {
        "name": "run_command",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "list_directory",
        "description": "List the contents of a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Path to the directory to list (default: current directory)"
                }
            }
        }
    }
]

# Function mapping for tool execution
TOOL_FUNCTIONS = {
    "read_file": read_local_file,
    "read_multiple_files": read_multiple_files,
    "create_file": create_file,
    "create_multiple_files": create_multiple_files,
    "edit_file": edit_file,
    "run_command": run_command,
    "list_directory": list_directory
}
