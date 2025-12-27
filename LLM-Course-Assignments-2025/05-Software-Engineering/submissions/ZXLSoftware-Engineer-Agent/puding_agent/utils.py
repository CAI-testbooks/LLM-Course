
"""
Utility functions for PUding Agent.
"""
import mimetypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from .config import EXCLUDED_FILES, EXCLUDED_EXTENSIONS

@dataclass
class ConversationMessage:
    """Represents a message in the conversation history."""
    role: str  # Can be 'user', 'assistant', 'system', 'tool'
    content: str  # The text content
    # For tool usage
    tool_calls: Optional[List[Any]] = None  # List of tool calls from assistant
    tool_call_id: Optional[str] = None      # ID for tool response
    name: Optional[str] = None              # Name of the tool

def clean_json_string(json_str: str) -> str:
    """Clean and fix common JSON formatting errors from LLM output."""
    import re
    
    # Remove markdown code blocks if present
    cleaned = json_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
        
    cleaned = cleaned.strip()
    
    # Fix common syntax errors
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    # Be careful with replacing single quotes, only do it if it looks like a JSON key/value wrapper
    # simplistic approach: replace all ' with " if it seems they are used as delimiters
    # But this is risky for content containing '. 
    # Better to rely on the model producing valid JSON, but we can fix Python-style None/True/False
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    
    return cleaned

def normalize_path(file_path: str) -> Path:
    """Normalize and validate file path to prevent directory traversal."""
    path = Path(file_path).resolve()
    cwd = Path.cwd().resolve()
    
    # Ensure the path is within the current working directory
    try:
        # Check if the resolved path starts with the current working directory
        if not str(path).startswith(str(cwd)):
            raise ValueError(f"Path {file_path} is outside the current directory")
    except ValueError: # This can happen if path is on a different drive on Windows, etc.
        raise ValueError(f"Path {file_path} is outside the current directory")
    
    return path

def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded from processing."""
    # Check if file name is in excluded files
    if file_path.name in EXCLUDED_FILES:
        return True
    
    # Check if file extension is in excluded extensions
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True
    
    # Check for compound extensions like .min.js, .chunk.css, etc.
    filename_lower = file_path.name.lower()
    for ext in EXCLUDED_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True
    
    # Check if it's a hidden file (starts with .)
    if file_path.name.startswith('.'):
        return True
    
    # Check if any parent directory is in excluded files
    for parent in file_path.parents:
        if parent.name in EXCLUDED_FILES:
            return True
    
    return False

def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        # If we can't read it, assume it's binary or inaccessible for safety
        return True

def is_text_file(file_path: Path) -> bool:
    """Check if a file is a text file based on extension and content."""
    # First check if it's in excluded extensions (early return for performance)
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False
    
    # Common text file extensions
    if file_path.suffix.lower() in ['.txt', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md', '.rst', '.sh', '.bat', '.gitignore', '.env', '.toml']:
        return True
    
    # Guess mimetype, if it's text, it's likely text
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and mime_type.startswith('text/'):
        return True
    
    # Fallback to binary check
    return not is_binary_file(file_path)
