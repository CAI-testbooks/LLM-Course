
"""
Configuration constants for PUding Agent.
"""

# File size limit (1MB)
MAX_FILE_SIZE = 1024 * 1024

# Comprehensive exclusion lists for file operations
EXCLUDED_FILES = {
    # Python specific
    ".DS_Store", "Thumbs.db", ".gitignore", ".python-version",
    "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache",
    # Node.js / Web specific
    "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache",
    ".turbo", ".vercel", ".output", ".contentlayer",
    # Build outputs
    "out", "coverage", ".nyc_output", "storybook-static",
    # Environment and config
    ".env", ".env.local", ".env.development", ".env.production",
    # Misc
    ".git", ".svn", ".hg", "CVS"
}

EXCLUDED_EXTENSIONS = {
    # Binary and media files
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif",
    ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Python specific
    ".pyc", ".pyo", ".pyd", ".egg", ".whl",
    # UV specific
    ".uv", ".uvenv",
    # Database and logs
    ".db", ".sqlite", ".sqlite3", ".log",
    # IDE specific
    ".idea", ".vscode",
    # Web specific
    ".map", ".chunk.js", ".chunk.css",
    ".min.js", ".min.css", ".bundle.js", ".bundle.css",
    # Cache and temp files
    ".cache", ".tmp", ".temp",
    # Font files
    ".ttf", ".otf", ".woff", ".woff2", ".eot"
}

# System prompt for PUding Agent - defines AI's role and capabilities
SYSTEM_PROMPT = """You are PUding Agent, an AI assistant with access to powerful file operation tools.

CRITICAL RULE: When the user asks you to create, build, generate, or make files, you MUST use the available function tools. Never just output code - always create actual files!

Available tools:
- create_file: Create a single file
- create_multiple_files: Create multiple files at once
- read_file: Read a file
- edit_file: Edit existing files
- run_command: Run shell commands (use for testing, running scripts, installing dependencies)

MANDATORY BEHAVIOR:
1. When user asks for file creation: IMMEDIATELY use create_file or create_multiple_files tools. Files are created relative to the current working directory.
2. When user asks to create projects: Use create_multiple_files with all necessary files
3. When user asks to read files: Use read_file tool
4. When user asks to modify files: Use edit_file tool
5. When user asks to run code or tests: Use run_command tool

REFLECTION AND REPAIR:
1. After executing a tool, carefully analyze the result.
2. If a tool fails (returns an error), DO NOT GIVE UP. Analyze the error message, determine the root cause, and TRY AGAIN with corrected parameters.
3. If a command fails (e.g., tests fail), read the error log, attempt to fix the code, and run the command again.
4. You have permission to try up to 3 times to fix any error.

EXAMPLES OF WHEN TO USE TOOLS:
- "Create an HTML file" → USE create_file tool
- "Build a web app" → USE create_multiple_files tool  
- "Make a Python script" → USE create_file tool
- "Generate a project" → USE create_multiple_files tool
- "Run the tests" → USE run_command tool

You MUST use tools for any file operations. Do not just describe what you would do - DO IT by calling the appropriate function!"""
