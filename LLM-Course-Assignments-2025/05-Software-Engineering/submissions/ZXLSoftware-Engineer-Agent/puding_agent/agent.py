
"""
Main agent logic for PUding Agent.
"""
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit.history import InMemoryHistory
import openai
import google.generativeai as genai
from dotenv import load_dotenv

from .utils import ConversationMessage, normalize_path, should_exclude_file, is_text_file, clean_json_string
from .config import SYSTEM_PROMPT
from .tools import TOOLS, TOOL_FUNCTIONS, read_local_file

# Load environment variables
load_dotenv()

class GeminiEngineer:
    """Main application class for PUding Agent."""
    
    def __init__(self):
        self.console = Console()
        self.conversation_history: List[ConversationMessage] = []
        self.model = None
        self.client = None
        self.provider = "gemini"
        self.history = InMemoryHistory()
        self.setup_llm_client()
        
        # Add system prompt as the first message to guide AI behavior
        self.conversation_history.append(ConversationMessage("system", SYSTEM_PROMPT))
        
    def setup_llm_client(self):
        """Initialize the LLM client (Gemini or OpenAI/DeepSeek)."""
        self.provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
        
        if self.provider in ['openai', 'deepseek', 'qwen']:
            self.setup_openai_client()
        else:
            self.setup_gemini_client()

    def setup_openai_client(self):
        """Initialize OpenAI compatible client."""
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')
        self.model_name = os.getenv('OPENAI_MODEL', 'deepseek-coder')
        
        if not api_key:
            self.console.print(Panel(
                "[red]❌ OpenAI API key not found![/red]\n\n"
                "Please set your OPENAI_API_KEY environment variable:\n"
                "1. Add to .env: OPENAI_API_KEY=your_key\n"
                "2. Optionally set OPENAI_BASE_URL for DeepSeek/Qwen",
                title="Setup Required",
                border_style="red"
            ))
            sys.exit(1)
            
        try:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.console.print(f"[green]✅ OpenAI compatible client initialized! (Provider: {self.provider}, Model: {self.model_name})[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Failed to initialize OpenAI client: {e}[/red]")
            sys.exit(1)

    def setup_gemini_client(self):
        """Initialize the Gemini client and model."""
        api_key = os.getenv('GEMINI_API_KEY')
        # Default to a stable model that supports function calling
        model_name = os.getenv('GEMINI_MODEL')
        
        if not api_key or api_key == 'your_api_key_here':
            self.console.print(Panel(
                "[red]❌ Gemini API key not found![/red]\n\n"
                "Please set your GEMINI_API_KEY environment variable:\n"
                "1. Create a .env file in this directory\n"
                "2. Add: GEMINI_API_KEY=your_actual_api_key\n"
                "3. Restart the application",
                title="Setup Required",
                border_style="red"
            ))
            sys.exit(1)
            
        genai.configure(api_key=api_key)
        
        # Choose model
        if not model_name:
            model_name = "gemini-2.0-flash-exp"
            
        self.model_name = model_name
        self.console.print(f"[green]✅ Gemini client initialized! (Model: {model_name})[/green]")
        
        try:
            self.model = genai.GenerativeModel(model_name=model_name)
        except Exception as e:
            self.console.print(f"[red]❌ Failed to create Gemini model: {e}[/red]")
            sys.exit(1)

    def display_welcome_banner(self):
        """Display the welcome banner and instructions."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                        🤖 PUding Agent                    ║
║                AI-Driven Software Architect                  ║
║              Autonomous Project Generation System            ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        instructions = """
[bold cyan]Core Capabilities:[/bold cyan]
• [green]Autonomous Project Generation[/green] - Create complete projects from high-level descriptions
• [green]Software Architecture[/green] - Design and structure complex applications
• [green]Code Analysis & Optimization[/green] - Review and improve existing code
• [green]File Management[/green] - Comprehensive file operations with safety

[bold cyan]Available Commands:[/bold cyan]
• [yellow]/add <file_path>[/yellow] - Add a file to conversation context
• [yellow]/add <folder_path>[/yellow] - Add all files in a folder to context
• [yellow]/exit[/yellow] or [yellow]/quit[/yellow] - Exit the application
• [yellow]/help[/yellow] - Show this help message
• [yellow]/clear[/yellow] - Clear conversation history

[bold cyan]Example Requests:[/bold cyan]
• "Create a Flask API for a task manager with SQLite database"
• "Build a React component library with TypeScript"
• "Generate a Python CLI tool with argument parsing"
• "Create a Node.js Express server with authentication"

[bold green]Ready to architect and build your software projects![/bold green]
        """
        
        self.console.print(Panel(banner, style="bold blue"))
        self.console.print(Panel(instructions, title="🚀 Getting Started", border_style="cyan"))

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with parameters."""
        if tool_name not in TOOL_FUNCTIONS:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            func = TOOL_FUNCTIONS[tool_name]
            return func(**parameters)
        except Exception as e:
            return {"error": f"Error executing {tool_name}: {str(e)}"}

    def display_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Display the result of a tool execution in a nice format."""
        if tool_name == "run_command":
            status = "[green]✅ Success[/green]" if result.get('success') else "[red]❌ Failed[/red]"
            output = result.get('stdout', '') + result.get('stderr', '')
            if len(output) > 1000:
                output = output[:1000] + "\n... (truncated)"
            
            self.console.print(Panel(
                f"{status}\nCommand: {result.get('command')}\nReturn Code: {result.get('returncode')}\n\nOutput:\n{output}",
                title="Command Execution",
                border_style="green" if result.get('success') else "red"
            ))
        elif tool_name == "create_file":
            self.console.print(Panel(
                f"[green]✅ {result['message']} ({result['size']} characters)[/green]",
                title="File Created",
                border_style="green"
            ))
        elif tool_name == "create_multiple_files":
            if result.get('success'):
                table = Table(title="🎉 Project Files Created Successfully")
                table.add_column("File Path", style="cyan", no_wrap=False)
                table.add_column("Size (chars)", style="green")
                table.add_column("Status", style="bold green")
                
                total_files = 0
                total_size = 0
                
                for file_path, file_result in result['files'].items():
                    table.add_row(
                        file_path, 
                        str(file_result['size']), 
                        "✅ Created"
                    )
                    total_files += 1
                    total_size += file_result['size']
                
                self.console.print(table)
                self.console.print(Panel(
                    f"[bold green]📊 Summary: {total_files} files created, {total_size:,} total characters[/bold green]",
                    title="Project Generation Complete",
                    border_style="green"
                ))
            
            if result.get('errors'):
                self.console.print(Panel(
                    "\n".join(result['errors']),
                    title="Creation Errors",
                    border_style="yellow"
                ))
        elif tool_name == "edit_file":
            self.console.print(Panel(
                f"[green]✅ {result['message']}[/green]\n"
                f"Changes: {result['changes']['diff']:+d} characters",
                title="File Edited",
                border_style="green"
            ))
        elif tool_name == "list_directory":
            table = Table(title=f"Directory: {result['directory']}")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Size", style="green")
            
            for item in result['items']:
                size_str = str(item['size']) if item['size'] else "-"
                table.add_row(item['name'], item['type'], size_str)
            
            self.console.print(table)
        elif tool_name == "read_file":
            content_preview = result.get('content', '')[:500] + "..." if len(result.get('content', '')) > 500 else result.get('content', '')
            self.console.print(Panel(
                f"[green]✅ Read {result.get('size')} characters from {result.get('file_path')}[/green]\n\n{content_preview}",
                title="File Read",
                border_style="green"
            ))
        elif tool_name == "read_multiple_files":
             self.console.print(Panel(
                f"[green]✅ Read {len(result.get('files', {}))} files[/green]",
                title="Files Read",
                border_style="green"
            ))

    def respond_once(self, user_input: str, on_loop_start=None) -> Dict[str, Any]:
        """
        Process a message and run the autonomous loop (Reflection & Repair).
        Returns aggregated text and tool executions.
        
        Args:
            user_input: The user's message
            on_loop_start: Optional callback function(iteration_count) called at start of each loop
        """
        self.conversation_history.append(ConversationMessage("user", user_input))
        
        aggregated_text = []
        all_tool_executions = []
        loop_count = 0
        MAX_LOOPS = 10  # Prevent infinite loops
        
        while loop_count < MAX_LOOPS:
            loop_count += 1
            if on_loop_start:
                try:
                    on_loop_start(loop_count)
                except:
                    pass
            
            if self.provider in ['openai', 'deepseek', 'qwen']:
                # OpenAI Logic
                messages = []
                for msg in self.conversation_history:
                    if msg.role == "system":
                        messages.append({"role": "system", "content": msg.content})
                    elif msg.role == "assistant":
                        m = {"role": "assistant", "content": msg.content}
                        if msg.tool_calls:
                            m["tool_calls"] = msg.tool_calls
                        messages.append(m)
                    elif msg.role == "user":
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.role == "tool":
                        messages.append({
                            "role": "tool", 
                            "content": msg.content,
                            "tool_call_id": msg.tool_call_id,
                            "name": msg.name
                        })
                
                openai_tools = []
                for tool in TOOLS:
                    openai_tools.append({"type": "function", "function": tool})
                
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages, 
                        tools=openai_tools
                    )
                except Exception as e:
                    return {"error": str(e), "assistant_text": "\n".join(aggregated_text)}
                
                choice = resp.choices[0]
                assistant_msg = choice.message
                content = assistant_msg.content or ""
                tool_calls = getattr(assistant_msg, "tool_calls", None) or []
                
                # Append assistant response to history
                # We need to store tool_calls in the message for OpenAI context
                self.conversation_history.append(ConversationMessage(
                    role="assistant", 
                    content=content,
                    tool_calls=tool_calls if tool_calls else None
                ))
                
                if content:
                    aggregated_text.append(content)
                
                if not tool_calls:
                    # No tools called, we are done
                    break
                
                # Execute tools
                for tc in tool_calls:
                    name = tc.function.name
                    args_str = tc.function.arguments or ""
                    call_id = tc.id
                    
                    params = {}
                    try:
                        params = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Attempt simple fix
                        try:
                            fixed = clean_json_string(args_str)
                            params = json.loads(fixed)
                        except:
                            params = {} # Fail gracefully
                            
                    result = self.execute_tool(name, params)
                    
                    # Store execution for return
                    all_tool_executions.append({
                        "name": name, 
                        "parameters": params, 
                        "result": result
                    })
                    
                    # Append tool result to history
                    self.conversation_history.append(ConversationMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=call_id,
                        name=name
                    ))
                    
            else:
                # Gemini Logic
                messages = []
                for msg in self.conversation_history:
                    if msg.role == "system":
                        messages.append({"role": "user", "parts": [msg.content]})
                    elif msg.role == "assistant":
                        parts = []
                        if msg.content:
                            parts.append({"text": msg.content})
                        if msg.tool_calls:
                            # Reconstruct function calls for Gemini history
                            for tc in msg.tool_calls:
                                parts.append({"function_call": tc})
                        messages.append({"role": "model", "parts": parts})
                    elif msg.role == "user":
                        messages.append({"role": "user", "parts": [msg.content]})
                    elif msg.role == "tool":
                        # Gemini expects function_response
                        messages.append({
                            "role": "function",
                            "parts": [{
                                "function_response": {
                                    "name": msg.name,
                                    "response": {"result": msg.content}
                                }
                            }]
                        })

                try:
                    resp = self.model.generate_content(messages, tools=[{"function_declarations": TOOLS}])
                except Exception as e:
                    return {"error": str(e), "assistant_text": "\n".join(aggregated_text)}
                
                if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                    if resp.prompt_feedback.block_reason:
                        return {"error": f"Response blocked: {resp.prompt_feedback.block_reason}"}

                text_parts = []
                gemini_tool_calls = []
                
                if hasattr(resp, "candidates") and resp.candidates:
                    for candidate in resp.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    gemini_tool_calls.append(part.function_call)
                                elif hasattr(part, "text") and part.text:
                                    text_parts.append(part.text)
                
                assistant_text = "".join(text_parts)
                if assistant_text:
                    aggregated_text.append(assistant_text)
                
                # Append assistant response
                self.conversation_history.append(ConversationMessage(
                    role="assistant",
                    content=assistant_text,
                    tool_calls=gemini_tool_calls if gemini_tool_calls else None
                ))
                
                if not gemini_tool_calls:
                    break
                    
                # Execute tools
                for fc in gemini_tool_calls:
                    name = fc.name
                    params = {}
                    if hasattr(fc, "args"):
                        for key, value in fc.args.items():
                            if hasattr(value, "string_value"):
                                params[key] = value.string_value
                            elif hasattr(value, "list_value"):
                                # Simplify list extraction
                                params[key] = [v.string_value for v in value.list_value.values if hasattr(v, "string_value")]
                            else:
                                params[key] = value

                    result = self.execute_tool(name, params)
                    
                    all_tool_executions.append({
                        "name": name,
                        "parameters": params,
                        "result": result
                    })
                    
                    # Append tool result
                    # For Gemini, we store it as 'tool' role in our abstract history,
                    # but map it to 'function' role in message construction above.
                    self.conversation_history.append(ConversationMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        name=name
                    ))

        return {"assistant_text": "\n\n".join(aggregated_text), "tools_executed": all_tool_executions}

    def add_file_to_context(self, file_path: str):
        """Add a file or directory to the conversation context."""
        try:
            path = normalize_path(file_path)
            
            if not path.exists():
                self.console.print(f"[red]❌ Path '{file_path}' does not exist[/red]")
                return
            
            if path.is_file():
                result = read_local_file(str(path))
                if "error" in result:
                    self.console.print(f"[red]❌ {result['error']}[/red]")
                else:
                    # Add file content as a user message for context
                    content_text = f"File: {result['file_path']}\n```\n{result['content']}\n```"
                    self.conversation_history.append(ConversationMessage("user", content_text))
                    self.console.print(f"[green]✅ Added '{path}' to context[/green]")
            
            elif path.is_dir():
                added_count = 0
                skipped_count = 0
                for file_path in path.rglob("*"):  # Recursive glob
                    # Skip excluded files and directories
                    if should_exclude_file(file_path):
                        skipped_count += 1
                        continue
                    
                    if file_path.is_file() and is_text_file(file_path):
                        try:
                            result = read_local_file(str(file_path))
                            if "success" in result:
                                content_text = f"File: {result['file_path']}\n```\n{result['content']}\n```"
                                self.conversation_history.append(ConversationMessage("user", content_text))
                                added_count += 1
                            else:
                                skipped_count += 1
                        except Exception as e:
                            skipped_count += 1
                            continue
                    elif file_path.is_file():  # If it's a file but not text
                        skipped_count += 1
                
                self.console.print(f"[green]✅ Added {added_count} text files from '{path}' to context. Skipped {skipped_count} files (binary/non-text/errors).[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error adding '{file_path}' to context: {e}[/red]")
