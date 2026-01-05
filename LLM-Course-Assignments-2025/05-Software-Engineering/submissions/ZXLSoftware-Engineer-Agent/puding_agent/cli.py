
"""
CLI entry point for PUding Agent.
"""
import sys
import shlex
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from .agent import GeminiEngineer
from .utils import ConversationMessage

def main():
    """Main CLI loop."""
    console = Console()
    engineer = GeminiEngineer()
    engineer.display_welcome_banner()
    
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })
    session = PromptSession(style=style)
    
    while True:
        try:
            user_input = session.prompt("\nUser > ", history=engineer.history).strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/exit', '/quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            if user_input.lower() == '/clear':
                engineer.conversation_history = [engineer.conversation_history[0]]  # Keep system prompt
                console.print("[green]Conversation history cleared.[/green]")
                continue
                
            if user_input.lower() == '/help':
                engineer.display_welcome_banner()
                continue
            
            # Handle /add command
            if user_input.lower().startswith('/add '):
                args = shlex.split(user_input[5:])
                if args:
                    engineer.add_file_to_context(args[0])
                else:
                    console.print("[red]Usage: /add <file_or_directory_path>[/red]")
                continue
            
            # Process user input
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots") as status:
                def loop_callback(count):
                    if count > 1:
                        status.update(f"[bold green]Thinking... (Reflection Cycle {count})[/bold green]")
                
                response = engineer.respond_once(user_input, on_loop_start=loop_callback)
            
            if "error" in response:
                console.print(Panel(f"[red]Error: {response['error']}[/red]", title="Error"))
            else:
                console.print(Panel(response["assistant_text"], title="🤖 Assistant", border_style="blue"))
                
                # Check if there were tool executions
                if "tools_executed" in response and response["tools_executed"]:
                    for tool_exec in response["tools_executed"]:
                        engineer.display_tool_result(tool_exec["name"], tool_exec["result"])
                        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]An error occurred: {e}[/red]")

if __name__ == "__main__":
    main()
