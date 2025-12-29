import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from models.code_generator import CodeGenerator
from core.test_generator import TestGenerator

console = Console()

@click.group()
def cli():
    """AI驱动的软件开发助手 (MVP版本)"""
    pass

@cli.command()
@click.option('--prompt', '-p', help='编程需求描述')
@click.option('--interactive', '-i', is_flag=True, help='交互模式')
def generate(prompt, interactive):
    """根据需求生成代码"""
    
    console.print(Panel.fit("🧠 AI代码生成助手", style="bold blue"))
    
    generator = CodeGenerator()
    test_gen = TestGenerator(generator)
    
    if interactive:
        console.print("请输入你的编程需求（输入'quit'退出）：")
        while True:
            user_input = console.input("\n[bold yellow]需求> [/bold yellow]")
            if user_input.lower() == 'quit':
                break
            
            if user_input.strip():
                process_generation(generator, test_gen, user_input)
    elif prompt:
        process_generation(generator, test_gen, prompt)
    else:
        console.print("[red]错误：请提供需求描述或使用交互模式[/red]")

def process_generation(generator, test_gen, prompt):
    """处理代码生成流程"""
    
    # 生成代码
    with console.status("[bold green]正在生成代码..."):
        code = generator.generate_code(prompt)
    
    console.print("\n[bold green]✓ 代码生成完成！[/bold green]")
    
    # 显示生成的代码
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="生成的代码", border_style="green"))
    
    # 生成测试
    with console.status("[bold blue]正在生成测试用例..."):
        test_code = test_gen.generate_with_llm(code)
    
    console.print("\n[bold blue]✓ 测试生成完成！[/bold blue]")
    
    # 显示测试代码
    test_syntax = Syntax(test_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(test_syntax, title="生成的测试", border_style="blue"))
    
    # 询问是否保存
    save = click.confirm("\n是否保存到文件？")
    if save:
        save_to_file(code, test_code, prompt[:30])

def save_to_file(code, test_code, description):
    """保存代码和测试到文件"""
    import os
    import re
    
    # 清理描述，作为文件名
    safe_name = re.sub(r'[^\w\s-]', '', description).strip().replace(' ', '_')
    
    # 创建目录
    os.makedirs('output', exist_ok=True)
    
    # 保存主代码
    code_filename = f"output/{safe_name}.py"
    with open(code_filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # 保存测试代码
    test_filename = f"output/test_{safe_name}.py"
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    console.print(f"\n[green]✓ 代码已保存到: {code_filename}[/green]")
    console.print(f"[green]✓ 测试已保存到: {test_filename}[/green]")

@cli.command()
def demo():
    """运行演示示例"""
    examples = [
        "写一个函数，计算阶乘",
        "写一个函数，找出列表中的最大值",
        "写一个函数，检查素数"
    ]
    
    generator = CodeGenerator()
    test_gen = TestGenerator(generator)
    
    for example in examples:
        console.print(f"\n[bold cyan]示例: {example}[/bold cyan]")
        process_generation(generator, test_gen, example)
        console.print("\n" + "="*60)

if __name__ == "__main__":
    cli()