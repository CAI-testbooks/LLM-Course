# src/core/bug_fixer.py
class BugFixer:
    """自动修复Bug"""
    
    def __init__(self, model):
        self.model = model
    
    def fix(self, code: str, error: str, traceback: str = None) -> str:
        """根据错误信息修复代码"""
        
        prompt = f"""修复以下Python代码中的错误：

代码：
{code}

错误信息：
{error}

{'堆栈追踪：' + traceback if traceback else ''}

请提供修复后的完整代码，并解释修复了什么："""
        
        response = self.model.generate(prompt)
        
        # 提取修复后的代码
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        else:
            # 尝试从响应中提取代码
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            
            return '\n'.join(code_lines) if code_lines else code
    
    def fix_with_tests(self, code: str, failing_test: str) -> str:
        """根据失败的测试修复代码"""
        prompt = f"""以下代码无法通过测试：

代码：
{code}

失败的测试：
{failing_test}

请修复代码使其通过测试："""
        
        return self.model.generate(prompt)