# src/core/test_generator.py
import ast

class SmartTestGenerator:
    """智能测试生成器"""
    
    def __init__(self, model):
        self.model = model
    
    def generate(self, code: str, function_name: str = None) -> str:
        """为代码生成测试"""
        
        if not function_name:
            function_name = self._extract_function_name(code)
        
        # 分析代码结构
        code_analysis = self._analyze_code(code)
        
        prompt = f"""请为以下Python函数生成完整的单元测试：

函数代码：
{code}

函数分析：
{code_analysis}

请生成pytest格式的测试代码，包含：
1. 正常情况测试
2. 边界条件测试
3. 异常情况测试
4. 性能测试（如果需要）

只返回测试代码："""
        
        test_code = self.model.generate(prompt)
        
        # 验证测试代码语法
        if self._validate_python_syntax(test_code):
            return test_code
        else:
            return self._generate_basic_test(code, function_name)
    
    def _analyze_code(self, code: str) -> str:
        """分析代码结构"""
        try:
            tree = ast.parse(code)
            analysis = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis.append(f"函数: {node.name}")
                    analysis.append(f"参数: {[arg.arg for arg in node.args.args]}")
                    analysis.append(f"返回值类型: {self._infer_return_type(node)}")
            
            return '\n'.join(analysis)
        except:
            return "无法解析代码结构"