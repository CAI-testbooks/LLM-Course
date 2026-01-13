# src/tools/code_analyzer.py
import ast
import os
import tempfile
import re
from typing import Dict, Any, List
import pylint.lint
from pylint.reporters.text import TextReporter
import io

class CodeAnalyzer:
    """代码静态分析"""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """分析代码质量"""
        
        analysis = {
            "syntax_valid": False,
            "ast_analysis": {},
            "pylint_score": 0,
            "issues": [],
            "suggestions": [],
            "complexity": 0,
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        # 检查语法
        try:
            tree = ast.parse(code)
            analysis["syntax_valid"] = True
            analysis["ast_analysis"] = self._analyze_ast(tree)
            analysis.update(analysis["ast_analysis"])  # 将AST分析结果合并到主分析结果中
        except SyntaxError as e:
            analysis["issues"].append(f"语法错误: {e}")
            return analysis
        
        # 运行pylint
        pylint_result = self._run_pylint(code)
        analysis.update(pylint_result)
        
        # 计算代码复杂度
        analysis["complexity"] = self._calculate_complexity(tree)
        
        return analysis
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """分析AST树"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "ast_nodes": 0,
            "max_depth": 0
        }
        
        # 统计函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": len(node.args.args),
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node)
                }
                analysis["functions"].append(func_info)
            
            # 统计类
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node)
                }
                analysis["classes"].append(class_info)
            
            # 统计导入
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append({"module": alias.name, "alias": alias.asname})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    analysis["imports"].append({
                        "from": module,
                        "import": alias.name,
                        "alias": alias.asname
                    })
        
        # 统计节点数和深度
        analysis["ast_nodes"] = len(list(ast.walk(tree)))
        analysis["max_depth"] = self._calculate_ast_depth(tree)
        
        return analysis
    
    def _calculate_ast_depth(self, node: ast.AST, depth: int = 0) -> int:
        """计算AST最大深度"""
        if not hasattr(node, 'body'):
            return depth
        
        max_depth = depth
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With)):
            depth += 1
            max_depth = depth
            
        if hasattr(node, 'body'):
            for child in ast.iter_child_nodes(node):
                child_depth = self._calculate_ast_depth(child, depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """计算代码复杂度"""
        complexity = 0
        
        for node in ast.walk(tree):
            # 控制流语句增加复杂度
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
            # 函数和类定义增加复杂度
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity += 1
        
        return complexity
    
    def _run_pylint(self, code: str) -> Dict[str, Any]:
        """运行pylint检查"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 捕获pylint输出
            pylint_output = io.StringIO()
            reporter = TextReporter(pylint_output)
            
            # 运行pylint，禁用一些常见的但可能不影响代码运行的警告
            pylint.lint.Run(
                [
                    '--disable=all',
                    '--enable=error,fatal',
                    '--reports=n',
                    '--score=n',
                    temp_file
                ],
                reporter=reporter,
                exit=False
            )
            
            output = pylint_output.getvalue()
            
            return {
                "pylint_output": output,
                "issues": self._parse_pylint_output(output),
                "pylint_score": self._extract_pylint_score(output)
            }
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _parse_pylint_output(self, output: str) -> List[str]:
        """解析pylint输出"""
        issues = []
        
        # 解析pylint错误和警告
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and ':' in line and ('error' in line.lower() or 'warning' in line.lower() or 'fatal' in line.lower()):
                # 提取有意义的信息
                match = re.search(r'([^:]+):(\d+):\s*(.+)', line)
                if match:
                    file_name, line_num, message = match.groups()
                    issues.append(f"Line {line_num}: {message}")
                else:
                    issues.append(line)
        
        return issues
    
    def _extract_pylint_score(self, output: str) -> float:
        """从pylint输出中提取分数"""
        # 查找分数模式
        match = re.search(r'Your code has been rated at (\d+\.?\d*)/10', output)
        if match:
            return float(match.group(1))
        return 0.0
    
    def get_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """根据分析结果提供建议"""
        suggestions = []
        
        # 基于复杂度提供建议
        if analysis.get("complexity", 0) > 10:
            suggestions.append("代码复杂度较高，建议拆分为更小的函数")
        
        # 基于函数数量提供建议
        functions = analysis.get("functions", [])
        if len(functions) == 0 and analysis.get("ast_nodes", 0) > 20:
            suggestions.append("代码较长但没有定义函数，建议将代码组织到函数中")
        
        # 基于导入提供建议
        imports = analysis.get("imports", [])
        if len(imports) > 10:
            suggestions.append("导入模块较多，建议检查是否有未使用的导入")
        
        # 基于pylint问题提供建议
        issues = analysis.get("issues", [])
        if issues:
            suggestions.append("请修复pylint发现的代码问题")
        
        return suggestions