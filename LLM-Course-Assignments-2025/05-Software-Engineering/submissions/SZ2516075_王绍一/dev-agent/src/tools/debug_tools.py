# src/tools/debug_tools.py
import sys
import traceback
import pdb
import inspect
from typing import Dict, List, Any, Optional


class DebugTools:
    """调试工具类"""
    
    def __init__(self):
        print("🔧 调试工具初始化")
    
    def trace_execution(self, code: str, input_data: str = None) -> Dict[str, Any]:
        """追踪代码执行"""
        print("🔍 开始追踪代码执行...")
        
        try:
            # 创建本地命名空间
            local_vars = {}
            
            # 追踪变量变化
            variable_history = {}
            
            # 重写print函数以捕获输出
            output_capture = []
            
            def custom_print(*args, **kwargs):
                output = " ".join(str(arg) for arg in args)
                output_capture.append(output)
                print(output)  # 同时输出到控制台
            
            # 将重写函数注入到命名空间
            local_vars['print'] = custom_print
            
            # 执行代码，逐行追踪
            lines = code.strip().split('\n')
            line_history = []
            
            for i, line in enumerate(lines, 1):
                try:
                    if line.strip() and not line.strip().startswith('#'):
                        # 记录当前行
                        line_history.append({
                            "line": i,
                            "code": line.strip(),
                            "variables": {}
                        })
                        
                        # 执行当前行
                        exec(line, {"__builtins__": __builtins__}, local_vars)
                        
                        # 记录变量变化
                        for var_name, var_value in list(local_vars.items()):
                            if not var_name.startswith('_'):
                                variable_history.setdefault(var_name, []).append({
                                    "line": i,
                                    "value": str(var_value),
                                    "type": type(var_value).__name__
                                })
                                
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"第{i}行: {str(e)}",
                        "line": i,
                        "traceback": traceback.format_exc(),
                        "variable_history": variable_history,
                        "output": output_capture
                    }
            
            return {
                "success": True,
                "output": output_capture,
                "variable_history": variable_history,
                "line_history": line_history,
                "final_variables": {k: v for k, v in local_vars.items() if not k.startswith('_')}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def analyze_stack_trace(self, traceback_text: str) -> Dict[str, Any]:
        """分析堆栈追踪"""
        if not traceback_text:
            return {"error": "无堆栈追踪信息"}
        
        lines = traceback_text.strip().split('\n')
        
        # 提取错误信息
        error_info = {
            "error_type": "",
            "error_message": "",
            "file": "",
            "line": 0,
            "function": "",
            "traceback_lines": []
        }
        
        for line in lines:
            line = line.strip()
            
            # 提取错误类型
            if "Error:" in line or "Exception:" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    error_info["error_type"] = parts[0].strip()
                    error_info["error_message"] = parts[1].strip()
            
            # 提取文件和行号
            elif "File" in line and "line" in line:
                # 格式: File "filename", line X, in function
                import re
                match = re.search(r'File "(.+)", line (\d+), in (.+)', line)
                if match:
                    if not error_info["file"]:
                        error_info["file"] = match.group(1)
                        error_info["line"] = int(match.group(2))
                        error_info["function"] = match.group(3)
                    
                    error_info["traceback_lines"].append({
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "function": match.group(3)
                    })
        
        # 分析可能的错误原因
        error_causes = self._suggest_error_causes(error_info)
        
        return {
            **error_info,
            "suggested_causes": error_causes,
            "fix_suggestions": self._suggest_fixes(error_info, error_causes)
        }
    
    def _suggest_error_causes(self, error_info: Dict[str, Any]) -> List[str]:
        """根据错误信息建议可能的原因"""
        error_type = error_info["error_type"]
        error_message = error_info["error_message"]
        
        causes = []
        
        # 常见的Python错误类型
        error_patterns = {
            "NameError": [
                "变量未定义",
                "函数名拼写错误",
                "导入模块错误"
            ],
            "TypeError": [
                "类型不匹配",
                "参数数量错误",
                "调用不可调用对象"
            ],
            "ValueError": [
                "参数值无效",
                "格式错误",
                "超出范围"
            ],
            "IndexError": [
                "列表索引超出范围",
                "字符串索引错误"
            ],
            "KeyError": [
                "字典键不存在",
                "访问不存在的键"
            ],
            "AttributeError": [
                "对象没有该属性",
                "属性名拼写错误"
            ],
            "SyntaxError": [
                "语法错误",
                "缩进错误",
                "括号不匹配"
            ],
            "IndentationError": [
                "缩进不一致",
                "缺少缩进"
            ],
            "ImportError": [
                "模块不存在",
                "导入路径错误"
            ],
            "ModuleNotFoundError": [
                "模块未安装",
                "模块名错误"
            ],
            "ZeroDivisionError": [
                "除以零",
                "分母为零"
            ]
        }
        
        # 根据错误类型添加原因
        if error_type in error_patterns:
            causes.extend(error_patterns[error_type])
        
        # 根据错误消息添加特定原因
        error_lower = error_message.lower()
        
        if "is not defined" in error_lower:
            causes.append("变量或函数未定义")
        if "takes" in error_lower and "arguments" in error_lower:
            causes.append("函数参数数量不正确")
        if "cannot" in error_lower and "concatenate" in error_lower:
            causes.append("类型不匹配，无法拼接")
        if "out of range" in error_lower:
            causes.append("索引超出范围")
        if "division by zero" in error_lower:
            causes.append("除数为零")
        if "invalid syntax" in error_lower:
            causes.append("语法错误")
        
        # 如果没有找到特定原因，添加通用原因
        if not causes:
            causes.append("未知错误，请检查代码逻辑")
        
        return list(set(causes))  # 去重
    
    def _suggest_fixes(self, error_info: Dict[str, Any], causes: List[str]) -> List[str]:
        """根据错误原因建议修复方法"""
        error_type = error_info["error_type"]
        fixes = []
        
        # 通用修复建议
        generic_fixes = [
            "检查拼写错误",
            "查看文档或API参考",
            "打印相关变量的值",
            "使用try-except捕获异常",
            "简化代码以隔离问题"
        ]
        
        # 特定错误类型的修复建议
        type_specific_fixes = {
            "NameError": [
                "检查变量是否正确定义",
                "确保函数已正确导入",
                "检查变量作用域"
            ],
            "TypeError": [
                "检查参数类型",
                "查看函数签名",
                "使用type()函数检查类型"
            ],
            "ValueError": [
                "检查参数值是否有效",
                "验证输入数据",
                "添加参数验证"
            ],
            "IndexError": [
                "检查索引是否在范围内",
                "使用len()函数获取长度",
                "添加边界检查"
            ],
            "KeyError": [
                "检查字典键是否存在",
                "使用dict.get()方法",
                "添加键存在性检查"
            ],
            "AttributeError": [
                "检查对象是否有该属性",
                "查看对象类型和可用属性",
                "检查属性名拼写"
            ],
            "SyntaxError": [
                "检查代码语法",
                "使用代码格式化工具",
                "检查括号是否匹配"
            ],
            "ImportError": [
                "检查模块是否已安装",
                "检查导入路径",
                "确保模块名正确"
            ]
        }
        
        # 添加类型特定的修复建议
        if error_type in type_specific_fixes:
            fixes.extend(type_specific_fixes[error_type])
        
        # 添加通用修复建议
        fixes.extend(generic_fixes)
        
        # 根据具体错误原因添加建议
        for cause in causes:
            if "未定义" in cause:
                fixes.append("在使用前定义变量")
            if "类型不匹配" in cause:
                fixes.append("确保操作的数据类型一致")
            if "参数数量" in cause:
                fixes.append("检查函数定义和调用时的参数数量")
            if "索引超出范围" in cause:
                fixes.append("使用0到len(list)-1的索引")
            if "除数为零" in cause:
                fixes.append("在除法前检查分母是否为零")
        
        return list(set(fixes))  # 去重
    
    def interactive_debug(self, code: str, breakpoints: List[int] = None):
        """交互式调试"""
        print("🐛 启动交互式调试模式")
        
        if breakpoints is None:
            breakpoints = []
        
        # 这里可以实现一个简单的调试器
        # 由于交互式调试比较复杂，这里提供一个简化版本
        return {
            "available_commands": [
                "break <line> - 设置断点",
                "step - 单步执行",
                "continue - 继续执行",
                "print <var> - 打印变量",
                "locals - 显示局部变量",
                "globals - 显示全局变量",
                "quit - 退出调试"
            ],
            "breakpoints": breakpoints,
            "note": "交互式调试功能需要完整实现调试器，当前为简化版本"
        }
    
    def find_potential_bugs(self, code: str) -> List[Dict[str, Any]]:
        """查找潜在Bug"""
        print("🔍 扫描潜在Bug...")
        
        bugs = []
        lines = code.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # 检查常见问题模式
            bug = self._analyze_line_for_bugs(line, i)
            if bug:
                bugs.append(bug)
        
        return bugs
    
    def _analyze_line_for_bugs(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """分析单行代码的潜在问题"""
        
        # 跳过空行和注释
        if not line or line.startswith('#'):
            return None
        
        # 检查常见问题
        checks = [
            self._check_for_hardcoded_values,
            self._check_for_empty_except,
            self._check_for_magic_numbers,
            self._check_for_unused_variables,
            self._check_for_dangerous_functions,
            self._check_for_possible_division_by_zero
        ]
        
        for check_func in checks:
            bug = check_func(line, line_number)
            if bug:
                return bug
        
        return None
    
    def _check_for_hardcoded_values(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查硬编码值"""
        # 简单的硬编码字符串检查
        if '"/' in line or "'/" in line:
            return {
                "line": line_number,
                "type": "硬编码路径",
                "description": "代码中使用了硬编码的路径",
                "severity": "中等",
                "suggestion": "考虑使用配置文件或环境变量"
            }
        return None
    
    def _check_for_empty_except(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查空的except块"""
        if "except:" in line or "except Exception:" in line:
            if "pass" in line or "..." in line:
                return {
                    "line": line_number,
                    "type": "空的异常处理",
                    "description": "空的except块会隐藏错误",
                    "severity": "高",
                    "suggestion": "至少记录异常信息"
                }
        return None
    
    def _check_for_magic_numbers(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查魔数（未命名的常量）"""
        import re
        
        # 查找数字字面量
        numbers = re.findall(r'\b\d+\b', line)
        for num in numbers:
            # 跳过0, 1等常见数字
            if num not in ['0', '1', '10', '100', '1000']:
                return {
                    "line": line_number,
                    "type": "魔数",
                    "description": f"代码中使用了未命名的常量: {num}",
                    "severity": "低",
                    "suggestion": "将常量定义为有意义的变量名"
                }
        return None
    
    def _check_for_unused_variables(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查未使用的变量（简化版）"""
        # 匹配变量赋值
        import re
        match = re.match(r'(\w+)\s*=', line)
        if match:
            var_name = match.group(1)
            # 这里应该检查变量是否被使用，简化实现
            return {
                "line": line_number,
                "type": "可能的未使用变量",
                "description": f"变量 {var_name} 可能未被使用",
                "severity": "低",
                "suggestion": "如果变量不需要，请删除"
            }
        return None
    
    def _check_for_dangerous_functions(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查危险函数调用"""
        dangerous = ['eval', 'exec', 'input', 'open']
        for func in dangerous:
            if f"{func}(" in line:
                return {
                    "line": line_number,
                    "type": "潜在危险函数",
                    "description": f"使用了 {func} 函数",
                    "severity": "中等",
                    "suggestion": f"确保 {func} 的输入是安全的"
                }
        return None
    
    def _check_for_possible_division_by_zero(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """检查可能的除以零"""
        if '/' in line or '//' in line or '%' in line:
            # 查找除数
            import re
            # 简单的检查：如果分母是变量，可能为零
            if re.search(r'/\s*(\w+)', line) or re.search(r'//\s*(\w+)', line) or re.search(r'%\s*(\w+)', line):
                return {
                    "line": line_number,
                    "type": "可能的除以零",
                    "description": "除法操作可能除数为零",
                    "severity": "高",
                    "suggestion": "在除法前检查分母是否为零"
                }
        return None

# 单例实例
debug_tools = DebugTools()