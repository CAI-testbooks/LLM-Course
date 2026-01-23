# src/tools/git_tools.py
import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional  # 添加这行
from datetime import datetime

class GitTools:
    """Git版本控制工具"""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or os.getcwd()
        print(f"🔧 Git工具初始化: {self.repo_path}")
    
    def run_git_command(self, command: str, cwd: str = None) -> Dict[str, Any]:  # 这里使用Dict
        """运行Git命令"""
        cwd = cwd or self.repo_path
        
        try:
            # 分割命令
            args = command.split()
            if args[0] != 'git':
                args.insert(0, 'git')
            
            # 运行命令
            result = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "命令执行超时",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def clone_repository(self, repo_url: str, target_dir: str = None) -> Dict[str, Any]:
        """克隆仓库"""
        if not target_dir:
            # 从URL提取仓库名
            repo_name = repo_url.split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            target_dir = os.path.join(os.getcwd(), 'repos', repo_name)
        
        print(f"📥 克隆仓库: {repo_url} -> {target_dir}")
        
        # 创建目录
        os.makedirs(target_dir, exist_ok=True)
        
        result = self.run_git_command(f"clone {repo_url} {target_dir}", cwd=os.path.dirname(target_dir))
        
        if result["success"]:
            self.repo_path = target_dir
            print(f"✅ 仓库克隆成功: {target_dir}")
        else:
            print(f"❌ 仓库克隆失败: {result.get('error', '未知错误')}")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取仓库状态"""
        result = self.run_git_command("status --porcelain")
        
        if not result["success"]:
            return {"success": False, "error": result.get("stderr")}
        
        files = []
        for line in result["stdout"].strip().split('\n'):
            if line:
                status = line[:2]
                filename = line[3:]
                
                file_status = {
                    "filename": filename,
                    "status": status,
                    "staged": status[0] != ' ',
                    "unstaged": status[1] != ' '
                }
                
                # 解释状态
                status_map = {
                    'M': '修改',
                    'A': '新增',
                    'D': '删除',
                    'R': '重命名',
                    'C': '复制',
                    'U': '更新但未合并',
                    '??': '未跟踪',
                    '!!': '忽略'
                }
                
                file_status["description"] = status_map.get(status.strip(), '未知')
                files.append(file_status)
        
        return {
            "success": True,
            "files": files,
            "total": len(files),
            "has_changes": len(files) > 0
        }
    
    def commit_changes(self, message: str, files: List[str] = None) -> Dict[str, Any]:
        """提交更改"""
        if not message:
            message = f"自动提交 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 添加文件
        if files:
            for file in files:
                add_result = self.run_git_command(f"add {file}")
                if not add_result["success"]:
                    return add_result
        else:
            add_result = self.run_git_command("add .")
            if not add_result["success"]:
                return add_result
        
        # 提交
        commit_result = self.run_git_command(f'commit -m "{message}"')
        
        return commit_result
    
    def create_branch(self, branch_name: str, checkout: bool = True) -> Dict[str, Any]:
        """创建分支"""
        result = self.run_git_command(f"branch {branch_name}")
        
        if not result["success"]:
            return result
        
        if checkout:
            return self.checkout_branch(branch_name)
        
        return result
    
    def checkout_branch(self, branch_name: str) -> Dict[str, Any]:
        """切换分支"""
        return self.run_git_command(f"checkout {branch_name}")
    
    def get_branches(self) -> Dict[str, Any]:
        """获取分支列表"""
        result = self.run_git_command("branch -a")
        
        if not result["success"]:
            return result
        
        branches = []
        current = None
        
        for line in result["stdout"].strip().split('\n'):
            if line:
                if line.startswith('*'):
                    current = line[2:].strip()
                    branches.append({
                        "name": current,
                        "current": True,
                        "remote": 'remotes/' in current
                    })
                else:
                    branch_name = line.strip()
                    branches.append({
                        "name": branch_name,
                        "current": False,
                        "remote": 'remotes/' in branch_name
                    })
        
        return {
            "success": True,
            "branches": branches,
            "current": current
        }
    
    def get_commits(self, limit: int = 10) -> Dict[str, Any]:
        """获取提交历史"""
        format_str = "%H|%an|%ad|%s"  # 哈希|作者|日期|主题
        result = self.run_git_command(f'log --pretty=format:"{format_str}" --date=short -{limit}')
        
        if not result["success"]:
            return result
        
        commits = []
        for line in result["stdout"].strip().split('\n'):
            if line:
                parts = line.split('|', 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    })
        
        return {
            "success": True,
            "commits": commits,
            "count": len(commits)
        }
    
    def create_patch(self, commit_hash: str = None) -> Dict[str, Any]:
        """创建补丁"""
        if commit_hash:
            result = self.run_git_command(f"format-patch {commit_hash}^..{commit_hash}")
        else:
            # 创建当前未提交更改的补丁
            result = self.run_git_command("diff HEAD")
        
        if not result["success"]:
            return result
        
        return {
            "success": True,
            "patch": result["stdout"],
            "patch_file": result["stdout"] if commit_hash else "当前更改的差异"
        }
    
    def apply_patch(self, patch_content: str) -> Dict[str, Any]:
        """应用补丁"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_content)
            patch_file = f.name
        
        try:
            result = self.run_git_command(f"apply {patch_file}")
            
            if result["success"]:
                print("✅ 补丁应用成功")
            else:
                print(f"❌ 补丁应用失败: {result.get('stderr')}")
            
            return result
            
        finally:
            # 清理临时文件
            try:
                os.unlink(patch_file)
            except:
                pass
    
    def reset_changes(self, hard: bool = False) -> Dict[str, Any]:
        """重置更改"""
        if hard:
            return self.run_git_command("reset --hard HEAD")
        else:
            return self.run_git_command("reset HEAD")
    
    def stash_changes(self, message: str = None) -> Dict[str, Any]:
        """暂存更改"""
        if message:
            return self.run_git_command(f'stash push -m "{message}"')
        else:
            return self.run_git_command("stash")
    
    def pop_stash(self) -> Dict[str, Any]:
        """弹出暂存"""
        return self.run_git_command("stash pop")
    
    def get_diff(self, file_path: str = None) -> Dict[str, Any]:
        """获取差异"""
        if file_path:
            return self.run_git_command(f"diff {file_path}")
        else:
            return self.run_git_command("diff")
    
    def is_git_repository(self) -> bool:
        """检查是否是Git仓库"""
        result = self.run_git_command("rev-parse --git-dir")
        return result["success"]
    
    def get_repo_info(self) -> Dict[str, Any]:
        """获取仓库信息"""
        info = {}
        
        # 获取远程URL
        remote_result = self.run_git_command("remote -v")
        if remote_result["success"]:
            info["remotes"] = remote_result["stdout"].strip().split('\n')
        
        # 获取当前分支
        branch_result = self.run_git_command("branch --show-current")
        if branch_result["success"]:
            info["current_branch"] = branch_result["stdout"].strip()
        
        # 获取提交数量
        count_result = self.run_git_command("rev-list --count HEAD")
        if count_result["success"]:
            info["commit_count"] = int(count_result["stdout"].strip())
        
        # 检查是否有未提交的更改
        status_result = self.get_status()
        if status_result["success"]:
            info["has_changes"] = status_result["has_changes"]
            info["changed_files"] = status_result["files"]
        
        return {
            "success": True,
            "info": info,
            "is_git_repo": self.is_git_repository()
        }

# 创建默认实例
git_tools = GitTools()