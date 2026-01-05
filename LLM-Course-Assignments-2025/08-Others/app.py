"""
气象智能对话系统 - 主程序
整合HTML、CSS和JavaScript的完整版本
"""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 导入你的智能体系统
try:
    from agent import MultiAgentSystem
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 警告：无法导入agent模块: {e}")
    print("⚠️ 将使用模拟模式运行")
    AGENT_AVAILABLE = False

# 创建FastAPI应用
app = FastAPI(
    title="气象智能对话系统",
    description="基于四智能体协作的气象分析与决策系统",
    version="2.0.0"
)

# 创建必要的目录
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# 确保目录存在
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ==================== 聊天会话管理 ====================

class ChatSession:
    """聊天会话类"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()

        # 添加系统欢迎消息
        self.add_message(
            role="assistant",
            content="👋 你好！我是气象智能助手，我可以帮你分析各种天气情况并提供专业建议。请问有什么可以帮助您的？"
        )

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """添加消息到会话"""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # "user", "assistant", "system"
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_active = datetime.now()
        return message

    def get_messages(self, limit: int = 50) -> List[Dict]:
        """获取消息历史"""
        return self.messages[-limit:] if self.messages else []

    def clear(self):
        """清空会话"""
        self.messages.clear()
        self.add_message(
            role="assistant",
            content="🗑️ 对话已重置。我是气象智能助手，我可以帮你分析各种天气情况。请问有什么可以帮助您的？"
        )

class ChatManager:
    """聊天管理器"""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.system = None

        # 初始化智能体系统
        if AGENT_AVAILABLE:
            try:
                self.system = MultiAgentSystem()
                print("✅ 智能体系统初始化成功")
            except Exception as e:
                print(f"❌ 智能体系统初始化失败: {e}")
                self.system = None

    def create_session(self, session_id: str = None) -> str:
        """创建新会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]  # 使用简短的ID

        self.sessions[session_id] = ChatSession(session_id)
        print(f"📝 创建新会话: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取会话"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️ 删除会话: {session_id}")

# 全局聊天管理器实例
chat_manager = ChatManager()

# ==================== 路由定义 ====================

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """首页 - 聊天界面"""
    # 创建新会话
    session_id = chat_manager.create_session()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session_id": session_id,
            "page_title": "气象智能对话系统",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str):
    """聊天页面（指定会话）"""
    session = chat_manager.get_session(session_id)

    if not session:
        # 如果会话不存在，创建新会话
        session_id = chat_manager.create_session()
        session = chat_manager.get_session(session_id)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session_id": session_id,
            "page_title": "气象智能对话系统",
            "initial_messages": session.get_messages(limit=20),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.post("/api/chat/send")
async def send_message(request: Request):
    """发送消息API"""
    try:
        data = await request.json()
        session_id = data.get("session_id", "")
        message = data.get("message", "").strip()

        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "消息内容不能为空"}
            )

        # 获取或创建会话
        session = chat_manager.get_session(session_id)
        if not session:
            session_id = chat_manager.create_session()
            session = chat_manager.get_session(session_id)

        # 添加用户消息
        user_msg = session.add_message("user", message)

        # 处理消息（使用智能体系统）
        response_content = ""
        metadata = {}

        if chat_manager.system and AGENT_AVAILABLE:
            try:
                # 调用智能体系统处理消息
                result = chat_manager.system.process_query(message)

                if result.get("success"):
                    response_content = result.get("response", "✅ 分析完成")
                    metadata = {
                        "confidence": result.get("confidence", 0.0),
                        "processing_time": "0.5s",
                        "source": "智能体系统"
                    }
                else:
                    response_content = f"❌ 处理失败: {result.get('error', '未知错误')}"
                    metadata = {"error": True}

            except Exception as e:
                response_content = f"⚠️ 系统错误: {str(e)}"
                metadata = {"error": True}
        else:
            # 模拟模式
            response_content = f"🤖 收到你的消息: '{message}'\n\n"
            response_content += "🔍 检索智能体: 正在检索相关气象知识...\n"
            response_content += "📊 分析智能体: 分析气象特征中...\n"
            response_content += "💡 决策智能体: 生成应对建议...\n"
            response_content += "👥 协调智能体: 整合最终结果...\n\n"
            response_content += "✅ 分析完成！\n"
            response_content += f"💡 建议: 根据'{message}'，建议关注当地气象预警，做好相应防护措施。"
            metadata = {"confidence": 0.8, "processing_time": "0.3s", "source": "模拟模式"}

        # 添加助手回复
        assistant_msg = session.add_message("assistant", response_content, metadata)

        return {
            "success": True,
            "session_id": session_id,
            "user_message": user_msg,
            "assistant_message": assistant_msg
        }

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "无效的JSON数据"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"服务器错误: {str(e)}"}
        )

@app.post("/api/chat/clear")
async def clear_chat(request: Request):
    """清空对话API"""
    data = await request.json()
    session_id = data.get("session_id", "")

    session = chat_manager.get_session(session_id)
    if session:
        session.clear()
        return {"success": True, "message": "对话已清空"}

    return {"success": False, "error": "会话不存在"}

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """获取对话历史API"""
    session = chat_manager.get_session(session_id)
    if not session:
        return {"error": "会话不存在", "history": []}

    return {
        "session_id": session_id,
        "history": session.get_messages(limit=limit),
        "message_count": len(session.messages),
        "created_at": session.created_at.isoformat(),
        "last_active": session.last_active.isoformat()
    }

@app.get("/api/system/status")
async def system_status():
    """系统状态API"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(chat_manager.sessions),
        "agent_available": AGENT_AVAILABLE,
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "time": datetime.now().isoformat()}

# ==================== WebSocket 支持 ====================

class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """连接WebSocket"""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        """发送消息到指定会话"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天接口"""
    # 连接WebSocket
    await manager.connect(websocket, session_id)

    # 获取或创建会话
    session = chat_manager.get_session(session_id)
    if not session:
        session_id = chat_manager.create_session()
        session = chat_manager.get_session(session_id)

    try:
        # 发送历史消息
        await websocket.send_json({
            "type": "history",
            "messages": session.get_messages(limit=20)
        })

        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "message":
                message_content = data.get("content", "").strip()

                if not message_content:
                    await websocket.send_json({
                        "type": "error",
                        "content": "消息不能为空"
                    })
                    continue

                # 添加用户消息
                user_msg = session.add_message("user", message_content)
                await websocket.send_json({
                    "type": "message",
                    "message": user_msg
                })

                # 处理消息（模拟进度）
                await websocket.send_json({
                    "type": "status",
                    "content": "🔍 检索智能体工作中...",
                    "progress": 25
                })

                # 使用智能体系统或模拟处理
                if chat_manager.system and AGENT_AVAILABLE:
                    result = chat_manager.system.process_query(message_content)
                    if result.get("success"):
                        response = result.get("response", "✅ 分析完成")
                    else:
                        response = f"❌ 处理失败: {result.get('error', '未知错误')}"
                else:
                    # 模拟回复
                    response = f"🤖 收到: '{message_content}'\n\n"
                    response += "✅ 已分析完成！这是一个模拟回复。\n"
                    response += "💡 实际系统中会调用四智能体进行专业分析。"

                # 添加助手回复
                assistant_msg = session.add_message("assistant", response, {
                    "confidence": 0.85,
                    "processing_time": "0.5s"
                })

                await websocket.send_json({
                    "type": "message",
                    "message": assistant_msg
                })

            elif message_type == "clear":
                # 清空对话
                session.clear()
                await websocket.send_json({
                    "type": "system",
                    "content": "🗑️ 对话已清空"
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.send_json({
            "type": "error",
            "content": f"连接错误: {str(e)}"
        })

# ==================== 启动服务器 ====================

if __name__ == "__main__":
    import uvicorn
    import socket
    import os

    print("=" * 60)
    print("🌤️ 气象智能对话系统 - 无root权限版")
    print("=" * 60)

    # 获取所有可用的IP地址
    print("📡 可用的访问方式:")
    print("")

    # 显示本地访问
    print("1. 🖥️  本地访问（在Linux服务器上）:")
    print("   http://localhost:8000")
    print("   curl http://localhost:8000/health")
    print("")

    # 显示可能的IP地址
    print("2. 🌐 外部访问（如果防火墙允许）:")
    try:
        # 获取主机名
        hostname = socket.gethostname()

        # 获取所有IP地址
        all_ips = []
        try:
            # 方法1：通过UDP连接获取外网IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            external_ip = s.getsockname()[0]
            all_ips.append(external_ip)
            s.close()
        except:
            pass

        # 方法2：获取所有网络接口的IP
        try:
            import netifaces

            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr.get('addr')
                        if ip and ip != '127.0.0.1':
                            all_ips.append(ip)
        except ImportError:
            # 如果netifaces不可用，使用socket
            pass

        # 去重并显示
        unique_ips = list(set(all_ips))
        for ip in unique_ips:
            print(f"   http://{ip}:8000")

        if not unique_ips:
            print("   ❌ 无法获取外部IP地址")
    except Exception as e:
        print(f"   ⚠️  获取IP地址失败: {e}")

    print("")
    print("3. 🚇 SSH隧道访问（推荐）:")
    print("   在Windows上运行:")
    print("   ssh -L 8000:localhost:8000 你的用户名@服务器IP")
    print("   然后在Windows浏览器访问: http://localhost:8000")
    print("")
    print("4. 🎯 测试命令:")
    print("   curl http://localhost:8000/health")
    print("")
    print("=" * 60)
    print("⏳ 服务器启动中...")
    print("🛑 按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 强制刷新输出
    import sys

    sys.stdout.flush()

    # 确保绑定到0.0.0.0
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)