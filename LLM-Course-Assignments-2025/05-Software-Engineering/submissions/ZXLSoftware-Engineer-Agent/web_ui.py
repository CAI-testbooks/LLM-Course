from flask import Flask, request, jsonify, send_from_directory, render_template
from pathlib import Path
from puding_agent.agent import GeminiEngineer
from puding_agent.utils import ConversationMessage
import os
import logging

app = Flask(__name__, static_folder="static", template_folder="templates")
# Initialize the Gemini Engineer Agent
engineer = GeminiEngineer()
sessions = {}
current_session = "default"
sessions[current_session] = engineer.conversation_history

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/send", methods=["POST"])
def api_send():
    data = request.get_json(silent=True) or {}
    logging.info(f"Received /api/send request. Data: {data}")
    base_text = data.get("text", "")
    flags = data.get("flags") or {}
    prefix = ""
    if flags.get("reflect"):
        prefix += "请在每次生成后进行反思，指出不足并改进。"
    if flags.get("tools"):
        prefix += "必须通过工具调用执行相关文件操作与命令。"
    if flags.get("tests"):
        prefix += "为代码自动编写单元测试并运行验证，如果失败请修复。"
    text = (prefix + "\n" + base_text).strip()
    if not text:
        logging.warning("Empty input text")
        return jsonify({"error": "empty_input"}), 400
    
    try:
        logging.info("Calling engineer.respond_once...")
        
        def loop_callback(count):
            logging.info(f"Engineer loop iteration: {count}")
            
        result = engineer.respond_once(text, on_loop_start=loop_callback)
        logging.info(f"respond_once result: {result}")
        
        if isinstance(result, dict) and result.get("error"):
            logging.error(f"Error from respond_once: {result.get('error')}")
            return jsonify(result), 500
            
        sessions[current_session] = engineer.conversation_history
        return jsonify(result)
    except Exception as e:
        logging.error(f"Exception in api_send: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    msgs = [{"role": m.role, "content": m.content} for m in sessions.get(current_session, [])]
    return jsonify({"messages": msgs})

@app.route("/api/add", methods=["POST"])
def api_add():
    data = request.get_json(silent=True) or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "empty_path"}), 400
    try:
        engineer.add_file_to_context(path)
        sessions[current_session] = engineer.conversation_history
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/list", methods=["GET"])
def api_list():
    path = request.args.get("path", ".")
    result = engineer.execute_tool("list_directory", {"dir_path": path})
    return jsonify(result)

@app.route("/api/read", methods=["GET"])
def api_read():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "empty_path"}), 400
    result = engineer.execute_tool("read_file", {"file_path": path})
    return jsonify(result)

@app.route("/api/status", methods=["GET"])
def api_status():
    provider = getattr(engineer, "provider", None)
    model_name = getattr(engineer, "model_name", None) or os.getenv("GEMINI_MODEL")
    return jsonify({"provider": provider, "model": model_name})

@app.route("/api/session/new", methods=["POST"])
def api_session_new():
    name = (request.get_json(silent=True) or {}).get("name") or ""
    sid = f"s{len(sessions)+1}"
    sessions[sid] = [ConversationMessage("system", engineer.conversation_history[0].content)]
    return jsonify({"id": sid, "name": name})

@app.route("/api/session/list", methods=["GET"])
def api_session_list():
    items = []
    for sid, msgs in sessions.items():
        items.append({"id": sid, "count": len(msgs)})
    return jsonify({"items": items, "current": current_session})

@app.route("/api/session/select", methods=["POST"])
def api_session_select():
    data = request.get_json(silent=True) or {}
    sid = data.get("id")
    if sid not in sessions:
        return jsonify({"error": "not_found"}), 404
    global current_session
    current_session = sid
    engineer.conversation_history = sessions[sid]
    return jsonify({"current": current_session})

@app.route("/api/session/clear", methods=["POST"])
def api_session_clear():
    global current_session
    sessions[current_session] = [ConversationMessage("system", engineer.conversation_history[0].content)]
    engineer.conversation_history = sessions[current_session]
    return jsonify({"success": True})

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
