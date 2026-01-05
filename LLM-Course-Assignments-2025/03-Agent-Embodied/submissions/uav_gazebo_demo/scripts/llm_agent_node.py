#!/usr/bin/env python3
import json
import rospy
from std_msgs.msg import String
import re

def build_default_delivery_plan():
    # 你要求的慢动作：等5s -> 起飞1m -> 前飞5m/5s -> 降0.5m悬停3s投包
    # -> 升1m -> 再前飞5m/5s -> 降落
    return {
        "task": "uav_delivery",
        "safety": {"z_safe": 1.2},
        "steps": [
            {"tool": "reset", "args": {}},
            {"tool": "hover", "args": {"seconds": 5}},
            {"tool": "takeoff", "args": {"x": 0.0, "y": 0.0, "z": 1.0, "timeout": 10.0}},
            {"tool": "goto", "args": {"x": 5.0, "y": 0.0, "z": 1.0, "timeout": 6.0}},
            {"tool": "goto", "args": {"x": 5.0, "y": 0.0, "z": 0.5, "timeout": 4.0}},
            {"tool": "hover", "args": {"seconds": 3}},
            {"tool": "drop_payload", "args": {}},
            {"tool": "goto", "args": {"x": 5.0, "y": 0.0, "z": 1.0, "timeout": 4.0}},
            {"tool": "goto", "args": {"x": 10.0, "y": 0.0, "z": 1.0, "timeout": 6.0}},
            {"tool": "land", "args": {"x": 10.0, "y": 0.0, "z": 0.2, "timeout": 8.0}}
        ]
    }

class AgentStub:
    def __init__(self):
        self.pub_plan = rospy.Publisher("/agent/plan", String, queue_size=1)
        rospy.Subscriber("/agent/command", String, self.on_cmd, queue_size=1)
        rospy.loginfo("llm_agent_node (stub) ready. Publish a sentence to /agent/command")

    def on_cmd(self, msg: String):
        text = msg.data.strip()
        rospy.loginfo(f"[agent] command: {text}")

        m = re.search(r"起飞到\s*([0-9.]+)\s*(m|米)", text)
        z = float(m.group(1)) if m else 1.0

        m = re.search(r"向前飞\s*([0-9.]+)\s*(m|米)\s*用时\s*([0-9.]+)\s*秒", text)
        d = float(m.group(1)) if m else 5.0
        t = float(m.group(3)) if m else 5.0

        do_drop = any(k in text for k in ["投包", "扔包", "投送", "空投", "drop"])
        do_land = ("降落" in text) or ("落地" in text)

        plan = {
          "task": "uav_manual",
          "steps": [
            {"tool":"reset","args":{}},
            {"tool":"takeoff","args":{"x":0.0,"y":0.0,"z":z,"timeout":10.0}},
            {"tool":"goto","args":{"x":d,"y":0.0,"z":z,"timeout":max(1.0,t)}},
          ] + ([{"tool":"drop_payload","args":{}}] if do_drop else []) + (
            [{"tool":"land","args":{"x":d,"y":0.0,"z":0.2,"timeout":8.0}}] if do_land else []
          )
        }

        self.pub_plan.publish(String(data=json.dumps(plan, ensure_ascii=False)))
        rospy.loginfo("[agent] plan published to /agent/plan")

if __name__ == "__main__":
    rospy.init_node("llm_agent_node")
    AgentStub()
    rospy.spin()
# v0.1.1 Mon 29 Dec 2025 01:23:25 AM PST
