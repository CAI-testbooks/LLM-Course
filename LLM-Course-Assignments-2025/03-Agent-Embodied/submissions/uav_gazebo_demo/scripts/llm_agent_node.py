#!/usr/bin/env python3
import json
import rospy
from std_msgs.msg import String

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

        # 规则：出现投包/扔包/投送/空投等关键词就走投送任务
        if any(k in text for k in ["投包", "扔包", "投送", "空投", "drop"]):
            plan = build_default_delivery_plan()
        else:
            plan = build_default_delivery_plan()

        self.pub_plan.publish(String(data=json.dumps(plan, ensure_ascii=False)))
        rospy.loginfo("[agent] plan published to /agent/plan")

if __name__ == "__main__":
    rospy.init_node("llm_agent_node")
    AgentStub()
    rospy.spin()
# v0.1.1 Mon 29 Dec 2025 01:23:25 AM PST
