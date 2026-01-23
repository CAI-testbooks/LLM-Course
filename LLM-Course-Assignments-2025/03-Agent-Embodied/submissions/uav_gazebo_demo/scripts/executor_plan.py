#!/usr/bin/env python3
import os, json, time
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

class ExecutorPlan:
    def __init__(self):
        self.reached = False
        rospy.Subscriber("/uav/reached", Bool, self.cb_reached, queue_size=1)
        self.pub_target = rospy.Publisher("/uav/target_pose", PoseStamped, queue_size=1)

        rospy.wait_for_service("/payload/drop")
        self.drop_srv = rospy.ServiceProxy("/payload/drop", Trigger)

        # 可选：重播时复位逻辑（如果你按后面步骤加了服务就会自动启用）
        self.payload_reset = None
        self.uav_reset_target = None
        try:
            rospy.wait_for_service("/payload/reset", timeout=0.5)
            self.payload_reset = rospy.ServiceProxy("/payload/reset", Trigger)
        except Exception:
            pass
        try:
            rospy.wait_for_service("/uav/reset_target", timeout=0.5)
            self.uav_reset_target = rospy.ServiceProxy("/uav/reset_target", Trigger)
        except Exception:
            pass

        self.trace_dir = rospy.get_param("~trace_dir", "$(find uav_gazebo_demo)/results/traces")
        # 兼容：如果你没有 rosparam 替换 find，这里就用相对目录
        if "$(find" in self.trace_dir:
            self.trace_dir = os.path.join(os.getcwd(), "results", "traces")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.trace_path = os.path.join(self.trace_dir, f"trace_plan_{int(time.time())}.jsonl")
        self.step_id = 0

        rospy.Subscriber("/agent/plan", String, self.on_plan, queue_size=1)
        rospy.loginfo("executor_plan ready. Waiting plan on /agent/plan ...")

    def cb_reached(self, msg: Bool):
        self.reached = msg.data

    def log(self, tool, args, ok=True, reason=""):
        self.step_id += 1
        rec = {
            "t": rospy.Time.now().to_sec(),
            "step": self.step_id,
            "tool": tool,
            "args": args,
            "ok": ok,
            "reason": reason
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rospy.loginfo(f"[STEP {self.step_id}] {tool} {args} ok={ok} {reason}")

    def goto(self, x, y, z, timeout):
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)

        self.reached = False
        t0 = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            ps.header.stamp = rospy.Time.now()
            self.pub_target.publish(ps)
            if self.reached:
                return True, ""
            if time.time() - t0 > timeout:
                return False, "TIMEOUT"
            rate.sleep()

    def exec_step(self, step: dict):
        tool = step.get("tool", "")
        args = step.get("args", {}) or {}

        if tool == "reset":
            if self.payload_reset:
                self.payload_reset()
            if self.uav_reset_target:
                self.uav_reset_target()
            self.log("reset", {}, True, "")
            return True

        if tool == "hover":
            sec = float(args.get("seconds", 1.0))
            rospy.sleep(sec)
            self.log("hover", {"seconds": sec}, True, "")
            return True

        if tool in ["takeoff", "goto", "land"]:
            x = float(args.get("x", 0.0))
            y = float(args.get("y", 0.0))
            z = float(args.get("z", 1.0))
            duration = float(args.get("duration", 5.0))
            timeout = float(args.get("timeout", duration + 1.0))
            ok, reason = self.goto(x, y, z, timeout=timeout)
            self.log(tool, {"x": x, "y": y, "z": z, "timeout": timeout}, ok, reason)
            return ok

        if tool == "drop_payload":
            try:
                resp = self.drop_srv()
                self.log("drop_payload", {}, resp.success, resp.message)
                return resp.success
            except Exception as e:
                self.log("drop_payload", {}, False, str(e))
                return False

        self.log("unknown", step, False, "unknown tool")
        return False

    def on_plan(self, msg: String):
        try:
            plan = json.loads(msg.data)
        except Exception as e:
            self.log("plan_parse", {"raw": msg.data[:200]}, False, str(e))
            return

        steps = plan.get("steps", [])
        z_safe = float(plan.get("safety", {}).get("z_safe", 1.2))
        rospy.loginfo(f"Received plan: {len(steps)} steps, z_safe={z_safe}")

        for i, st in enumerate(steps, 1):
            ok = self.exec_step(st)
            if ok:
                continue

            # 简单错误恢复：goto 失败 -> 升到安全高度 -> 重试一次
            if st.get("tool") in ["goto", "takeoff", "land"]:
                args = st.get("args", {}) or {}
                x = float(args.get("x", 0.0)); y = float(args.get("y", 0.0))
                self.log("recovery", {"action": "ascend_safe", "x": x, "y": y, "z_safe": z_safe}, True, "")
                self.goto(x, y, z_safe, timeout=3.0)
                ok2 = self.exec_step(st)
                if ok2:
                    continue

            self.log("abort", {"failed_step": i, "step": st}, False, "mission aborted")
            return

        self.log("done", {}, True, "plan finished")

if __name__ == "__main__":
    rospy.init_node("executor_plan")
    ExecutorPlan()
    rospy.spin()
