#!/usr/bin/env python3
import os, json, time
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

def now():
    return rospy.Time.now().to_sec()

class ExecutorDemo:
    def __init__(self):
        self.reached = False
        rospy.Subscriber("/uav/reached", Bool, lambda m: setattr(self, "reached", m.data), queue_size=1)
        self.pub = rospy.Publisher("/uav/target_pose", PoseStamped, queue_size=1)

        rospy.wait_for_service("/payload/drop")
        self.drop_srv = rospy.ServiceProxy("/payload/drop", Trigger)

        # 任务参数（你也可以改成从文件读）
        self.z_cruise = rospy.get_param("~z_cruise", 10.0)
        self.z_drop   = rospy.get_param("~z_drop", 3.0)
        self.drop_xy  = rospy.get_param("~drop_xy", [10.0, -3.0])
        self.goal_xy  = rospy.get_param("~goal_xy", [20.0, 5.0])

        self.trace_dir = rospy.get_param("~trace_dir", "results/traces")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.trace_path = os.path.join(self.trace_dir, f"trace_{int(time.time())}.jsonl")
        self.step_id = 0

    def log(self, action, args, ok=True, reason=""):
        self.step_id += 1
        rec = {
            "t": now(),
            "step": self.step_id,
            "action": {"tool": action, "args": args},
            "result": {"ok": ok, "reason": reason}
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rospy.loginfo(f"[STEP {self.step_id}] {action} {args} -> ok={ok} {reason}")

    def goto(self, x, y, z, timeout=60.0):
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
            self.pub.publish(ps)
            if self.reached:
                return True, ""
            if time.time() - t0 > timeout:
                return False, "TIMEOUT"
            rate.sleep()

    def run(self):
        # ====== 0) 起点等待 5 秒 ======
        rospy.sleep(5.0)
        self.log("HOVER", {"t": 5.0, "note": "wait at start"}, True, "")

        # 设定关键点：起点(0,0), 中点(5,0), 终点(10,0)
        x0, y0 = 0.0, 0.0
        x1, y1 = 5.0, 0.0
        x2, y2 = 10.0, 0.0

        z_takeoff = 1.0
        z_low = 0.5

        # ====== 1) 起飞到 1m ======
        ok, reason = self.goto(x0, y0, z_takeoff, timeout=10.0)
        self.log("TAKEOFF", {"x": x0, "y": y0, "z": z_takeoff}, ok, reason)
        if not ok: return

        # ====== 2) 向前飞 5m，用时 5 秒 ======
        # timeout 设 6s 给一点裕度
        ok, reason = self.goto(x1, y1, z_takeoff, timeout=6.0)
        self.log("MOVE_TO", {"x": x1, "y": y1, "z": z_takeoff, "expect_s": 5.0}, ok, reason)
        if not ok: return

        # ====== 3) 降到 0.5m ======
        ok, reason = self.goto(x1, y1, z_low, timeout=6.0)
        self.log("DESCEND", {"x": x1, "y": y1, "z": z_low}, ok, reason)
        if not ok: return

        # ====== 4) 0.5m 悬停 3 秒 ======
        rospy.sleep(3.0)
        self.log("HOVER", {"t": 3.0, "z": z_low}, True, "")

        # ====== 5) 扔包 ======
        try:
            resp = self.drop_srv()
            self.log("DROP_PAYLOAD", {}, resp.success, resp.message)
            if not resp.success:
                return
        except Exception as e:
            self.log("DROP_PAYLOAD", {}, False, str(e))
            return

        # ====== 6) 升高到 1m ======
        ok, reason = self.goto(x1, y1, z_takeoff, timeout=6.0)
        self.log("ASCEND", {"x": x1, "y": y1, "z": z_takeoff}, ok, reason)
        if not ok: return

        # ====== 7) 再向前飞 5m，用时 5 秒 ======
        ok, reason = self.goto(x2, y2, z_takeoff, timeout=6.0)
        self.log("MOVE_TO", {"x": x2, "y": y2, "z": z_takeoff, "expect_s": 5.0}, ok, reason)
        if not ok: return

        # ====== 8) 最后降落 ======
        # UAV 模型初始 z=0.2，为避免插进地面，降到 0.2 更稳
        ok, reason = self.goto(x2, y2, 0.2, timeout=8.0)
        self.log("LAND", {"x": x2, "y": y2, "z": 0.2}, ok, reason)
        if not ok: return

        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"terminal": True, "success": True, "t": now()}, ensure_ascii=False) + "\n")
        rospy.loginfo(f"Mission done. Trace saved: {self.trace_path}")


if __name__ == "__main__":
    rospy.init_node("executor_demo")
    ExecutorDemo().run()

