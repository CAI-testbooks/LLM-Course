#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger, TriggerResponse

class ReplayService:
    def __init__(self):
        # 可配置：重播后等待一点时间，让 /clock 走起来
        self.wait_sec = float(rospy.get_param("~wait_sec", 0.5))

        # 这些服务名固定（Gazebo + 你的工具）
        self.srv_pause = "/gazebo/pause_physics"
        self.srv_unpause = "/gazebo/unpause_physics"
        self.srv_reset_world = "/gazebo/reset_world"
        self.srv_payload_reset = "/payload/reset"
        self.srv_uav_reset = "/uav/reset_target"

        self.replay_srv = rospy.Service("/demo/replay", Trigger, self.on_replay)
        rospy.loginfo("Replay service ready: rosservice call /demo/replay")

    def _call_trigger(self, name, timeout=2.0, optional=False):
        try:
            rospy.wait_for_service(name, timeout=timeout)
            proxy = rospy.ServiceProxy(name, Trigger)
            resp = proxy()
            return True, f"{name}: {resp.success} {resp.message}"
        except Exception as e:
            if optional:
                return False, f"{name}: optional-missing ({e})"
            return False, f"{name}: FAILED ({e})"

    def on_replay(self, req):
        logs = []
        ok = True

        # 1) pause physics
        s, msg = self._call_trigger(self.srv_pause, optional=False)
        logs.append(msg); ok &= s

        # 2) reset world
        s, msg = self._call_trigger(self.srv_reset_world, optional=False)
        logs.append(msg); ok &= s

        # 3) tool resets（这两个如果你还没加服务，也不会致命：optional=True）
        s, msg = self._call_trigger(self.srv_payload_reset, optional=True)
        logs.append(msg)
        s2, msg2 = self._call_trigger(self.srv_uav_reset, optional=True)
        logs.append(msg2)

        # 4) unpause physics
        s, msg = self._call_trigger(self.srv_unpause, optional=False)
        logs.append(msg); ok &= s

        # 5) small wait
        try:
            rospy.sleep(self.wait_sec)
            logs.append(f"sleep {self.wait_sec}s")
        except Exception as e:
            logs.append(f"sleep failed: {e}")

        message = " | ".join(logs)
        return TriggerResponse(success=ok, message=message)

if __name__ == "__main__":
    rospy.init_node("replay_service")
    ReplayService()
    rospy.spin()
