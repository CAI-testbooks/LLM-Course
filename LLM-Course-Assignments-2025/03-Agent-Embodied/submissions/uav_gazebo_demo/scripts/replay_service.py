#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
import math

class ReplayService:
    def __init__(self):
        # 可配置：重播后等待一点时间，让 /clock 走起来
        self.wait_sec = float(rospy.get_param("~wait_sec", 0.5))
        
        # 模型名（要和 spawn 时的 -model 一致）
        self.uav_model = rospy.get_param("~uav_model", "uav_dummy")
        self.payload_model = rospy.get_param("~payload_model", "payload_box")

        # 初始位姿（按你 spawn 的 x/y/z 改）
        self.uav_init = rospy.get_param("~uav_init_xyz", [0.0, 0.0, 0.2])
        self.payload_init = rospy.get_param("~payload_init_xyz", [0.0, 0.0, 0.0])
        self.uav_init_yaw = float(rospy.get_param("~uav_init_yaw", 0.0))

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

    def _yaw_to_quat(self, yaw):
        # 简单 yaw -> quaternion（roll=pitch=0）
        half = yaw * 0.5
        return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))

    def _set_model_pose(self, model_name, xyz, yaw=0.0, timeout=2.0, optional=False):
        srv = "/gazebo/set_model_state"
        try:
            rospy.wait_for_service(srv, timeout=timeout)
            proxy = rospy.ServiceProxy(srv, SetModelState)

            st = ModelState()
            st.model_name = model_name
            st.reference_frame = "world"
            st.pose = Pose(position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                          orientation=self._yaw_to_quat(yaw))
            # twist 默认 0 即可
            resp = proxy(st)
            return True, f"{srv}({model_name}): {resp.success} {resp.status_message}"
        except Exception as e:
            if optional:
                return False, f"{srv}({model_name}): optional-missing ({e})"
            return False, f"{srv}({model_name}): FAILED ({e})"


    def on_replay(self, req):
        logs = []
        ok = True

        # 1) pause physics
        s, msg = self._call_trigger(self.srv_pause, optional=False)
        logs.append(msg); ok &= s

        # 2) reset world
        s, msg = self._call_trigger(self.srv_reset_world, optional=False)
        logs.append(msg); ok &= s

        # 2.5) 强制把模型放回初始位姿（解决“从上次终点飞回原点再开始”的问题）
        s, msg = self._set_model_pose(self.uav_model, self.uav_init, yaw=self.uav_init_yaw, optional=False)
        logs.append(msg); ok &= s
        s, msg = self._set_model_pose(self.payload_model, self.payload_init, yaw=0.0, optional=True)
        logs.append(msg)


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
