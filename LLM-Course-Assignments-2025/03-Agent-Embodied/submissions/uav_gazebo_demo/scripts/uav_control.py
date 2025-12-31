#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Trigger, TriggerResponse


UAV_MODEL = "uav_dummy"

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

class UAVControl:
    def __init__(self):
        self.target = None
        self.reached_pub = rospy.Publisher("/uav/reached", Bool, queue_size=1)
        self.pose_pub = rospy.Publisher("/uav/pose", PoseStamped, queue_size=1)

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.Subscriber("/uav/target_pose", PoseStamped, self.cb_target, queue_size=1)

        self.tol = rospy.get_param("~tol", 0.2)
        self.step = rospy.get_param("~step", 0.15)  # 每次逼近的距离（越大越快）
        self.rate = rospy.Rate(20)
        self.reset_srv = rospy.Service("/uav/reset_target", Trigger, self.on_reset_target)
        
    def on_reset_target(self, req):
        # 清空目标点，避免 reset 后继续飞向旧目标
        self.target = None
        try:
            self.reached_pub.publish(Bool(data=False))
        except Exception:
            pass
        return TriggerResponse(success=True, message="UAV reset: target cleared")

    def cb_target(self, msg):
        self.target = msg
        self.reached_pub.publish(Bool(data=False))

    def run(self):
        while not rospy.is_shutdown():
            # 发布当前位姿
            cur = self.get_state(UAV_MODEL, "world").pose
            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = "world"
            ps.pose = cur
            self.pose_pub.publish(ps)

            if self.target is None:
                self.rate.sleep()
                continue

            tx, ty, tz = (self.target.pose.position.x,
                          self.target.pose.position.y,
                          self.target.pose.position.z)

            cx, cy, cz = (cur.position.x, cur.position.y, cur.position.z)

            d = dist((cx,cy,cz), (tx,ty,tz))
            if d <= self.tol:
                self.reached_pub.publish(Bool(data=True))
                self.rate.sleep()
                continue

            # 向目标走一步（插值）
            k = min(1.0, self.step / max(d, 1e-6))
            nx = cx + (tx - cx) * k
            ny = cy + (ty - cy) * k
            nz = cz + (tz - cz) * k

            st = ModelState()
            st.model_name = UAV_MODEL
            st.pose = cur
            st.pose.position.x = nx
            st.pose.position.y = ny
            st.pose.position.z = nz
            self.set_state(st)

            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node("uav_control")
    UAVControl().run()

