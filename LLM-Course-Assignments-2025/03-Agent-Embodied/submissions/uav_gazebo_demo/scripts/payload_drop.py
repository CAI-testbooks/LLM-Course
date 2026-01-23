#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState

UAV_MODEL = "uav_dummy"
PAYLOAD_MODEL = "payload_box"

class PayloadDrop:
    def __init__(self):
        self.attached = True
        self.offset_z = rospy.get_param("~offset_z", -0.25)  # 包裹在机体下方

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.srv = rospy.Service("/payload/drop", Trigger, self.on_drop)
        self.rate = rospy.Rate(20)

    def on_drop(self, req):
        self.attached = False
        return TriggerResponse(success=True, message="Payload released (stop following UAV).")

    def run(self):
        while not rospy.is_shutdown():
            if self.attached:
                uav_pose = self.get_state(UAV_MODEL, "world").pose
                st = ModelState()
                st.model_name = PAYLOAD_MODEL
                st.pose = uav_pose
                st.pose.position.z = uav_pose.position.z + self.offset_z
                self.set_state(st)
            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node("payload_drop")
    PayloadDrop().run()

