import random
import time


def communicate_uav_states(uav_states, uav_ids, comm_delay=0.1, loss_rate=0.1):
    """
    模拟无人机集群通信：传递状态，带延迟和丢包
    """
    # 模拟通信延迟
    time.sleep(comm_delay)

    # 模拟丢包
    transmitted_states = {}
    for uav_id, state in zip(uav_ids, uav_states):
        if random.random() > loss_rate:  # 成功传输
            transmitted_states[uav_id] = state
        else:  # 丢包
            transmitted_states[uav_id] = None

    return transmitted_states


# 测试通信模拟（可选执行）
if __name__ == "__main__":
    import numpy as np

    uav_states = [np.random.rand(8) for _ in range(3)]
    uav_ids = ["UAV0", "UAV1", "UAV2"]
    comm_result = communicate_uav_states(uav_states, uav_ids)
    print("通信结果（None表示丢包）：", comm_result)