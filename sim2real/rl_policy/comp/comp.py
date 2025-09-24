import os
import sys
import time
from xml.parsers.expat import model

import numpy as np
import argparse
import yaml
from scipy.linalg import logm, expm, eigh

sys.path.append("../")
sys.path.append("./rl_policy")

import pinocchio as pin
from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy
from pinocchio import casadi as cpin

from termcolor import colored
from sim2real.utils.arm_ik.robot_arm_ik_h1 import H1_ArmIK


class CompPolicy(LocoManipPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.arm_ik = H1_ArmIK(robot_config=config, unit_test=False, visualization=False)
        self.torque_log = []  # 新增：用于记录实际测量力矩
        self.base_vel_log = []  # 实测base线速度
        self.target_vel_log = []  # 目标线速度命令

        self.calc_upper_torque = np.zeros(8)  # 上肢8个关节力矩

        # 新增：用于积分计算base线速度
        self.integrated_base_vel = np.zeros(3)  # x, y, z线速度的积分值
        self.prev_time = None  # 记录上一次的时间戳
        self.dt = 1.0 / 50.0  # 控制频率，50Hz

        self.start_record_time = None
        self.record_delay = 10.0  # 10秒后开始记录
        self.record_duration = 10.0  # 记录10秒的数据

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)

        # # 正确提取上半身关节力矩
        # tau_est_start = 7 + 2 * self.num_dofs + 6  # tau_est 的起始位置
        # joint_torque_start = tau_est_start + 6  # 跳过 base 的 6 个力矩分量
        # # 提取所有关节力矩
        # all_joint_torques = robot_state_data[:, joint_torque_start : joint_torque_start + self.num_dofs]
        # # 只取上半身关节的力矩
        # current_obs_buffer_dict["upper_torq"] = all_joint_torques[:, self.upper_dof_indices]

        current_obs_buffer_dict["upper_torq"] = self.calc_upper_torque.reshape(1, -1)

        # print("Measured upper body torques:", current_obs_buffer_dict["upper_torq"])

        return current_obs_buffer_dict

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data

        current_time = time.time()

        # Manually set shoulder joints
        shoulder_joint_indices = [11, 15]
        for idx in shoulder_joint_indices:
            self.ref_upper_dof_pos[0, idx - 11] = -0.5
        elbow_joint_indices = [14, 18]
        for idx in elbow_joint_indices:
            self.ref_upper_dof_pos[0, idx - 11] = 0.5

        # print("ref upper body pos:", self.ref_upper_dof_pos)

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = scaled_policy_action + self.default_dof_angles
            if self.start_record_time is None:
                self.start_record_time = current_time
                print("Starting time:", self.start_record_time)
        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # # Send command
        # cmd_q = q_target[0]
        # self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs]) # for PD control

        # === PD control for tau ===
        # 当前upper_body关节位置和速度
        q_cur = robot_state_data[0, 7 : 7 + self.num_dofs][self.upper_dof_indices] # 
        # dq_cur = robot_state_data[0, 13 + self.num_dofs : 13 + 2 * self.num_dofs]
        # print("target upper body pos:", q_target[0][self.upper_dof_indices])
        # print("current upper body pos:", q_cur)

        # PD参数（可根据实际机器人调整）
        kp = np.ones(q_cur.shape) * 100.0
        # kd = np.ones(q_cur.shape) * 2.0

        # calculate stiffness matrix
        mat_stiff = self._compute_rvc_matrix(q_cur, kp)

        # 计算力矩
        q_target_up = q_target[0][self.upper_dof_indices]
        calc_tau_pd = kp * (q_target_up - q_cur)
        calc_tau = mat_stiff @ (q_target_up - q_cur)

        # 上半身重力补偿
        # 只用上半身关节，速度和加速度都为0
        grav_tau_full = pin.rnea(
            self.arm_ik.reduced_model,
            self.arm_ik.reduced_data,
            q_cur,
            np.zeros_like(q_cur),
            np.zeros_like(q_cur)
        )
        calc_tau += grav_tau_full

        self.calc_upper_torque = calc_tau  # 保存计算的上肢力矩

        # calc_tau -= kd * dq_cur
        # cmd_tau[self.upper_dof_indices] = calc_tau
        cmd_tau[self.left_arm_dof_indices] = calc_tau[:4] # 只对左臂关节分配力矩 # calc_tau[:4]
        cmd_tau[self.right_arm_dof_indices] = calc_tau[4:] # 只对右臂关节分配力矩 # calc_tau_pd[4:]

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, q_cur) # for PD control

        base_lin_acc = robot_state_data[0, 19 + 3 * self.num_dofs : 19 + 3 * self.num_dofs + 3]  # x, y, z线加速度
        # 通过积分计算base线速度
        if self.prev_time is not None:
            actual_dt = current_time - self.prev_time
            # 使用实际的时间步长进行积分（梯形积分）
            self.integrated_base_vel += base_lin_acc * actual_dt
        else:
            # 第一次调用，使用固定时间步长
            self.integrated_base_vel += base_lin_acc * self.dt
        
        self.prev_time = current_time

        # 使用积分得到的base线速度
        base_lin_vel = self.integrated_base_vel

        target_lin_vel = self.lin_vel_command[0, :2]  # 只记录x, y方向的命令
        # 记录数据
        # 只有过了延迟时间才开始记录
        if self.start_record_time is not None and self.record_delay <= current_time - self.start_record_time < self.record_delay + self.record_duration:
            print("Recording data...")
            measured_tau = robot_state_data[0, 7 + 6 + 2*self.num_dofs + 6 : 19 + 3*self.num_dofs][self.upper_dof_indices]
            self.torque_log.append(measured_tau.copy())
            self.base_vel_log.append(base_lin_vel[:2].copy())  # 只记录x, y方向
            self.target_vel_log.append(target_lin_vel.copy())

        # print("base_lin_vel:", base_lin_vel)

    #################################
    # Compliance control functions #
    #################################

    def _compute_rvc_matrix(self, q_cur, kp):

        """Compute the RVC stiffness matrix based on the current robot state."""
        # 使用 H1_ArmIK 的 reduced_model 和 reduced_data
        # 假设 self.arm_ik 已在 __init__ 初始化为 H1_ArmIK 实例
        model = self.arm_ik.reduced_model
        data = self.arm_ik.reduced_data

        # 获取 torso link 和末端 frame 的 id
        # torso_frame_name = "torso_link"  # 请替换为你模型实际的 torso link 名称
        left_ee_frame_name = "left_elbow_ee"   # 请替换为实际左手末端 frame 名称 # left_elbow_ee
        right_ee_frame_name = "right_elbow_ee" # 请替换为实际右手末端 frame 名称 # right_elbow_ee

        left_ee_frame_id = model.getFrameId(left_ee_frame_name)
        right_ee_frame_id = model.getFrameId(right_ee_frame_name)

        # 更新当前关节位置
        pin.forwardKinematics(model, data, q_cur)
        pin.updateFramePlacements(model, data)

        # 计算左手和右手末端在世界系下的雅可比
        J_left_world = pin.computeFrameJacobian(model, data, q_cur, left_ee_frame_id, pin.ReferenceFrame.WORLD)  # (6, nq)
        J_right_world = pin.computeFrameJacobian(model, data, q_cur, right_ee_frame_id, pin.ReferenceFrame.WORLD)  # (6, nq)

        # 只取位置部分 (前3行)
        J_left_pos = J_left_world[:3, :]
        J_right_pos = J_right_world[:3, :]

        # 合并雅可比 (6, nq)
        J_hands_pos = np.vstack([J_left_pos, J_right_pos])

        # Define desired stiffness in Cartesian space (can be tuned)
        kx = 100.0  # Stiffness in x direction
        ky = 100.0  # Stiffness in y direction
        kz = 300.0  # Stiffness in z direction
        k_null = 25.0  # Null space stiffness

        K_task = np.diag([kx, ky, kz, kx, ky, kz])
        C_task = np.linalg.pinv(K_task)  # Damping for critical damping

        # Compute the RVC stiffness matrix in joint space
        J_transpose = J_hands_pos.T
        J_inv_transpose = np.linalg.pinv(J_transpose)

        comp_matrix = np.linalg.pinv(J_hands_pos) @ C_task @ J_inv_transpose  # (nq, nq)
        # null-space stiffness
        c_null = 1 / k_null
        c_null_mat = np.eye(model.nq) * c_null
        comp_matrix += c_null_mat - np.linalg.pinv(J_hands_pos) @ J_hands_pos @ c_null_mat @ J_transpose @ J_inv_transpose

        stiffness_matrix = np.linalg.pinv(comp_matrix + np.eye(model.nq) * 1e-6)

        # condition number check
        mat_pd = np.eye(model.nq) * kp
        cond_number = np.linalg.cond(J_hands_pos)

        temp = np.maximum(cond_number - 30, 1e-6)

        ee_alpha = 0.7 # 0.3 0.7
        alpha_val = ee_alpha / (1.0 + temp)
        # print(f"cond_number: %.2f, alpha_val: %.4f" % (cond_number, alpha_val))

        # stiffness_matrix = self.log_euclidean_blend(stiffness_matrix, mat_pd, alpha=alpha_val)

        return stiffness_matrix

    def log_euclidean_blend(self, K_old, K_new, alpha, kmin: float = 1e-2, kmax: float = 1e6):
        """
        Log-Euclidean interpolation (numpy version):
            K_mix = exp( alpha * log(K_old) + (1-alpha) * log(K_new) )
        Args:
            K_old: [n, n] SPD
            K_new: [n, n] SPD
            alpha: float in [0,1], 1->old, 0->new
            kmin, kmax: spectral clamps applied to the final K_mix
        Returns:
            K_mix: [n, n] SPD
        """
        assert K_old.shape == K_new.shape
        n = K_old.shape[0]
        # 计算对数
        logK_old = logm(K_old)
        logK_new = logm(K_new)
        logK_mix = alpha * logK_old + (1.0 - alpha) * logK_new
        K_mix = expm(logK_mix)

        # final spectral clamp (保证正定性和条件数)
        evals, evecs = eigh((K_mix + K_mix.T) / 2)
        evals = np.clip(evals, kmin, kmax)
        K_mix = (evecs * evals) @ evecs.T

        return K_mix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = CompPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()

    torque_arr = np.array(policy.torque_log)  # shape: [steps, num_upper_dofs]
    np.save("upper_body_measured_torque_log.npy", torque_arr)
    base_vel_arr = np.array(policy.base_vel_log)  # shape: [steps, 2]
    np.save("upper_body_base_velocity_log.npy", base_vel_arr)
    target_vel_arr = np.array(policy.target_vel_log)  # shape: [steps, 2]
    np.save("upper_body_target_velocity_log.npy", target_vel_arr)
