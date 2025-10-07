import os
import sys

import casadi
import numpy as np
import pinocchio as pin

from .weighted_moving_filter import WeightedMovingFilter

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

class H1_ArmIK:
    def __init__(self, robot_config, unit_test=False, visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.unit_test = unit_test
        self.visualization = visualization

        # === Model Load ===
        urdf_path = robot_config["ASSET_FILE"]
        mesh_dir = robot_config["ASSET_ROOT"]
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self.model = self.robot.model
        self.data = self.model.createData()

        # === Joints to Lock (只锁定下肢和躯干，不包含手指和手部) ===
        self.mixed_jointsToLockIDs = [
            "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
            "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "torso_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_ankle_joint", "right_ankle_joint"
        ]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.zeros(self.model.nq),
        )
        self.reduced_model = self.reduced_robot.model

        # 检查 joint 是否存在
        left_joint_name = "left_elbow_joint"
        right_joint_name = "right_elbow_joint"
        left_ee_name = "left_elbow_ee"
        right_ee_name = "right_elbow_ee"
        offset = np.array([0.25, 0, 0])

        for joint_name, ee_name in [(left_joint_name, left_ee_name), (right_joint_name, right_ee_name)]:
            joint_id = self.reduced_model.getJointId(joint_name)
            print(f"{joint_name} id:", joint_id)
            if joint_id <= 0:
                raise ValueError(f"Joint {joint_name} not found in reduced_model! Available joints: {[j.name for j in self.reduced_model.joints]}")
            if not self.reduced_model.existFrame(ee_name):
                frame_SE3 = pin.SE3(np.eye(3), offset)
                self.reduced_model.addFrame(
                    pin.Frame(
                        ee_name,
                        joint_id,
                        frame_SE3,
                        pin.FrameType.OP_FRAME
                    )
                )
            else:
                print(f"Frame {ee_name} already exists!")

        for i, f in enumerate(self.reduced_model.frames):
            if f.name == "left_elbow_ee":
                print(f"Frame {f.name}: parent={f.parent}, type={f.type}, placement={f.placement}")

        self.reduced_data = self.reduced_model.createData()

        # === Casadi Symbolic Model (仅用于后续扩展) ===
        self.cmodel = pin.casadi.Model(self.reduced_model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_model.nq, 1)

        # === Filters ===
        self.init_data = np.zeros(self.reduced_model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_model.nq)

        # Attributes for interpolation
        self.current_L_tf = None
        self.current_R_tf = None
        self.current_L_orientation = None
        self.current_R_orientation = None
        self.speed_factor = 0.02  # You can adjust this as needed
        

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_arm_motor_q=None,
        current_arm_motor_dq=None,
        EE_efrc_L=None,  # noqa: N803
        EE_efrc_R=None,  # noqa: N803
        collision_check=None,
        max_iter=50,
        use_rotation=False
    ):
        """
        IK模式: left_wrist/right_wrist 为目标SE3（或4x4齐次变换/SE3对象），自动提取平移部分作为末端目标。
        兼容旧接口: 若未给target则直接用关节角过滤。
        """
        # ---------- 1. 若未进入 IK 模式，走原逻辑 ----------
        ik_mode = (left_wrist is not None) and (right_wrist is not None)
        if not ik_mode:
            if current_arm_motor_q is None:
                raise ValueError("Either provide left_wrist/right_wrist or current_arm_motor_q.")
            self.smooth_filter.add_data(current_arm_motor_q)
            filtered_q = self.smooth_filter.filtered_data
            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (filtered_q - self.init_data) * 0.0
            self.init_data = filtered_q
            tau_ff = pin.rnea(self.reduced_model, self.reduced_data, filtered_q, v, np.zeros(self.reduced_model.nv))
            return filtered_q, tau_ff

        # ---------- 2. IK 模式准备 ----------
        if not hasattr(self, "_ik_built"):
            self._ik_built = False
        if not self._ik_built:
            # Frame IDs
            self._left_frame_id = self.reduced_model.getFrameId("left_elbow_ee")
            self._right_frame_id = self.reduced_model.getFrameId("right_elbow_ee")

            # Casadi Opti
            self._opti = casadi.Opti()
            nq = self.reduced_model.nq
            self._var_q = self._opti.variable(nq, 1)
            self._par_q_last = self._opti.parameter(nq, 1)
            self._par_pL = self._opti.parameter(3, 1)
            self._par_pR = self._opti.parameter(3, 1)

            # 关节限
            q_lower = self.reduced_model.lowerPositionLimit
            q_upper = self.reduced_model.upperPositionLimit
            self._opti.subject_to(self._var_q >= q_lower)
            self._opti.subject_to(self._var_q <= q_upper)

            # 前向运动学 Casadi 构造
            cmodel_fk = pin.casadi.Model(self.reduced_model)
            cdata_fk = cmodel_fk.createData()
            q_sym = casadi.SX.sym("q_fk", nq, 1)
            pin.casadi.framesForwardKinematics(cmodel_fk, cdata_fk, q_sym)
            pL_expr = cdata_fk.oMf[self._left_frame_id].translation
            pR_expr = cdata_fk.oMf[self._right_frame_id].translation
            self._fk_fun = casadi.Function("fk_lr", [q_sym], [pL_expr, pR_expr])

            # 目标误差
            pL_cur, pR_cur = self._fk_fun(self._var_q)
            pos_err_L = pL_cur - self._par_pL
            pos_err_R = pR_cur - self._par_pR

            w_pos = 50.0
            w_reg = 0.02
            w_smooth = 0.1

            cost = w_pos * casadi.sumsqr(pos_err_L) + w_pos * casadi.sumsqr(pos_err_R)
            cost += w_reg * casadi.sumsqr(self._var_q)
            cost += w_smooth * casadi.sumsqr(self._var_q - self._par_q_last)

            self._opti.minimize(cost)
            self._opti.solver("ipopt", {"print_time": 0},
                              {"print_level": 0, "max_iter": max_iter, "tol": 1e-6})

            self._ik_built = True

        # ---------- 3. 设置初值与参数 ----------
        nq = self.reduced_model.nq
        if current_arm_motor_q is not None and current_arm_motor_q.shape[0] == nq:
            q_init = current_arm_motor_q
        else:
            q_init = self.init_data if self.init_data.shape[0] == nq else np.zeros(nq)

        # 提取目标平移部分
        def get_translation(T):
            if hasattr(T, "translation"):
                return np.array(T.translation).reshape(3)
            elif hasattr(T, "homogeneous"):
                return np.array(T.homogeneous[:3, 3]).reshape(3)
            elif isinstance(T, np.ndarray) and T.shape == (4, 4):
                return T[:3, 3].reshape(3)
            else:
                raise ValueError("left_wrist/right_wrist must be SE3, or 4x4 ndarray, or have .translation")

        left_target = get_translation(left_wrist)
        right_target = get_translation(right_wrist)

        self._opti.set_initial(self._var_q, q_init.reshape(nq, 1))
        self._opti.set_value(self._par_q_last, q_init.reshape(nq, 1))
        self._opti.set_value(self._par_pL, left_target.reshape(3, 1))
        self._opti.set_value(self._par_pR, right_target.reshape(3, 1))

        # ---------- 4. 求解 ----------
        try:
            sol = self._opti.solve()
            q_sol = np.asarray(sol.value(self._var_q)).flatten()
        except RuntimeError:
            # 失败回退
            q_sol = q_init

        print("IK solve: q_init =", np.round(q_init, 3), "-> q_sol =", np.round(q_sol, 3))

        # ---------- 5. 平滑 ----------
        self.smooth_filter.add_data(q_sol)
        filtered_q = self.smooth_filter.filtered_data

        # 速度估计（此处保守置 0）
        if current_arm_motor_dq is not None and current_arm_motor_dq.shape[0] == self.reduced_model.nv:
            v = current_arm_motor_dq * 0.0
        else:
            v = (filtered_q - q_init) * 0.0

        self.init_data = filtered_q

        # ---------- 6. 前馈力矩 ----------
        tau_ff = pin.rnea(self.reduced_model, self.reduced_data, filtered_q, v, np.zeros(self.reduced_model.nv))

        return filtered_q, tau_ff

    
    def set_initial_poses(self, L_tf, R_tf, L_orientation, R_orientation):  # noqa: N803
        """Set the initial poses for interpolation."""
        self.current_L_tf = L_tf.copy()
        self.current_R_tf = R_tf.copy()
        self.current_L_orientation = pin.Quaternion(L_orientation)
        self.current_R_orientation = pin.Quaternion(R_orientation)

    def get_q_tau(self, L_tf_target, R_tf_target, EE_efrc_L, EE_efrc_R):  # noqa: N803
        """Interpolate and solve IK for the given target poses."""
        # Interpolate positions and orientations
        self.current_L_tf = (1 - self.speed_factor) * self.current_L_tf + self.speed_factor * L_tf_target.translation
        self.current_R_tf = (1 - self.speed_factor) * self.current_R_tf + self.speed_factor * R_tf_target.translation

        self.current_L_orientation = self.current_L_orientation.slerp(
            self.speed_factor, pin.Quaternion(L_tf_target.rotation)
        )
        self.current_R_orientation = self.current_R_orientation.slerp(
            self.speed_factor, pin.Quaternion(R_tf_target.rotation)
        )

        L_tf_interpolated = pin.SE3(self.current_L_orientation.toRotationMatrix(), self.current_L_tf)
        R_tf_interpolated = pin.SE3(self.current_R_orientation.toRotationMatrix(), self.current_R_tf)

        # Solve IK
        sol_q, sol_tauff = self.solve_ik(
            L_tf_interpolated.homogeneous,
            R_tf_interpolated.homogeneous,
            EE_efrc_L=EE_efrc_L,
            EE_efrc_R=EE_efrc_R,
            collision_check=False,
        )
        return sol_q, sol_tauff