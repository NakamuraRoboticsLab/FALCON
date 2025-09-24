import numpy as np
import matplotlib.pyplot as plt

# 读取保存的力矩数据
torque_arr = np.load("upper_body_measured_torque_log.npy")  # shape: [steps, num_upper_dofs]

# 读取base线速度数据
base_vel_arr = np.load("upper_body_base_velocity_log.npy")  # shape: [steps, 2] (x, y)
target_vel_arr = np.load("upper_body_target_velocity_log.npy")  # shape: [steps, 2] (x, y)

# 设置时间轴（假设采样周期为 0.02s，即50Hz）
dt = 0.02
timesteps = np.arange(torque_arr.shape[0]) * dt
time_arr = np.arange(base_vel_arr.shape[0]) * dt  # 用于线速度数据的时间轴

left_elbow_idx = 3
right_elbow_idx = 7

# 第一个图：上半身关节力矩
plt.figure(figsize=(10, 6))
plt.plot(timesteps, torque_arr[:, right_elbow_idx], label='Right Elbow (PD)', linewidth=2)
plt.plot(timesteps, torque_arr[:, left_elbow_idx], label='Left Elbow (CoTaP)', linewidth=2, color='red')
plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Measured Torque (Nm)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)  # 坐标轴数字字体大小
plt.xlim([0, 10])  # 只显示前10秒
plt.title('Upper Body Joint Measured Torques', fontsize=25)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("upper_body_measured_torque_plot.eps", format='eps', dpi=300)
plt.show()

# 第二个图：Base线速度跟踪对比
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# X方向线速度
axes[0].plot(time_arr, base_vel_arr[:, 0], 'b-', label='Actual X Velocity', linewidth=3)
axes[0].plot(time_arr, target_vel_arr[:, 0], 'r--', label='Target X Velocity', linewidth=3)
axes[0].set_xlabel('Time (s)', fontsize=25)
axes[0].set_ylabel('Linear Velocity X (m/s)', fontsize=25)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].set_xlim([0, 10])  # 只显示前10秒
axes[0].set_ylim([-5, 5])  # Y轴范围
axes[0].set_title('Base Linear Velocity X Tracking', fontsize=25)
axes[0].legend(fontsize=20)
axes[0].grid(True, alpha=0.3)

# Y方向线速度
axes[1].plot(time_arr, base_vel_arr[:, 1], 'b-', label='Actual Y Velocity', linewidth=3)
axes[1].plot(time_arr, target_vel_arr[:, 1], 'r--', label='Target Y Velocity', linewidth=3)
axes[1].set_xlabel('Time (s)', fontsize=25)
axes[1].set_ylabel('Linear Velocity Y (m/s)', fontsize=25)
axes[1].tick_params(axis='both', which='major', labelsize=20)
axes[1].set_xlim([0, 10])  # 只显示前10秒
axes[1].set_ylim([-5, 5])  # Y轴范围
axes[1].set_title('Base Linear Velocity Y Tracking', fontsize=25)
axes[1].legend(fontsize=20)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("base_velocity_tracking.eps", format='eps', dpi=300)
plt.show()

# 第三个图：速度跟踪误差
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 计算速度误差
vel_error_x = base_vel_arr[:, 0] - target_vel_arr[:, 0]
vel_error_y = base_vel_arr[:, 1] - target_vel_arr[:, 1]

# X方向误差
axes[0].plot(time_arr, vel_error_x, 'g-', linewidth=3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
axes[0].set_xlabel('Time (s)', fontsize=25)
axes[0].set_ylabel('Velocity Error X (m/s)', fontsize=25)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].set_xlim([0, 10])  # 只显示前10秒
axes[0].set_ylim([-5, 5])
axes[0].set_title('Linear Velocity Tracking Error (X)', fontsize=25)
axes[0].grid(True, alpha=0.3)

# Y方向误差
axes[1].plot(time_arr, vel_error_y, 'g-', linewidth=3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
axes[1].set_xlabel('Time (s)', fontsize=25)
axes[1].set_ylabel('Velocity Error Y (m/s)', fontsize=25)
axes[1].tick_params(axis='both', which='major', labelsize=20)
axes[1].set_xlim([0, 10])  # 只显示前10秒
axes[1].set_ylim([-5, 5])
axes[1].set_title('Linear Velocity Tracking Error (Y)', fontsize=25)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("velocity_tracking_error.eps", format='eps', dpi=300)
plt.show()

# 打印统计信息
print(f"Velocity tracking statistics:")
print(f"X direction - Mean error: {np.mean(vel_error_x):.4f} m/s, RMS error: {np.sqrt(np.mean(vel_error_x**2)):.4f} m/s")
print(f"Y direction - Mean error: {np.mean(vel_error_y):.4f} m/s, RMS error: {np.sqrt(np.mean(vel_error_y**2)):.4f} m/s")
print(f"Data shape - Torque: {torque_arr.shape}, Base velocity: {base_vel_arr.shape}, Target velocity: {target_vel_arr.shape}")