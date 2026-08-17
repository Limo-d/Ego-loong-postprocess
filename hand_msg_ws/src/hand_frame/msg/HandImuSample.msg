# 单个 IMU 一帧数据（对齐 glove_demo/msg/ImuData）
# int32   imu_id    # IMU 编号，左手 0-15，右手映射到 16-31（按 GlovePacket.imu 的索引）
# 注：imu_id 已移除，左右手通过 HandFrame.imu_left / imu_right 区分
float32 q_w
float32 q_x
float32 q_y
float32 q_z
float32 acc_x
float32 acc_y
float32 acc_z
float32 gyr_x
float32 gyr_y
float32 gyr_z