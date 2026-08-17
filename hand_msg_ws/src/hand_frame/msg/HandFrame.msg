# 手部数据合并帧：imu 姿态 + pressure + 关节角
# 对齐 glove_demo/msg/GlovePacket 字段布局（左/右分离）
builtin_interfaces/Time imu_stamp_left
builtin_interfaces/Time pressure_stamp_left
builtin_interfaces/Time imu_stamp_right
builtin_interfaces/Time pressure_stamp_right

# IMU 原始数据（左16 + 右16，和 GlovePacket 对齐）
hand_frame/HandImuSample[16] imu_left
hand_frame/HandImuSample[16] imu_right

# pressure 原始数据（左68 + 右68）
float32[68] pressure_left
float32[68] pressure_right

# SolveState 原始输出（每手27个double，缺数据填 kMissing=1e9）
# [0..15]  四指关节角（食/中/无名/小指，每指: MCP屈,MCP展,PIP屈,DIP屈）单位:度
# [16..18] 拇指关节角（MCP屈, MCP展, IP屈）单位:度
# [19..22] 拇指CMC四元数（掌骨相对手背，w,x,y,z）
# [23..26] 手背绝对四元数（整手世界朝向，w,x,y,z）
float64[27] solve_state_left
float64[27] solve_state_right
