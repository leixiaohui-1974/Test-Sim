import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

# 忽略可能出现的除零或无效值警告，这在数值计算中可能在特定步骤发生，但会被处理
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 模型参数定义 ---
# 渠道几何参数
L = 20000.0  # 渠道长度 (m)
nx = 101      # 空间网格点数
dx = L / (nx - 1) # 空间步长 (m)
B = 20.0      # 渠道宽度 (m)
n_manning = 0.025 # 曼宁糙率系数
S0 = 0.0001   # 渠底坡度

# 模拟控制参数
T = 3600 * 5  # 总模拟时长 (s), 5 hours
dt = 60       # 时间步长 (s)
nt = int(T / dt) # 时间步数
theta = 0.6   # 普林斯曼格式的加权系数 (0.5-1.0)

# 闸门参数
gate_location_km = 10.0 # 闸门位置 (km)
gate_pos_idx = int(gate_location_km * 1000 / dx) # 闸门在网格上的索引
gate_opening = 0.5 # 闸门开度 (m)
Cd = 0.6           # 流量系数

# 物理常数
g = 9.81

# 边界条件
Q_initial = 30.0  # 初始上游流量 (m^3/s)
Q_upstream_new = 35.0 # 阶跃后的上游流量 (m^3/s)
H_downstream = 5.0 # 下游固定水位 (m)，假设这是相对于某个基准面的高程

# --- 2. 网格和渠底高程设置 ---
x = np.linspace(0, L, nx)
# 假设下游出口处渠底高程为0
z_bed = S0 * (L - x)

# --- 3. 初始条件计算 ---
def calculate_initial_conditions(Q, H_down, z_bed, B, n, S0, nx, dx):
    """使用曼宁公式估算均匀流作为初始条件"""
    # 计算正常水深 y_n
    # Q = (1/n) * A * R^(2/3) * S0^(1/2)
    # A = B*y, P = B+2y, R = A/P
    # 为简化，用宽浅渠道假设 R ≈ y
    y_n = (Q * n / (B * np.sqrt(S0)))**(3.0/5.0)
    print(f"初始流量 {Q} m³/s 对应的正常水深 (估算): {y_n:.2f} m")

    # 实际中，背水曲线会更复杂，这里用一个简化剖面
    # 从下游水位开始，向上游逐段计算
    H = np.zeros(nx)
    H[-1] = H_down
    y = np.zeros(nx)
    y[-1] = H[-1] - z_bed[-1]

    for i in range(nx - 2, -1, -1):
        A_i1 = B * y[i+1]
        P_i1 = B + 2*y[i+1]
        R_i1 = A_i1 / P_i1
        Sf_i1 = (Q * n / (A_i1 * R_i1**(2.0/3.0)))**2
        # 简单的一步法（欧拉法）来估算上游水位
        H[i] = H[i+1] + (S0 - Sf_i1) * dx
        y[i] = H[i] - z_bed[i]
        if y[i] < 0: y[i] = y_n # 避免负水深

    Q_vec = np.full(nx, Q)
    return H, Q_vec

print("正在计算初始条件...")
H, Q = calculate_initial_conditions(Q_initial, H_downstream, z_bed, B, n_manning, S0, nx, dx)

# --- 4. 普林斯曼格式求解器 ---
def solve_preissmann(H, Q, dt, dx, B, n, S0, z_bed, theta, Q_upstream_bc, H_downstream_bc, gate_pos_idx, gate_opening, Cd):
    """
    求解一个时间步长的圣维南方程组
    返回更新后的 H_new, Q_new
    """
    nx = len(H)
    # 系数矩阵 A 和右端项 b
    # 未知量是 [dQ1, dH1, dQ2, dH2, ...]
    num_vars = 2 * nx
    A_matrix = np.zeros((num_vars, num_vars))
    b_vector = np.zeros(num_vars)

    # 组装内部点的方程
    for i in range(nx - 1):
        # --- 物理参数取平均值 ---
        A_avg = B * (H[i] - z_bed[i] + H[i+1] - z_bed[i+1]) / 2
        P_avg = B + (H[i] - z_bed[i] + H[i+1] - z_bed[i+1])
        R_avg = A_avg / P_avg
        Q_avg = (Q[i] + Q[i+1]) / 2

        if A_avg <= 1e-6: A_avg = 1e-6 # 避免除零

        # --- 圣维南方程离散化后的系数 ---
        # 连续性方程
        C1 = dt * theta / dx
        C2 = dt * (1 - theta) / dx
        b_cont = - (Q[i+1] - Q[i]) / dx - (B * ( (H[i+1]+H[i])/2 - (z_bed[i+1]+z_bed[i])/2 ) ) / dt * 0 # 这项理论上有，但简化处理

        # 填充矩阵
        # dQ_i, dH_i, dQ_{i+1}, dH_{i+1}
        A_matrix[2*i, 2*i] = -C1             # dQ_i
        A_matrix[2*i, 2*i+1] = B * theta / dx # dH_i
        A_matrix[2*i, 2*(i+1)] = C1           # dQ_{i+1}
        A_matrix[2*i, 2*(i+1)+1] = B * theta / dx # dH_{i+1}
        b_vector[2*i] = -(Q[i+1] - Q[i]) / dx # 简化版右端项

        # 动量方程
        Sf = (n**2 * abs(Q_avg) * Q_avg) / (A_avg**2 * R_avg**(4/3))

        L1 = (1 - 2 * theta * dt * Q_avg * B / (A_avg**2))
        L2 = (g * A_avg / dx - 2 * theta * dt * Q_avg**2 * B / (A_avg**3)) * theta * dt
        L3 = - (g * A_avg / dx + 2 * theta * dt * Q_avg**2 * B / (A_avg**3)) * theta * dt
        L4 = - ( (Q_avg**2 * B) / (g * A_avg**3) - 1 ) * B * dt * theta / dx

        # 填充矩阵
        # dQ_i, dH_i, dQ_{i+1}, dH_{i+1}
        A_matrix[2*i+1, 2*i] = L1 / dx # dQ_i
        A_matrix[2*i+1, 2*i+1] = L2
        A_matrix[2*i+1, 2*(i+1)] = L1 / dx # dQ_{i+1}
        A_matrix[2*i+1, 2*(i+1)+1] = L3
        b_vector[2*i+1] = - ( (Q[i+1]**2 / A_avg) - (Q[i]**2 / A_avg) ) / dx - g * (H[i+1]-H[i])/dx - g*Sf

    # --- 处理闸门内部边界 ---
    # 在闸门位置，用闸门公式替换动量方程
    i = gate_pos_idx
    H_up = H[i]
    H_down = H[i+1]
    y_up = H_up - z_bed[i]

    # 闸孔出流公式 Q = Cd * B * a * sqrt(2*g*y_up)
    # 线性化: dQ = (dQ/dH_up)*dH_up
    dQ_dH_up = Cd * B * gate_opening * np.sqrt(g / (2 * y_up)) if y_up > 0 else 0

    # 替换掉 momentum-like equation at gate_pos_idx
    # Eq1: Q_i = Q_{i+1} => dQ_i - dQ_{i+1} = 0
    A_matrix[2*i, :] = 0
    A_matrix[2*i, 2*i] = 1
    A_matrix[2*i, 2*(i+1)] = -1
    b_vector[2*i] = Q[i+1] - Q[i] # RHS should be 0 for steady state

    # Eq2: Q_{i+1} - f(H_i) = 0 => dQ_{i+1} - (dQ/dH_i)*dH_i = ...
    Q_gate_calc = Cd * B * gate_opening * np.sqrt(2 * g * y_up) if y_up > 0 else 0
    A_matrix[2*i+1, :] = 0
    A_matrix[2*i+1, 2*i+1] = -dQ_dH_up
    A_matrix[2*i+1, 2*(i+1)] = 1
    b_vector[2*i+1] = Q_gate_calc - Q[i+1]

    # --- 边界条件 ---
    # 上游: 流量边界 dQ_0 = Q_new - Q_old
    A_matrix[num_vars-2, :] = 0
    A_matrix[num_vars-2, 0] = 1  # dQ at index 0
    b_vector[num_vars-2] = Q_upstream_bc - Q[0]

    # 下游: 水位边界 dH_{nx-1} = H_new - H_old
    A_matrix[num_vars-1, :] = 0
    A_matrix[num_vars-1, num_vars-1] = 1 # dH at index nx-1
    b_vector[num_vars-1] = H_downstream_bc - H[nx-1]

    # 求解线性方程组
    try:
        sol = np.linalg.solve(A_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("矩阵奇异，无法求解。可能由于不稳定的初始条件或参数。")
        return H, Q # 返回旧值

    # 更新 H 和 Q
    dQ = sol[0::2]
    dH = sol[1::2]
    H_new = H + dH
    Q_new = Q + dQ

    return H_new, Q_new

# --- 5. 主模拟循环 ---
H_results = [H.copy()]
Q_results = [Q.copy()]

print("开始进行非恒定流模拟...")
for t_step in range(nt):
    if (t_step * dt) < 3600: # 1小时后改变流量
      Q_bc = Q_initial
    else:
      Q_bc = Q_upstream_new

    H_new, Q_new = solve_preissmann(
        H, Q, dt, dx, B, n_manning, S0, z_bed, theta,
        Q_bc, H_downstream,
        gate_pos_idx, gate_opening, Cd
    )
    H, Q = H_new, Q_new

    # 每隔一段时间保存一次结果
    if t_step % 10 == 0:
        H_results.append(H.copy())
        Q_results.append(Q.copy())

    if t_step % (nt // 10) == 0:
      print(f"模拟进度: {100 * t_step / nt:.0f}%")

print("模拟完成！")

# --- 6. 可视化 ---
fig, ax = plt.subplots(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 找到绘图范围
max_h = max(res.max() for res in H_results)
min_h = z_bed.min()

def draw_gate(ax, gate_idx, z_bottom, h_water, gate_open, total_height):
    """在图上绘制闸门"""
    gate_x = x[gate_idx]
    gate_width_plot = dx * 2 # 闸门在图上的宽度

    # 闸门墩
    ax.fill_between([gate_x - gate_width_plot/2, gate_x + gate_width_plot/2],
                    [z_bottom, z_bottom],
                    [total_height, total_height],
                    color='gray', alpha=0.8, zorder=10)
    # 闸门本身
    gate_bottom_el = z_bottom
    gate_top_el = gate_bottom_el + gate_open
    ax.fill_between([gate_x - gate_width_plot/2, gate_x + gate_width_plot/2],
                    [gate_top_el, gate_top_el],
                    [total_height, total_height],
                    color='black', alpha=0.9, zorder=11, label=f'闸门 (开度: {gate_opening}m)')

def update_plot(frame_index):
    ax.clear()

    # 绘制渠底
    ax.plot(x / 1000, z_bed, color='brown', linewidth=2, label='渠底高程')
    ax.fill_between(x / 1000, z_bed, 0, color='peru', alpha=0.5)

    # 绘制水位线
    H_t = H_results[frame_index]
    Q_t = Q_results[frame_index]
    ax.plot(x / 1000, H_t, color='blue', linewidth=2.5, label='水位线')

    # 绘制闸门
    draw_gate(ax, gate_pos_idx, z_bed[gate_pos_idx], H_t[gate_pos_idx], gate_opening, max_h * 1.1)

    # 设置图表属性
    time_in_hours = frame_index * 10 * dt / 3600
    ax.set_title(f'一维水力学模型 (Preissmann法)\n时间: {time_in_hours:.2f} 小时', fontsize=16)
    ax.set_xlabel('渠道里程 (km)', fontsize=12)
    ax.set_ylabel('高程 (m)', fontsize=12)
    ax.set_ylim(min_h - 1, max_h * 1.1)
    ax.set_xlim(0, L / 1000)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 添加流量信息文本
    q_up = Q_t[0]
    q_down = Q_t[-1]
    ax.text(0.05, 0.9, f'上游流量: {q_up:.2f} m³/s', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.05, 0.8, f'下游流量: {q_down:.2f} m³/s', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(loc='upper right')

print("正在生成GIF动画，这可能需要几分钟...")
# 创建动画
ani = FuncAnimation(fig, update_plot, frames=len(H_results), interval=100, blit=False)

# 保存为GIF
# 需要安装Pillow: pip install Pillow
try:
    ani.save('hydraulic_simulation.gif', writer='pillow', fps=10)
    print("成功保存GIF动画到: hydraulic_simulation.gif")
except Exception as e:
    print(f"保存GIF失败: {e}")
    print("请确保已安装Pillow库 (pip install Pillow)")

# The following line is commented out because it requires a display, which is not available in the environment.
# plt.show()
