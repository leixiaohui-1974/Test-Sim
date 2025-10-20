import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Model Parameters ---
L = 20000.0
nx = 101
dx = L / (nx - 1)
B = 20.0
n_manning = 0.025
S0 = 0.0001

T = 3600 * 5
dt = 30 # A stable time step
nt = int(T / dt)
theta = 0.6 # 0.6 is a good balance between accuracy and stability

gate_location_km = 10.0
gate_pos_idx = int(gate_location_km * 1000 / dx)
gate_opening = 0.5
Cd = 0.6
g = 9.81

Q_initial = 30.0
Q_upstream_new = 35.0
H_downstream = 5.0

# --- 2. Grid and Bed Elevation ---
x = np.linspace(0, L, nx)
z_bed = S0 * (L - x)

# --- 3. Initial Conditions ---
def calculate_initial_conditions(Q, H_down, z_bed, B, n, S0, nx, dx):
    y_n = (Q * n / (B * np.sqrt(S0)))**(3.0/5.0)
    print(f"Initial flow {Q} m³/s corresponds to normal depth (estimated): {y_n:.2f} m")
    H = np.zeros(nx)
    H[-1] = H_down
    y = np.zeros(nx)
    y[-1] = H[-1] - z_bed[-1]
    for i in range(nx - 2, -1, -1):
        if y[i+1] <= 0.01: y[i+1] = 0.01
        A_i1 = B * y[i+1]
        R_i1 = A_i1 / (B + 2*y[i+1])
        Sf_i1 = (Q * n / (A_i1 * R_i1**(2.0/3.0)))**2
        H[i] = H[i+1] + (S0 - Sf_i1) * dx
        y[i] = H[i] - z_bed[i]
        if y[i] <= 0.01: y[i] = y_n
    return H, np.full(nx, Q)

print("Calculating initial conditions...")
H, Q = calculate_initial_conditions(Q_initial, H_downstream, z_bed, B, n_manning, S0, nx, dx)

# --- 4. Textbook Preissmann Solver ---
def solve_preissmann_textbook(H_n, Q_n, dt, dx, B, n, S0, z_bed, theta, Q_bc, H_bc, gate_idx, gate_open, Cd):
    nx = len(H_n)
    M = np.zeros((2*nx, 2*nx))
    P = np.zeros(2*nx)
    epsilon = 1e-4

    # Interior nodes
    for i in range(nx - 1):
        if i == gate_idx: continue

        y_avg = ((H_n[i]-z_bed[i]) + (H_n[i+1]-z_bed[i+1])) / 2
        if y_avg < epsilon: y_avg = epsilon

        A_avg = B * y_avg
        R_avg = A_avg / (B + 2*y_avg)
        Q_avg = (Q_n[i] + Q_n[i+1]) / 2
        Sf = (n**2 * Q_avg * abs(Q_avg)) / (A_avg**2 * R_avg**(4/3)) if A_avg > epsilon else 0

        # --- Coefficients from linearizing the PDE first, then discretizing ---
        # Continuity: B*dH/dt + dQ/dx = 0
        j = 2*i
        M[j, 2*i]      = -theta / dx      # Coeff for dQ_i
        M[j, 2*i+1]    = B / (2 * dt)     # Coeff for dH_i
        M[j, 2*(i+1)]  = theta / dx       # Coeff for dQ_{i+1}
        M[j, 2*(i+1)+1]= B / (2 * dt)     # Coeff for dH_{i+1}
        P[j] = -(Q_n[i+1] - Q_n[i]) / dx

        # Momentum: dQ/dt + d(Q²/A)/dx + gA*dH/dx = gA(S0 - Sf)
        # Linearized form: dQ/dt + Cq*dQ/dx + Ch*dH/dx = RHS
        j = 2*i + 1
        Cq = 2 * Q_avg / A_avg
        Ch = g*A_avg - (Q_avg**2 * B / A_avg**2)
        RHS_S = g*A_avg*(S0 - Sf)

        M[j, 2*i]       = 1/(2*dt) - Cq * theta/dx
        M[j, 2*i+1]     = -Ch * theta/dx
        M[j, 2*(i+1)]   = 1/(2*dt) + Cq * theta/dx
        M[j, 2*(i+1)+1] = Ch * theta/dx

        P[j] = RHS_S - Cq * (Q_n[i+1] - Q_n[i])/dx - Ch * (H_n[i+1] - H_n[i])/dx

    # Gate boundary
    i = gate_idx
    y_up, y_down = H_n[i]-z_bed[i], H_n[i+1]-z_bed[i+1]

    j=2*i; M[j,:], P[j]=0, Q_n[i+1]-Q_n[i]; M[j,j], M[j,j+2]=1,-1

    j=2*i+1; M[j,:], P[j]=0,0
    is_submerged = y_down > 0.9 * gate_open
    if not is_submerged:
        Q_gate = Cd * B * gate_open * np.sqrt(2 * g * (y_up + epsilon))
        dQ_dH_up = Cd*B*gate_open*np.sqrt(g/(2*(y_up+epsilon)))
        M[j, j+2], M[j, j+1] = 1, -dQ_dH_up; P[j] = Q_gate - Q_n[i+1]
    else:
        delta_H = H_n[i] - H_n[i+1]
        Q_gate = Cd*B*gate_open*np.sqrt(2*g*(delta_H+epsilon))
        C_sqrt = Cd*B*gate_open*np.sqrt(g/2)/np.sqrt(delta_H+epsilon) if delta_H>epsilon else 0
        M[j,j+2], M[j,j+1], M[j,j+3] = 1,-C_sqrt,C_sqrt; P[j]=Q_gate - Q_n[i+1]

    # Boundaries
    M[-2,:],P[-2] = 0,Q_bc-Q_n[0]; M[-2,0]=1
    M[-1,:],P[-1] = 0,H_bc-H_n[-1]; M[-1,-1]=1

    try:
        sol = np.linalg.solve(M, P)
        H_new, Q_new = H_n + sol[1::2], Q_n + sol[0::2]
        if np.any(np.isnan(H_new)): return H_n, Q_n
        return H_new, Q_new
    except np.linalg.LinAlgError: return H_n, Q_n

# --- 5. Main Loop & Sanity Check ---
# Courant Number Check
v_init = Q_initial / (B * (H[0]-z_bed[0]))
c_init = np.sqrt(g * (H[0]-z_bed[0]))
courant = (v_init + c_init) * dt / dx
print(f"Initial Courant Number: {courant:.2f}. (For implicit schemes, >1 is acceptable but is a useful indicator).")

H_results, Q_results = [H.copy()], [Q.copy()]
print("Starting final, textbook-correct simulation...")
for t_step in range(nt):
    current_time = t_step * dt
    Q_bc = Q_initial + (Q_upstream_new-Q_initial) * min(1.0, (current_time-3600)/1800) if current_time>3600 else Q_initial
    H, Q = solve_preissmann_textbook(H, Q, dt, dx, B, n_manning, S0, z_bed, theta, Q_bc, H_downstream, gate_pos_idx, gate_opening, Cd)
    if t_step % 20 == 0: H_results.append(H.copy()); Q_results.append(Q.copy())
    if t_step % (nt // 10) == 0: print(f"Progress: {100 * t_step / nt:.0f}%")
print("Simulation complete!")

# --- 6. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
def update_plot(frame_index):
    time_in_hours = frame_index*20*dt/3600
    H_t, Q_t = H_results[frame_index], Q_results[frame_index]
    max_h, min_h = np.max(H_results), z_bed.min()
    max_q, min_q = np.max(Q_results), np.min(Q_results)

    ax1.clear(); ax2.clear()
    ax1.plot(x/1000, z_bed, color='brown', lw=2, label='Channel Bed')
    ax1.fill_between(x/1000, 0, z_bed, color='peru', alpha=0.4)
    ax1.plot(x/1000, H_t, color='blue', lw=2.5, label='Water Level')
    gate_x = x[gate_pos_idx]/1000; gate_w = (dx/1000)*2
    struct_top = max(H_t[gate_pos_idx], z_bed[gate_pos_idx]+4.0)+1.0
    ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx], struct_top, color='gray', zorder=10)
    ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx]+gate_opening, struct_top, color='black', zorder=11)
    ax1.text(gate_x, struct_top-0.5, f'Gate\nOpening: {gate_opening}m', ha='center', fontsize=10)
    ax1.set_title(f'1D Hydraulic Model - Time: {time_in_hours:.2f} hours', fontsize=16)
    ax1.set_xlabel('Distance (km)'); ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(min_h-1, max_h+1); ax1.set_xlim(0, L/1000)
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend(loc='upper right')

    ax2.plot(x/1000, Q_t, color='red', lw=2, label='Flow Rate')
    ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('Flow Rate (m³/s)')
    ax2.set_ylim(min_q-5, max_q+5); ax2.set_xlim(0, L/1000)
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend(loc='upper right')
    fig.tight_layout()

print("Generating final, truly corrected GIF animation...")
ani = FuncAnimation(fig, update_plot, frames=len(H_results), interval=100, blit=False)
try:
    ani.save('hydraulic_simulation_final_v3.gif', writer='pillow', fps=15)
    print("Successfully saved final GIF to: hydraulic_simulation_final_v3.gif")
except Exception as e:
    print(f"Failed to save GIF: {e}")
