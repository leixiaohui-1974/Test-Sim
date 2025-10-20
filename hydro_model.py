import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Model Parameters ---
L = 20000.0; nx = 101; dx = L / (nx - 1)
B = 20.0; n_manning = 0.025; S0 = 0.0001; g = 9.81
T_total = 3600 * 5; dt = 60
gate_pos_idx = int(10000 / dx); gate_opening = 0.5; Cd = 0.6
Q_initial = 30.0; Q_upstream_new = 35.0; H_downstream_initial = 5.0
theta = 0.6

# --- 2. Grid and Bed Elevation ---
x = np.linspace(0, L, nx)
z_bed = S0 * (L - x)

# --- 3. Initial Conditions ---
def calculate_initial_conditions(Q_val, H_down, z_bed_arr, B_val, n_val, S0_val, nx_val, dx_val):
    H = np.zeros(nx_val); H[-1] = H_down
    for i in range(nx_val - 2, -1, -1):
        y_i1 = H[i+1] - z_bed_arr[i+1]
        if y_i1 <= 0.01: y_i1 = 0.01
        A_i1 = B_val * y_i1; R_i1 = A_i1 / (B_val + 2*y_i1)
        Sf_i1 = (Q_val * n_val / (A_i1 * R_i1**(2.0/3.0)))**2
        H[i] = H[i+1] + (S0_val - Sf_i1) * dx_val
    y = H - z_bed_arr
    if np.any(y <= 0): y[y <= 0] = 0.01; H = y + z_bed_arr
    return H, np.full(nx_val, Q_val)

print("Calculating initial conditions...")
H, Q = calculate_initial_conditions(Q_initial, H_downstream_initial, z_bed, B, n_manning, S0, nx, dx)

# --- 4. Preissmann Solver (Corrected Gate Logic) ---
def solve_preissmann_final_final(H_n, Q_n, dt, dx, B, n, S0, g, z_bed, theta, Q_bc, H_bc, gate_idx, gate_open, Cd):
    nx = len(H_n)
    M = np.zeros((2*nx, 2*nx)); P = np.zeros(2*nx)
    epsilon = 1e-5

    for i in range(nx - 1):
        if i == gate_idx: continue
        y_i, y_i1 = H_n[i]-z_bed[i], H_n[i+1]-z_bed[i+1]
        A_i, A_i1 = B*y_i, B*y_i1
        R_i, R_i1 = (A_i/(B+2*y_i) if y_i>epsilon else 0), (A_i1/(B+2*y_i1) if y_i1>epsilon else 0)
        A_avg, R_avg = (A_i+A_i1)/2, (R_i+R_i1)/2
        Q_avg = (Q_n[i]+Q_n[i+1])/2

        j=2*i
        M[j, 2*i], M[j, 2*i+2] = -theta/dx, theta/dx
        M[j, 2*i+1], M[j, 2*i+3] = B/(2*dt), B/(2*dt)
        P[j] = -(Q_n[i+1] - Q_n[i])/dx

        j=2*i+1
        Sf = n**2 * Q_avg * abs(Q_avg) / (A_avg**2*R_avg**(4/3)) if A_avg>epsilon else 0
        K1 = 1/(2*dt)
        K2 = theta * (2*Q_avg/A_avg)/dx if A_avg>epsilon else 0
        K3 = theta * (g*A_avg - Q_avg**2*B/A_avg**2)/dx if A_avg>epsilon else 0
        P_mom = -( (Q_n[i+1]**2/(A_i1 if A_i1>epsilon else 1)) - (Q_n[i]**2/(A_i if A_i>epsilon else 1)) )/dx - g*A_avg*(H_n[i+1]-H_n[i])/dx + g*A_avg*(S0 - Sf)

        M[j, 2*i], M[j, 2*i+2] = K1 - K2, K1 + K2
        M[j, 2*i+1], M[j, 2*i+3] = -K3, K3
        P[j] = P_mom

    i = gate_idx
    y_up, y_down = H_n[i]-z_bed[i], H_n[i+1]-z_bed[i+1]

    j=2*i; M[j,:], P[j]=0, Q_n[i+1]-Q_n[i]; M[j,j], M[j,j+2]=1,-1

    j=2*i+1; M[j,:], P[j]=0,0
    is_submerged = y_down > 0.9*gate_open
    if not is_submerged:
        Q_gate = Cd*B*gate_open*np.sqrt(2*g*(y_up+epsilon))
        dQ_dH_up = Cd*B*gate_open*np.sqrt(g/(2*(y_up+epsilon))) if y_up>epsilon else 0
        # --- !! FINAL BUG FIX per Code Review !! ---
        # Equation: 1*dQ_{i+1} - (dQ/dH_i)*dH_i = RHS
        M[j, 2*(i+1)] = 1.0         # Coeff for dQ_{i+1}
        M[j, 2*i+1]   = -dQ_dH_up     # Coeff for dH_i
        P[j] = Q_gate - Q_n[i+1]
    else:
        delta_H = H_n[i] - H_n[i+1]
        Q_gate = Cd*B*gate_open*np.sqrt(2*g*(delta_H+epsilon))
        C_sqrt = Cd*B*gate_open*np.sqrt(g/2)/np.sqrt(delta_H+epsilon) if delta_H>epsilon else 0
        # --- !! FINAL BUG FIX per Code Review !! ---
        # Equation: 1*dQ_{i+1} - C_sqrt*dH_i + C_sqrt*dH_{i+1} = RHS
        M[j, 2*(i+1)]   = 1.0         # Coeff for dQ_{i+1}
        M[j, 2*i+1]     = -C_sqrt     # Coeff for dH_i
        M[j, 2*(i+1)+1] = C_sqrt      # Coeff for dH_{i+1}
        P[j] = Q_gate - Q_n[i+1]

    M[-2,:],P[-2]=0, Q_bc-Q_n[0]; M[-2,0]=1
    M[-1,:],P[-1]=0, H_bc-H_n[-1]; M[-1,-1]=1
    try:
        sol = np.linalg.solve(M, P)
        H_new, Q_new = H_n+sol[1::2], Q_n+sol[0::2]
        if np.any(np.isnan(H_new)): return H_n, Q_n
        return H_new, Q_new
    except np.linalg.LinAlgError: return H_n, Q_n

# --- 5. Main Loop ---
nt = int(T_total / dt)
H_results, Q_results = [], []
print(f"Using final, correct Preissmann scheme. dt={dt}s, nt={nt} steps.")
for t_step in range(nt):
    current_time = t_step * dt
    if current_time < 3600: Q_bc = Q_initial
    else: Q_bc = Q_initial + (Q_upstream_new - Q_initial) * min(1.0, (current_time-3600)/1800)

    if t_step % 10 == 0: H_results.append(H.copy()); Q_results.append(Q.copy())
    H, Q = solve_preissmann_final_final(H, Q, dt, dx, B, n_manning, S0, g, z_bed, theta, Q_bc, H_downstream_initial, gate_pos_idx, gate_opening, Cd)
    if np.any(H-z_bed < 0): H[H-z_bed<0] = z_bed[H-z_bed<0] + 0.01

    if t_step % (nt // 10) == 0: print(f"Progress: {100 * t_step / nt:.0f}%")
print("Simulation complete!")

# --- 6. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
max_h, min_h = np.max(H_results), z_bed.min()
max_q, min_q = np.max(Q_results), np.min(Q_results)

def update_plot(frame_index):
    if frame_index >= len(H_results): return
    time_in_hours = (frame_index * 10 * dt) / 3600
    H_t, Q_t = H_results[frame_index], Q_results[frame_index]

    ax1.clear(); ax2.clear()
    ax1.plot(x/1000, z_bed, color='brown', lw=2, label='Channel Bed')
    ax1.fill_between(x/1000, 0, z_bed, color='peru', alpha=0.4)

    gate_x, gate_w = x[gate_pos_idx]/1000, (dx/1000)*2
    struct_top = max(H_t[gate_pos_idx], z_bed[gate_pos_idx]+4.0)+1.5
    ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx], struct_top, color='gray', zorder=10)
    ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx]+gate_opening, struct_top, color='black', zorder=11)
    ax1.text(gate_x, struct_top-0.75, f'Gate\nOpening: {gate_opening}m', ha='center', fontsize=10)
    ax1.plot(x/1000, H_t, color='blue', lw=2.5, label='Water Level', zorder=20)

    ax1.set_title(f'1D Hydraulic Model (Preissmann) - Time: {time_in_hours:.2f} hours', fontsize=16)
    ax1.set_xlabel('Distance (km)'); ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(min_h-1, max_h+1.5); ax1.set_xlim(0, L/1000)
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend(loc='upper right')

    ax2.plot(x/1000, Q_t, color='red', lw=2, label='Flow Rate')
    ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('Flow Rate (m³/s)')
    ax2.set_ylim(min_q-5, max_q+5); ax2.set_xlim(0, L/1000)
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend(loc='upper right')
    fig.tight_layout()

print("Generating final, truly corrected Preissmann GIF animation...")
num_frames = len(H_results)
ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=100)
try:
    ani.save('hydraulic_simulation_preissmann_final.gif', writer='pillow', fps=15)
    print("Successfully saved final GIF to: hydraulic_simulation_preissmann_final.gif")
except Exception as e:
    print(f"Failed to save GIF: {e}")
