import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Model Parameters ---
L = 20000.0; nx = 101; dx = L / (nx - 1)
B = 20.0; n_manning = 0.025; S0 = 0.0001; g = 9.81
T_total = 3600 * 5

# Gate parameters
gate_pos_idx = int(10000 / dx); gate_opening = 0.5; Cd = 0.6

# Boundary Conditions
Q_initial = 30.0; Q_upstream_new = 35.0; H_downstream_initial = 5.0

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
    if np.any(y <= 0.01): y[y <= 0.01] = 0.01; H = y + z_bed_arr
    return H, np.full(nx_val, Q_val)

print("Calculating initial conditions...")
H, Q = calculate_initial_conditions(Q_initial, H_downstream_initial, z_bed, B, n_manning, S0, nx, dx)

# --- 4. Lax-Friedrichs Solver ---
def solve_lax_friedrichs(H, Q, dt, dx, B, n, S0, g, z_bed):
    A = B * (H - z_bed)
    U = np.array([A, Q])
    U_new = U.copy()
    epsilon = 1e-6

    # Calculate Flux F and Source S vectors
    F = np.zeros_like(U)
    F[0,:] = Q
    F[1,:] = Q**2 / (A + epsilon) + 0.5 * g * A**2 / B

    y = A / B
    R = A / (B + 2*y)
    Sf = n**2 * Q**2 / ((A+epsilon)**2 * (R+epsilon)**(4/3))
    S = np.array([np.zeros(nx), g*A*(S0 - Sf)])

    # Apply Lax-Friedrichs scheme to interior points
    for i in range(1, nx - 1):
        U_new[:, i] = 0.5 * (U[:, i+1] + U[:, i-1]) - (dt/(2*dx)) * (F[:, i+1] - F[:, i-1]) + dt * S[:, i]

    # --- Robustness Check ---
    A_new = U_new[0,:]
    A_new[A_new < 0.01] = 0.01

    return A_new/B + z_bed, U_new[1,:]

# --- 5. Main Loop with Stability Control ---
y = H - z_bed
v = Q / (B*y)
c = np.sqrt(g*y)
dt = 0.5 * dx / np.max(np.abs(v) + c) # CFL condition dt <= dx / (|v|+c)
print(f"Using Lax-Friedrichs scheme. Calculated stable dt={dt:.2f}s.")

H_results, Q_results = [], []
current_time = 0
t_step = 0
while current_time < T_total:
    # Set Boundary Conditions
    # Upstream (Flow)
    if current_time < 3600: Q[0] = Q_initial
    else: Q[0] = Q_initial + (Q_upstream_new - Q_initial) * min(1.0, (current_time-3600)/1800)
    H[0] = H[1] # Zero-gradient for H

    # Downstream (Water Level)
    H[-1] = H_downstream_initial
    Q[-1] = Q[-2] # Zero-gradient for Q

    # Solve for one time step
    H, Q = solve_lax_friedrichs(H, Q, dt, dx, B, n_manning, S0, g, z_bed)

    # Apply Gate Internal Boundary Condition
    y_up = H[gate_pos_idx] - z_bed[gate_pos_idx]
    y_down = H[gate_pos_idx+1] - z_bed[gate_pos_idx+1]
    is_submerged = y_down > 0.9 * gate_opening
    if not is_submerged: Q_gate = Cd*B*gate_opening*np.sqrt(2*g*max(0, y_up))
    else: Q_gate = Cd*B*gate_opening*np.sqrt(2*g*max(0, H[gate_pos_idx]-H[gate_pos_idx+1]))
    Q[gate_pos_idx] = Q[gate_pos_idx+1] = Q_gate

    # Store results
    if t_step % 50 == 0:
        H_results.append(H.copy()); Q_results.append(Q.copy())

    if t_step % 500 == 0: print(f"Progress: {100*current_time/T_total:.0f}% (t={current_time/3600:.2f}h)")

    current_time += dt
    t_step += 1

print("Simulation complete!")

# --- 6. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
max_h, min_h = np.max(H_results), z_bed.min()
max_q, min_q = np.max(Q_results), np.min(Q_results)

def update_plot(frame_index):
    if frame_index >= len(H_results): return
    time_in_hours = (frame_index * 50 * dt) / 3600
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

    ax1.set_title(f'1D Hydraulic Model (Lax-Friedrichs) - Time: {time_in_hours:.2f} hours', fontsize=16)
    ax1.set_xlabel('Distance (km)'); ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(min_h-1, max_h+1.5); ax1.set_xlim(0, L/1000)
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend(loc='upper right')

    ax2.plot(x/1000, Q_t, color='red', lw=2, label='Flow Rate')
    ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('Flow Rate (m³/s)')
    ax2.set_ylim(min_q-5, max_q+5); ax2.set_xlim(0, L/1000)
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend(loc='upper right')
    fig.tight_layout()

print("Generating final Lax-Friedrichs GIF animation...")
num_frames = len(H_results)
ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=100)
try:
    ani.save('hydraulic_simulation_final_lax.gif', writer='pillow', fps=15)
    print("Successfully saved final GIF to: hydraulic_simulation_final_lax.gif")
except Exception as e:
    print(f"Failed to save GIF: {e}")
