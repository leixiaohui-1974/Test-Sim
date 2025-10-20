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
dt = 10
nt = int(T / dt)
theta = 0.6

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
        A_i1 = B * y[i+1]
        R_i1 = A_i1 / (B + 2*y[i+1])
        Sf_i1 = (Q * n / (A_i1 * R_i1**(2.0/3.0)))**2
        H[i] = H[i+1] + (S0 - Sf_i1) * dx
        y[i] = H[i] - z_bed[i]
        if y[i] < 0.1: y[i] = y_n
    Q_vec = np.full(nx, Q)
    return H, Q_vec

print("Calculating initial conditions...")
H, Q = calculate_initial_conditions(Q_initial, H_downstream, z_bed, B, n_manning, S0, nx, dx)

# --- 4. Preissmann Solver (Final Corrected Version) ---
def solve_preissmann_final(H, Q, dt, dx, B, n, S0, z_bed, theta, Q_upstream_bc, H_downstream_bc, gate_pos_idx, gate_opening, Cd):
    nx = len(H)
    num_vars = 2 * nx
    A_matrix = np.zeros((num_vars, num_vars))
    b_vector = np.zeros(num_vars)
    epsilon = 1e-6

    # Assemble equations for internal points
    for i in range(nx - 1):
        if i == gate_pos_idx: continue

        H_avg = (H[i] + H[i+1]) / 2
        y_avg = H_avg - (z_bed[i] + z_bed[i+1]) / 2
        if y_avg < epsilon: y_avg = epsilon

        A_avg = B * y_avg
        R_avg = A_avg / (B + 2 * y_avg)
        Q_avg = (Q[i] + Q[i+1]) / 2

        # --- !! FINAL FIX: Correct and stable discretization for continuity equation ---
        # B * dH_avg/dt + dQ/dx = 0  => B*(H_new-H)/dt + theta*dQ_new/dx + (1-theta)*dQ/dx = 0
        # This leads to: B*(dH_i+dH_{i+1})/(2*dt) + theta/dx*(dQ_{i+1}-dQ_i) = -(Q_{i+1}-Q_i)/dx
        row = 2 * i
        A_matrix[row, 2*i]      = -theta / dx  # Coeff for dQ_i
        A_matrix[row, 2*(i+1)]  =  theta / dx  # Coeff for dQ_{i+1}
        A_matrix[row, 2*i+1]    =  B / (2 * dt)  # Coeff for dH_i
        A_matrix[row, 2*(i+1)+1]=  B / (2 * dt)  # Coeff for dH_{i+1}
        b_vector[row] = -(Q[i+1] - Q[i]) / dx

        # Momentum Equation
        row = 2 * i + 1
        Sf = (n**2 * Q_avg * abs(Q_avg)) / (A_avg**2 * R_avg**(4/3)) if A_avg > epsilon else 0

        # Coefficients for linearized momentum equation (dQ/dt + d(Q^2/A)/dx + gA*dH/dx = gA(S0-Sf))
        # dQ/dt term -> 1/dt * dQ
        # d(Q^2/A)/dx term -> 2Q/A*dQ/dx - Q^2/A^2*dA/dx
        # gA*dH/dx term -> gA*dH/dx
        A_matrix[row, 2*i]      = theta * (-2*Q_avg/(A_avg*dx))
        A_matrix[row, 2*(i+1)]  = theta * (2*Q_avg/(A_avg*dx))
        A_matrix[row, 2*i+1]    = theta * (g*Q_avg**2*B/(A_avg**2*dx) - g*A_avg/dx)
        A_matrix[row, 2*(i+1)+1]= theta * (-g*Q_avg**2*B/(A_avg**2*dx) + g*A_avg/dx)
        # Add dQ/dt term, averaged over the element
        A_matrix[row, 2*i]     += 1 / (2*dt)
        A_matrix[row, 2*(i+1)] += 1 / (2*dt)

        b_vector[row] = - (Q[i+1]**2/A_avg - Q[i]**2/A_avg)/dx - g*A_avg*(H[i+1]-H[i])/dx - g*A_avg*(S0-Sf)

    # Gate boundary
    i = gate_pos_idx
    H_up, H_down, y_up, y_down = H[i], H[i+1], H[i]-z_bed[i], H[i+1]-z_bed[i+1]

    A_matrix[2*i, :] = 0; A_matrix[2*i, 2*i] = 1.0; A_matrix[2*i, 2*(i+1)] = -1.0; b_vector[2*i] = Q[i+1] - Q[i]

    row = 2*i + 1
    A_matrix[row, :] = 0
    is_submerged = y_down > 0.9 * gate_opening
    if not is_submerged:
        Q_gate = Cd * B * gate_opening * np.sqrt(2 * g * (y_up + epsilon))
        dQ_dH_up = Cd * B * gate_opening * np.sqrt(g / (2 * (y_up + epsilon)))
        A_matrix[row, 2*(i+1)] = 1.0; A_matrix[row, 2*i+1] = -dQ_dH_up; b_vector[row] = Q_gate - Q[i+1]
    else:
        delta_H = H_up - H_down
        Q_gate = Cd * B * gate_opening * np.sqrt(2 * g * (delta_H + epsilon))
        C_sqrt = Cd * B * gate_opening * np.sqrt(g / 2) / (np.sqrt(delta_H + epsilon)) if delta_H > epsilon else 0
        A_matrix[row, 2*(i+1)] = 1.0; A_matrix[row, 2*i+1] = -C_sqrt; A_matrix[row, 2*(i+1)+1] = C_sqrt; b_vector[row] = Q_gate - Q[i+1]

    # Upstream/Downstream boundaries
    A_matrix[-2, :] = 0; A_matrix[-2, 0] = 1.0; b_vector[-2] = Q_upstream_bc - Q[0]
    A_matrix[-1, :] = 0; A_matrix[-1, -1] = 1.0; b_vector[-1] = H_downstream_bc - H[-1]

    try:
        sol = np.linalg.solve(A_matrix, b_vector)
        H_new, Q_new = H + sol[1::2], Q + sol[0::2]
        if np.any(np.isnan(H_new)) or np.any(np.isnan(Q_new)): return H, Q # Reject unstable step
        return H_new, Q_new
    except np.linalg.LinAlgError:
        return H, Q

# --- 5. Main Loop ---
H_results = [H.copy()]
Q_results = [Q.copy()]
print("Starting simulation...")
for t_step in range(nt):
    current_time = t_step * dt
    Q_bc = Q_initial + (Q_upstream_new - Q_initial) * min(1.0, (current_time - 3600) / 1800) if current_time > 3600 else Q_initial
    H, Q = solve_preissmann_final(H, Q, dt, dx, B, n_manning, S0, z_bed, theta, Q_bc, H_downstream, gate_pos_idx, gate_opening, Cd)
    if t_step % 60 == 0:
        H_results.append(H.copy())
        Q_results.append(Q.copy())
    if t_step % (nt // 10) == 0: print(f"Progress: {100 * t_step / nt:.0f}%")
print("Simulation complete!")

# --- 6. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
max_h, min_h = np.max(H_results), z_bed.min()
max_q, min_q = np.max(Q_results), np.min(Q_results)

def draw_gate(ax, gate_idx, z_bottom, h_water, gate_open):
    gate_x = x[gate_idx] / 1000
    gate_width_plot = (dx / 1000) * 2
    structure_top = max(h_water, z_bottom + 4.0) + 1.0
    ax.fill_between([gate_x - gate_width_plot, gate_x + gate_width_plot], z_bottom, structure_top, color='gray', alpha=0.8, zorder=10)
    ax.fill_between([gate_x - gate_width_plot, gate_x + gate_width_plot], z_bottom + gate_open, structure_top, color='black', alpha=0.9, zorder=11)
    ax.text(gate_x, structure_top - 0.5, f'Gate\nOpening: {gate_opening}m', ha='center', fontsize=10)

def update_plot(frame_index):
    if frame_index >= len(H_results): return # Safety check
    time_in_hours = frame_index * 60 * dt / 3600
    H_t = H_results[frame_index]
    Q_t = Q_results[frame_index]
    # Water Level Plot
    ax1.clear()
    ax1.plot(x / 1000, z_bed, color='brown', lw=2, label='Channel Bed')
    ax1.fill_between(x / 1000, 0, z_bed, color='peru', alpha=0.4)
    ax1.plot(x / 1000, H_t, color='blue', lw=2.5, label='Water Level')
    draw_gate(ax1, gate_pos_idx, z_bed[gate_pos_idx], H_t[gate_pos_idx], gate_opening)
    ax1.set_title(f'1D Hydraulic Model - Time: {time_in_hours:.2f} hours', fontsize=16)
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_ylim(min_h - 1, max_h + 1)
    ax1.set_xlim(0, L / 1000)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    # Flow Rate Plot
    ax2.clear()
    ax2.plot(x / 1000, Q_t, color='red', lw=2, label='Flow Rate')
    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel('Flow Rate (m³/s)', fontsize=12)
    ax2.set_ylim(min_q - 5, max_q + 5)
    ax2.set_xlim(0, L / 1000)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    fig.tight_layout()

print("Generating final GIF animation...")
ani = FuncAnimation(fig, update_plot, frames=len(H_results), interval=150, blit=False)
try:
    ani.save('hydraulic_simulation_final.gif', writer='pillow', fps=10)
    print("Successfully saved final GIF to: hydraulic_simulation_final.gif")
except Exception as e:
    print(f"Failed to save GIF: {e}")
# plt.show()
