import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters ---
L = 20000.0; nx = 101; dx = L / (nx - 1)
B = 20.0; n_manning = 0.025; S0 = 0.0001; g = 9.81
gate_pos_idx = int(10000 / dx); gate_opening = 0.5; Cd = 0.6
Q_initial = 30.0; H_downstream_initial = 5.0

# --- 2. Grid and Bed Elevation ---
x = np.linspace(0, L, nx)
z_bed = S0 * (L - x)

# --- 3. Initial Conditions (Corrected for Gate Head Loss) ---
def calculate_initial_conditions_corrected(Q_val, H_down, z_bed_arr, B_val, n_val, S0_val, nx_val, dx_val, gate_idx, gate_open, cd_val):
    H = np.zeros(nx_val)

    # Part 1: Downstream of the gate
    H[nx_val-1] = H_down
    for i in range(nx_val - 2, gate_idx, -1):
        y_i1 = H[i+1] - z_bed_arr[i+1]
        if y_i1 <= 0.01: y_i1 = 0.01
        A_i1 = B_val * y_i1; R_i1 = A_i1 / (B_val + 2*y_i1)
        Sf_i1 = (Q_val * n_val / (A_i1 * R_i1**(2.0/3.0)))**2
        H[i] = H[i+1] + (S0_val - Sf_i1) * dx_val

    # Part 2: Across the gate
    H_down_gate = H[gate_idx+1]
    y_down_gate = H_down_gate - z_bed_arr[gate_idx+1]

    # Calculate required upstream energy head (Bernoulli)
    # For free flow: Q = Cd*a*B*sqrt(2gy_up) => y_up = (Q/(Cd*a*B))^2 / (2g)
    y_up_gate = (Q_val / (cd_val * gate_open * B_val))**2 / (2 * g)
    H_up_gate = y_up_gate + z_bed_arr[gate_idx]

    # Check for submerged flow and adjust if necessary
    if y_down_gate > 0.9 * gate_open:
        # Q = Cd*a*B*sqrt(2g*(H_up - H_down)) => H_up = (Q/(Cd*a*B))^2/(2g) + H_down
        H_up_gate = (Q_val / (cd_val * gate_open * B_val))**2 / (2 * g) + H_down_gate

    H[gate_idx] = H_up_gate

    # Part 3: Upstream of the gate
    for i in range(gate_idx - 1, -1, -1):
        y_i1 = H[i+1] - z_bed_arr[i+1]
        if y_i1 <= 0.01: y_i1 = 0.01
        A_i1 = B_val * y_i1; R_i1 = A_i1 / (B_val + 2*y_i1)
        Sf_i1 = (Q_val * n_val / (A_i1 * R_i1**(2.0/3.0)))**2
        H[i] = H[i+1] + (S0_val - Sf_i1) * dx_val

    return H, np.full(nx_val, Q_val)

print("Calculating corrected initial conditions...")
H_initial, Q_initial_profile = calculate_initial_conditions_corrected(Q_initial, H_downstream_initial, z_bed, B, n_manning, S0, nx, dx, gate_pos_idx, gate_opening, Cd)

# --- 4. Visualize Initial Conditions ---
print("Visualizing corrected initial conditions...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(x/1000, z_bed, color='brown', lw=2, label='Channel Bed')
ax1.fill_between(x/1000, 0, z_bed, color='peru', alpha=0.4)
ax1.plot(x/1000, H_initial, color='blue', lw=2.5, label='Initial Water Level')

gate_x = x[gate_pos_idx]/1000; gate_w = (dx/1000)*2
struct_top = np.max(H_initial) + 1.0
ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx], struct_top, color='gray', zorder=10)
ax1.fill_between([gate_x-gate_w, gate_x+gate_w], z_bed[gate_pos_idx]+gate_opening, struct_top, color='black', zorder=11)
ax1.text(gate_x, struct_top - 0.2, f'Gate\nOpening: {gate_opening}m', ha='center', va='top', fontsize=10)

ax1.set_title('Corrected Initial Steady Flow Conditions (with Gate Head Loss)', fontsize=16)
ax1.set_xlabel('Distance (km)'); ax1.set_ylabel('Elevation (m)')
ax1.set_ylim(np.min(z_bed)-1, np.max(H_initial)+1.5)
ax1.set_xlim(0, L/1000)
ax1.grid(True, linestyle='--'); ax1.legend(loc='upper right')

ax2.plot(x/1000, Q_initial_profile, color='red', lw=2, label='Initial Flow Rate')
ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('Flow Rate (m³/s)')
ax2.set_ylim(0, Q_initial * 1.5)
ax2.set_xlim(0, L/1000)
ax2.grid(True, linestyle='--'); ax2.legend(loc='upper right')

fig.tight_layout()
plt.savefig('initial_profile_corrected.png')
print("Successfully saved corrected initial profile to initial_profile_corrected.png")
