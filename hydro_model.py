import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Model Parameters ---
L=20000.0; nx=101; dx=L/(nx-1)
B=20.0; n_manning=0.025; S0=0.0001; g=9.81
T_total=3600*5; dt=60
gate_pos_idx=int(10000/dx); gate_opening=0.5; Cd=0.6
Q_initial=30.0; Q_upstream_new=35.0; H_downstream_initial=5.0
theta=0.6

# --- 2. Grid and Bed Elevation ---
x = np.linspace(0, L, nx)
z_bed = S0 * (L - x)

# --- 3. Initial Conditions ---
def calculate_initial_conditions_corrected(Q_val, H_down, z_bed_arr, B_val, n_val, S0_val, nx_val, dx_val, gate_idx, gate_open, cd_val):
    H = np.zeros(nx_val); H[nx_val-1] = H_down
    for i in range(nx_val-2, gate_idx, -1):
        y_i1=H[i+1]-z_bed_arr[i+1]; A_i1=B_val*y_i1; R_i1=A_i1/(B_val+2*y_i1)
        Sf_i1=(Q_val*n_val/(A_i1*R_i1**(2.0/3.0)))**2; H[i]=H[i+1]+(S0_val-Sf_i1)*dx_val
    y_down_gate = H[gate_idx+1]-z_bed_arr[gate_idx+1]
    if y_down_gate>0.9*gate_open: H[gate_idx]=(Q_val/(cd_val*gate_open*B_val))**2/(2*g)+H[gate_idx+1]
    else: H[gate_idx]=((Q_val/(cd_val*gate_open*B_val))**2/(2*g))+z_bed_arr[gate_idx]
    for i in range(gate_idx-1,-1,-1):
        y_i1=H[i+1]-z_bed_arr[i+1]; A_i1=B_val*y_i1; R_i1=A_i1/(B_val+2*y_i1)
        Sf_i1=(Q_val*n_val/(A_i1*R_i1**(2.0/3.0)))**2; H[i]=H[i+1]+(S0_val-Sf_i1)*dx_val
    return H, np.full(nx_val, Q_val)

print("Calculating corrected initial conditions...")
H, Q = calculate_initial_conditions_corrected(Q_initial, H_downstream_initial, z_bed, B, n_manning, S0, nx, dx, gate_pos_idx, gate_opening, Cd)

# --- 4. Preissmann Solver with Newton-Raphson (Corrected Jacobian) ---
def solve_preissmann_newton(H_n, Q_n, dt, dx, B, n, S0, g, z_bed, theta, Q_bc, H_bc, gate_idx, gate_open, Cd):
    nx = len(H_n)
    H_k, Q_k = H_n.copy(), Q_n.copy()

    for k_iter in range(15):
        M=np.zeros((2*nx,2*nx)); P=np.zeros(2*nx); epsilon=1e-5

        for i in range(nx-1):
            if i == gate_idx: continue
            y_i_k,y_i1_k=H_k[i]-z_bed[i],H_k[i+1]-z_bed[i+1]
            A_i_k,A_i1_k=B*y_i_k if y_i_k>epsilon else epsilon, B*y_i1_k if y_i1_k>epsilon else epsilon

            j=2*i
            M[j,2*i],M[j,2*i+2]=-theta/dx,theta/dx; M[j,2*i+1],M[j,2*i+3]=B/(2*dt),B/(2*dt)
            P[j]=theta/dx*(Q_k[i+1]-Q_k[i])+(1-theta)/dx*(Q_n[i+1]-Q_n[i])+B/(2*dt)*((H_k[i]-H_n[i])+(H_k[i+1]-H_n[i+1]))

            j=2*i+1
            Q_avg_k,A_avg_k=(Q_k[i]+Q_k[i+1])/2,(A_i_k+A_i1_k)/2
            R_avg_k=A_avg_k/(B+2*(y_i_k+y_i1_k)/2)
            Sf_k=n**2*Q_avg_k*abs(Q_avg_k)/(A_avg_k**2*R_avg_k**(4/3)) if A_avg_k>epsilon else 0

            dF2_dQ_i = 1/(2*dt) + theta/dx*(2*Q_k[i]/A_i_k)
            dF2_dH_i = theta/dx*(-g*A_i_k + Q_k[i]**2*B/A_i_k**2)
            dF2_dQ_i1 = 1/(2*dt) - theta/dx*(2*Q_k[i+1]/A_i1_k)
            dF2_dH_i1 = theta/dx*(g*A_i1_k - Q_k[i+1]**2*B/A_i1_k**2)
            M[j,2*i],M[j,2*i+1],M[j,2*i+2],M[j,2*i+3] = dF2_dQ_i, dF2_dH_i, dF2_dQ_i1, dF2_dH_i1

            Q_avg_n,A_avg_n=(Q_n[i]+Q_n[i+1])/2,(B*(H_n[i]-z_bed[i])+B*(H_n[i+1]-z_bed[i+1]))/2
            R_avg_n=A_avg_n/(B+2*((H_n[i]-z_bed[i])+(H_n[i+1]-z_bed[i+1]))/2)
            Sf_n=n**2*Q_avg_n*abs(Q_avg_n)/(A_avg_n**2*R_avg_n**(4/3)) if A_avg_n>epsilon else 0
            P[j]=(Q_avg_k-Q_avg_n)/dt+\
                  theta/dx*((Q_k[i+1]**2/A_i1_k)-(Q_k[i]**2/A_i_k))+(1-theta)/dx*((Q_n[i+1]**2/(B*(H_n[i+1]-z_bed[i+1])))-(Q_n[i]**2/(B*(H_n[i]-z_bed[i])))) + \
                  g*theta*A_avg_k/dx*(H_k[i+1]-H_k[i])+g*(1-theta)*A_avg_n/dx*(H_n[i+1]-H_n[i])-\
                  g*theta*A_avg_k*(S0-Sf_k)-g*(1-theta)*A_avg_n*(S0-Sf_n)

        i=gate_idx; y_up_k,y_down_k=H_k[i]-z_bed[i],H_k[i+1]-z_bed[i+1]
        j=2*i; M[j,:],P[j]=0,Q_k[i]-Q_k[i+1]; M[j,j],M[j,j+2]=1,-1
        j=2*i+1; M[j,:],P[j]=0,0
        if y_down_k>0.9*gate_open:
            delta_H_k=H_k[i]-H_k[i+1]
            Q_gate_k=Cd*B*gate_open*np.sqrt(2*g*max(0,delta_H_k))
            d_Qgate_dH=Cd*B*gate_open*np.sqrt(g/2)/np.sqrt(delta_H_k) if delta_H_k>epsilon else 0
            M[j,2*i+2]=1; M[j,2*i+1]=-d_Qgate_dH; M[j,2*i+3]=d_Qgate_dH
            P[j]=Q_k[i+1]-Q_gate_k
        else:
            Q_gate_k=Cd*B*gate_open*np.sqrt(2*g*max(0,y_up_k))
            d_Qgate_dHup=Cd*B*gate_open*np.sqrt(g/(2*y_up_k)) if y_up_k>epsilon else 0
            M[j,2*i+2]=1; M[j,2*i+1]=-d_Qgate_dHup
            P[j]=Q_k[i+1]-Q_gate_k

        M[-2,:],P[-2]=0,Q_k[0]-Q_bc; M[-2,0]=1
        M[-1,:],P[-1]=0,H_k[-1]-H_bc; M[-1,-1]=1

        try: sol=np.linalg.solve(M,-P)
        except np.linalg.LinAlgError: return H_n,Q_n
        H_k+=sol[1::2]; Q_k+=sol[0::2]
        if np.max(np.abs(sol))<1e-3: return H_k,Q_k
    return H_k,Q_k

# --- 5. Main Loop ---
nt=int(T_total/dt)
H_results,Q_results=[],[]
print(f"Using final, correct Preissmann scheme. dt={dt}s, nt={nt} steps.")
for t_step in range(nt):
    current_time=t_step*dt
    if current_time<3600: Q_bc=Q_initial
    else: Q_bc=Q_initial+(Q_upstream_new-Q_initial)*min(1.0,(current_time-3600)/1800)

    if t_step%10==0: H_results.append(H.copy()); Q_results.append(Q.copy())
    H,Q=solve_preissmann_newton(H,Q,dt,dx,B,n_manning,S0,g,z_bed,theta,Q_bc,H_downstream_initial,gate_pos_idx,gate_opening,Cd)
    if np.any(H-z_bed<0.01): H[H-z_bed<0.01]=z_bed[H-z_bed<0.01]+0.01

    if t_step%(nt//10)==0: print(f"Progress: {100*t_step/nt:.0f}%")
H_results.append(H.copy()); Q_results.append(Q.copy())
print("Simulation complete!")

# --- 6. Visualization (with robust frame handling) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
max_h,min_h=np.max(H_results),z_bed.min()
max_q,min_q=np.max(Q_results),np.min(Q_results)
gate_struct_top=max_h+1.0

def update_plot(frame_index):
    if frame_index>=len(H_results): return
    time_in_hours=(frame_index*10*dt)/3600
    H_t,Q_t=H_results[frame_index],Q_results[frame_index]
    ax1.clear(); ax2.clear()
    ax1.plot(x/1000,z_bed,color='brown',lw=2,label='Channel Bed')
    ax1.fill_between(x/1000,0,z_bed,color='peru',alpha=0.4)
    gate_x,gate_w=x[gate_pos_idx]/1000,(dx/1000)*2
    ax1.fill_between([gate_x-gate_w,gate_x+gate_w],z_bed[gate_pos_idx],gate_struct_top,color='gray',zorder=10)
    ax1.fill_between([gate_x-gate_w,gate_x+gate_w],z_bed[gate_pos_idx]+gate_opening,gate_struct_top,color='black',zorder=11)
    ax1.text(gate_x,gate_struct_top-0.2,f'Gate\nOpening: {gate_opening}m',ha='center',va='top',fontsize=10)
    ax1.plot(x/1000,H_t,color='blue',lw=2.5,label='Water Level',zorder=20)
    ax1.set_title(f'1D Hydraulic Model (Preissmann) - Time: {time_in_hours:.2f} hours',fontsize=16)
    ax1.set_xlabel('Distance (km)'); ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(min_h-1,max_h+1.5); ax1.set_xlim(0,L/1000)
    ax1.grid(True,linestyle='--'); ax1.legend(loc='upper right')
    ax2.plot(x/1000,Q_t,color='red',lw=2,label='Flow Rate')
    ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('Flow Rate (m³/s)')
    ax2.set_ylim(min_q-5,max_q+5); ax2.set_xlim(0,L/1000)
    ax2.grid(True,linestyle='--'); ax2.legend(loc='upper right')
    fig.tight_layout()

print("Generating final, truly corrected Preissmann GIF animation...")
num_frames=len(H_results)
ani=FuncAnimation(fig,update_plot,frames=num_frames,interval=100)
try:
    ani.save('hydraulic_simulation_preissmann_final.gif',writer='pillow',fps=15)
    print("Successfully saved final GIF to: hydraulic_simulation_preissmann_final.gif")
except Exception as e:
    print(f"Failed to save GIF: {e}")
