# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import matplotlib.animation as animation


# Parameters
# Geometry
Lx = 2e-3 # Length of the domain (m)
Ly = 2e-3 # Length of the domain (m)
Lslot = 0.5e-3 # Length of the slot (m)
Lcoflow = 0.5e-3 # Length of the coflow (m)

Nx = 128  # Number of grid points in x-direction
Ny = 128# Number of grid points in y-direction
x = np.linspace(0, Lx, Nx)  # x-coordinates
y = np.linspace(0, Ly, Ny)  # y-coordinates
dx = x[1] - x[0]
dy = y[1] - y[0]
X,Y= np.meshgrid(x,y,indexing='ij')


# Flow conditions
Uslot = 1 # Velocity in the slot (m/s)
Tslot = 300 # Temperature in the slot (K)
Ucoflow = 0.2 # Velocity in the coflow (m/s)
Tcoflow = 300 # Temperature in the coflow (K)

# Fluid properties
rho = 1.1614 #Fluid density (kg/m^3)
nu = 15e-6 #Kinematic viscosity (m^2/s)
cp = 1200 #Specific heat capacity (J/kg/K)
D = nu #Schmidt number = 1
a = nu #Prandtl number = 1

#Time array
Umax = max(Uslot, Ucoflow)
CFL = 0.17
Fo=0.22

dt_adv = CFL * min(dx, dy) / Umax
dt_diff = Fo * min(dx, dy)**2 / a
dt = min(dt_adv, dt_diff)
print(dt)
t = np.arange(0.0, 2e-2 + dt, dt)

# Chemistry
Ta = 1e4 #Activation temperature (K)
A = 1.1e8 #Pre-exponential factor (1/s)
deltahN2 = 0 #Enthalpy of formation of N2 at T0 (J/mol)
deltahO2 = 0 #Enthalpy of formation of O2 at T0 (J/mol)
deltahCH4 = -74.9e3 #Enthalpy of formation of CH4 at T0 (J/mol)
deltahH2O = -241.818e3 #Enthalpy of formation of H2O at T0 (J/mol)
deltahCO2 = -393.52e3 #Enthalpy of formation of CO2 at T0 (J/mol)

WN2 = 28.0134e-3 #Molar mass of N2 (kg/mol)
WO2 = 31.9988e-3 #Molar mass of O2 (kg/mol)
WCH4 = 16.04e-3 #Molar mass of CH4 (kg/mol)
WH2O = 18.01528e-3 #Molar mass of H2O (kg/mol)
WCO2 = 44.01e-3 #Molar mass of CO2 (kg/mol)

Fo = D*dt/dx**2 #Fourier number
CFL = Uslot*dt/dx #CFL number

print(f'Fourier number: {Fo:.4f}, CFL number: {CFL:.4f}')
#%% Flow field

#Fractional step method to solve for the flow field

@jit(nopython=True)
def U_double_star(U, n, dt, dx, dy, nu, Lslot_idx, Lcoflow_idx, Uslot, Ucoflow):
    """
    Vectorized version: Compute advection-diffusion step for velocity field
    """
    Nx, Ny = U.shape[1], U.shape[2]
    U_double_star = np.zeros((Nx, Ny, 2))
    
    # Pre-compute constants
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    nu_dt = nu * dt
    
    # Extract interior domain
    u = U[n, 1:-1, 1:-1, 0]
    v = U[n, 1:-1, 1:-1, 1]
    
    # 1st order upwind scheme for advection terms - vectorized (STABLE)
    # u-component derivatives in x-direction
    du_dx_forward = (U[n, 2:, 1:-1, 0] - u) * dx_inv
    du_dx_backward = (u - U[n, :-2, 1:-1, 0]) * dx_inv
    du_dx = np.where(u > 0, du_dx_backward, du_dx_forward)
    
    dv_dx_forward = (U[n, 2:, 1:-1, 1] - v) * dx_inv
    dv_dx_backward = (v - U[n, :-2, 1:-1, 1]) * dx_inv
    dv_dx = np.where(u > 0, dv_dx_backward, dv_dx_forward)
    
    # u-component derivatives in y-direction
    du_dy_forward = (U[n, 1:-1, 2:, 0] - u) * dy_inv
    du_dy_backward = (u - U[n, 1:-1, :-2, 0]) * dy_inv
    du_dy = np.where(v > 0, du_dy_backward, du_dy_forward)
    
    dv_dy_forward = (U[n, 1:-1, 2:, 1] - v) * dy_inv
    dv_dy_backward = (v - U[n, 1:-1, :-2, 1]) * dy_inv
    dv_dy = np.where(v > 0, dv_dy_backward, dv_dy_forward)
    
    # Advection step
    u_star = u - dt * (u * du_dx + v * du_dy)
    v_star = v - dt * (u * dv_dx + v * dv_dy)
    
    # Diffusion step (central differences) - vectorized
    d2u_dx2 = (U[n, 2:, 1:-1, 0] - 2*u + U[n, :-2, 1:-1, 0]) * dx2_inv
    d2u_dy2 = (U[n, 1:-1, 2:, 0] - 2*u + U[n, 1:-1, :-2, 0]) * dy2_inv
    
    d2v_dx2 = (U[n, 2:, 1:-1, 1] - 2*v + U[n, :-2, 1:-1, 1]) * dx2_inv
    d2v_dy2 = (U[n, 1:-1, 2:, 1] - 2*v + U[n, 1:-1, :-2, 1]) * dy2_inv
    
    u_double_star_interior = u_star + nu_dt * (d2u_dx2 + d2u_dy2)
    v_double_star_interior = v_star + nu_dt * (d2v_dx2 + d2v_dy2)
    
    # Update interior points
    U_double_star[1:-1, 1:-1, 0] = u_double_star_interior
    U_double_star[1:-1, 1:-1, 1] = v_double_star_interior
    

    # Update left wall (stagnation plane)
    # No-penetration: u = 0
    U_double_star[0, :, 0] = 0
    
    # Free-slip: v is free with zero normal gradient (dv/dx = 0)
    # Simple implementation: copy from adjacent interior cell
    U_double_star[0, :, 1] = U_double_star[1, :, 1]


    # Update right wall (outlet)
    U_double_star[-1, :, :] = U_double_star[-2, :, :]  # Neumann BC at right wall
    
    # Update bottom boundary (y=0): inlets and walls
    # Slot inlet
    U_double_star[:Lslot_idx, 0, 0] = 0.0
    U_double_star[:Lslot_idx, 0, 1] = Uslot
    # Coflow inlet
    U_double_star[Lslot_idx:Lcoflow_idx, 0, 0] = 0.0
    U_double_star[Lslot_idx:Lcoflow_idx, 0, 1] = Ucoflow
    # Wall (excluding outlet at x=-1)
    U_double_star[Lcoflow_idx:-1, 0, 0] = 0.0
    U_double_star[Lcoflow_idx:-1, 0, 1] = 0.0
    
    # Update top boundary (y=-1): inlets and walls (symmetric)
    # Slot inlet
    U_double_star[:Lslot_idx, -1, 0] = 0.0
    U_double_star[:Lslot_idx, -1, 1] = -Uslot
    # Coflow inlet
    U_double_star[Lslot_idx:Lcoflow_idx, -1, 0] = 0.0
    U_double_star[Lslot_idx:Lcoflow_idx, -1, 1] = -Ucoflow
    # Wall (excluding outlet at x=-1)
    U_double_star[Lcoflow_idx:-1, -1, 0] = 0.0
    U_double_star[Lcoflow_idx:-1, -1, 1] = 0.0

    return U_double_star

@jit(nopython=True)
def laplacian(P, dx, dy):
    Nx, Ny = P.shape
    L = np.zeros_like(P)
    dx2 = dx*dx
    dy2 = dy*dy

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            L[i,j] = (
                (P[i+1,j] - 2*P[i,j] + P[i-1,j]) / dx2 +
                (P[i,j+1] - 2*P[i,j] + P[i,j-1]) / dy2
            )
    return L

@jit(nopython=True)
def smooth(P, b, dx, dy, omega, n_iter):
    Nx, Ny = P.shape
    dx2 = dx*dx
    dy2 = dy*dy
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)

    for _ in range(n_iter):
        Pnew = P.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                Pnew[i,j] = (1.0-omega)*P[i,j] + omega * (
                    ((P[i+1,j] + P[i-1,j]) / dx2 +
                     (P[i,j+1] + P[i,j-1]) / dy2 -
                     b[i,j]) / denom
                )

        # Same BCs as your SOR solver
        Pnew[0,:]  = Pnew[1,:]
        Pnew[-1,:] = 0.0
        Pnew[:,0]  = Pnew[:,1]
        Pnew[:,-1] = Pnew[:,-2]

        P[:] = Pnew

@jit(nopython=True)
def restrict(res):
    Nx, Ny = res.shape
    Nc_x = Nx // 2
    Nc_y = Ny // 2
    rc = np.zeros((Nc_x, Nc_y))

    for i in range(1, Nc_x-1):
        for j in range(1, Nc_y-1):
            ii = 2*i
            jj = 2*j
            rc[i,j] = (
                4*res[ii,jj] +
                2*(res[ii+1,jj] + res[ii-1,jj] +
                   res[ii,jj+1] + res[ii,jj-1]) +
                (res[ii+1,jj+1] + res[ii-1,jj-1] +
                 res[ii+1,jj-1] + res[ii-1,jj+1])
            ) / 16.0
    return rc

@jit(nopython=True)
def prolong(ec):
    Nc_x, Nc_y = ec.shape
    ef = np.zeros((2*Nc_x, 2*Nc_y))

    for i in range(Nc_x):
        for j in range(Nc_y):
            ef[2*i,2*j]       += ec[i,j]
            ef[2*i+1,2*j]     += 0.5*ec[i,j]
            ef[2*i,2*j+1]     += 0.5*ec[i,j]
            ef[2*i+1,2*j+1]   += 0.25*ec[i,j]
    return ef

@jit(nopython=True)
def v_cycle(P, b, dx, dy, level, max_level):

    # Pre-smoothing
    smooth(P, b, dx, dy, omega=0.8, n_iter=3)

    if level == max_level:
        smooth(P, b, dx, dy, omega=0.8, n_iter=20)
        return

    # Residual
    r = b - laplacian(P, dx, dy)

    # Restrict
    rc = restrict(r)
    ec = np.zeros_like(rc)

    # Recursive call
    v_cycle(ec, rc, 2*dx, 2*dy, level+1, max_level)

    # Prolongate + correct
    P += prolong(ec)

    # Post-smoothing
    smooth(P, b, dx, dy, omega=0.8, n_iter=3)

@jit(nopython=True)
def solve_poisson_pressure(U_star_star, dt, dx, dy, rho):

    Nx, Ny = U_star_star.shape[0], U_star_star.shape[1]
    P = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))

    # RHS: identical second-order divergence
    b[1:-1,1:-1] = (
        (U_star_star[2:,1:-1,0] - U_star_star[:-2,1:-1,0]) / (2*dx) +
        (U_star_star[1:-1,2:,1] - U_star_star[1:-1,:-2,1]) / (2*dy)
    ) * rho / dt

    # 4–6 V-cycles is usually plenty
    for _ in range(5):
        v_cycle(P, b, dx, dy, level=0, max_level=4)

    return P

@jit(nopython=True)
def velocity_correction(U, U_star_star, P_field, n, dt, dx, dy, rho):
    """
    Vectorized version: Apply velocity correction using pressure gradient
    """
    # Pre-compute constants
    dt_rho_2dx = dt / (rho * 2.0 * dx)
    dt_rho_2dy = dt / (rho * 2.0 * dy)
    
    # 2nd order central differences for pressure gradient - vectorized
    U[n+1, 1:-1, 1:-1, 0] = (U_star_star[1:-1, 1:-1, 0] - 
                              (P_field[2:, 1:-1] - P_field[:-2, 1:-1]) * dt_rho_2dx)
    U[n+1, 1:-1, 1:-1, 1] = (U_star_star[1:-1, 1:-1, 1] - 
                              (P_field[1:-1, 2:] - P_field[1:-1, :-2]) * dt_rho_2dy)
    
    U[n+1, 0, :, 0] = 0  # Left wall: u = 0
    U[n+1, 0, :, 1] = U_star_star[0, :, 1]  # Left wall: v from U_star_star
    
    # Right wall: Apply zero-gradient BC (Neumann) by copying adjacent interior values
    U[n+1, -1, :, :] = U[n+1, -2, :, :]
    
    return U

@jit(nopython=True)
def apply_velocity_bcs(U, n, Lslot_idx, Lcoflow_idx, Uslot, Ucoflow):
    """
    Apply velocity boundary conditions (already vectorized with array slicing)
    """
    # x = 0 (left wall): no-slip for u only
    U[n, 0, :, 0] = 0
    
    # x = Lx (right wall/outlet): do nothing (let flow exit naturally)
    
    # y = 0 (bottom inlet): Set velocity profile
    U[n, :Lslot_idx, 0, 0] = 0  # u = 0 in slot
    U[n, :Lslot_idx, 0, 1] = Uslot  # v = Uslot in slot
    
    U[n, Lslot_idx:Lcoflow_idx, 0, 0] = 0  # u = 0 in coflow
    U[n, Lslot_idx:Lcoflow_idx, 0, 1] = Ucoflow  # v = Ucoflow in coflow
    
    U[n, Lcoflow_idx:-1, 0, :] = 0  # u = v = 0 on wall (after coflow), excluding outlet
    
    # y = Ly (top inlet): Set velocity profile (symmetric to bottom)
    U[n, :Lslot_idx, -1, 0] = 0  # u = 0 in slot
    U[n, :Lslot_idx, -1, 1] = -Uslot  # v = -Uslot in slot (flowing inward)

    U[n, Lslot_idx:Lcoflow_idx, -1, 0] = 0  # u = 0 in coflow
    U[n, Lslot_idx:Lcoflow_idx, -1, 1] = -Ucoflow  # v = -Ucoflow in coflow (flowing inward)
    
    U[n, Lcoflow_idx:-1, -1, :] = 0  # u = v = 0 on wall (after coflow), excluding outlet
    
    return U

def U_fractional_step(U_ini, dt, dx, dy, rho, nu, t, tol=1e-6, check_interval=50):
    U = U_ini.copy()
    P_history = np.zeros((len(t), len(x), len(y)))  # Store pressure history
    
    # Compute indices for boundary conditions (using x-direction since inlet is along x)
    Lslot_idx = int(Lslot/dx)
    Lcoflow_idx = int((Lslot+Lcoflow)/dx)
    
    n_final = None  # Track when steady state is reached
    
    for n in range(len(t)-1):
        U_star_star = U_double_star(U,n,dt,dx,dy,nu,Lslot_idx,Lcoflow_idx,Uslot,Ucoflow)
        P_field = solve_poisson_pressure(U_star_star, dt, dx, dy, rho)
        P_history[n] = P_field  # Store pressure field
        U = velocity_correction(U, U_star_star, P_field, n, dt, dx, dy, rho)
        
        # Apply velocity boundary conditions
        U = apply_velocity_bcs(U, n+1, Lslot_idx, Lcoflow_idx, Uslot, Ucoflow)
        
        # Check for steady state every check_interval steps
        if n > 0 and n % check_interval == 0:
            # Compute L2 norm of velocity change
            du = U[n+1] - U[n]
            change = np.sqrt(np.mean(du**2))
            U_magnitude = np.sqrt(np.mean(U[n+1]**2))
            relative_change = change / (U_magnitude + 1e-10)
            
            if n % 100 == 0:
                print(f'Time step {n+1}/{len(t)-1} - Relative change: {relative_change:.2e}')
            
            if relative_change < tol:
                print(f'\nSteady state reached at time step {n+1} (t = {t[n+1]*1000:.2f} ms)')
                print(f'Relative velocity change: {relative_change:.2e}')
                n_final = n + 1
                break
        elif n % 100 == 0:
            print(f'Time step {n+1}/{len(t)-1} completed.')
    
    # If steady state not reached, use last time step
    if n_final is None:
        n_final = len(t) - 1
        print(f'\nMaximum time reached without achieving steady state.')
    
    # Store final pressure field
    U_star_star = U_double_star(U, min(n_final-1, len(t)-2), dt, dx, dy, nu, Lslot_idx, Lcoflow_idx, Uslot, Ucoflow)
    P_history[n_final] = solve_poisson_pressure(U_star_star, dt, dx, dy, rho)
    
    # Trim arrays to actual simulation length
    U = U[:n_final+1]
    P_history = P_history[:n_final+1]
    t_actual = t[:n_final+1]
    
    return U, P_history, t_actual

# Initial condition for velocity field
U_ini = np.zeros((len(t),len(x),len(y),2)) # 4D array to hold velocity field at each time step  

# Initial boundary conditions at y=0 (bottom inlet)
U_ini[:, :int(Lslot/dx), 0, 1] = Uslot # Inlet slot (v-velocity)
U_ini[:, int(Lslot/dx):int((Lslot+Lcoflow)/dx), 0, 1] = Ucoflow # Inlet coflow (v-velocity)

# Initial boundary conditions at y=Ly (top inlet)
U_ini[:, :int(Lslot/dx), -1, 1] = -Uslot # Inlet slot (v-velocity, negative = flowing inward)
U_ini[:, int(Lslot/dx):int((Lslot+Lcoflow)/dx), -1, 1] = -Ucoflow # Inlet coflow (v-velocity, negative = flowing inward)




U, P_history, t = U_fractional_step(U_ini, dt, dx, dy, rho, nu, t, tol=1e-6, check_interval=50)

# Plot velocity field at final time
fig_velocity = plt.figure(figsize=(10, 8))
ax_velocity = fig_velocity.add_subplot(111)

# Create quiver plot (subsample for visibility)
skip = 4  # Show every 4th vector

# Check velocity field statistics
u_final = U[-1, :, :, 0]
v_final = U[-1, :, :, 1]
print(f"U velocity range: [{np.min(u_final):.4f}, {np.max(u_final):.4f}]")
print(f"V velocity range: [{np.min(v_final):.4f}, {np.max(v_final):.4f}]")

# Calculate appropriate scale
max_vel = np.sqrt(np.max(u_final**2 + v_final**2))
print(f"Maximum velocity magnitude: {max_vel:.4f} m/s")

# Use automatic scaling if velocity is too small/large
if max_vel > 0.01:
    ax_velocity.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                       u_final[::skip, ::skip], v_final[::skip, ::skip],
                       width=0.002, headwidth=4, headlength=5)
else:
    print("WARNING: Velocities are very small or zero!")
    ax_velocity.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                       u_final[::skip, ::skip], v_final[::skip, ::skip])

ax_velocity.set_title(f'Velocity Field at t = {t[-1]*1000:.2f} ms (max vel: {max_vel:.3f} m/s)')
ax_velocity.set_xlabel('x (m)')
ax_velocity.set_ylabel('y (m)')
ax_velocity.set_aspect('equal')
ax_velocity.set_xlim(0, Lx)
ax_velocity.set_ylim(0, Ly)
plt.tight_layout()
plt.show()

# Calculate maximum strain rate on left wall
# Strain rate a = |dv/dy| on left wall (x=0)
dv_dy_left_wall = np.gradient(U[-1, 0, :, 1], y)
max_strain_rate = np.max(np.abs(dv_dy_left_wall))
print(f'Maximum strain rate on left wall: a = {max_strain_rate:.2f} 1/s')
#%% Velocity profiles on left wall
# Plot v-velocity profile on left wall
fig_vy_wall = plt.figure(figsize=(8, 6))
ax_vy = fig_vy_wall.add_subplot(111)
ax_vy.plot(y*1000, U[-1, 0, :, 1], 'b-', linewidth=2)
ax_vy.set_xlabel('y (mm)')
ax_vy.set_ylabel('v-velocity (m/s)')
ax_vy.set_title('v-velocity Profile on Left Wall at Final Time')
ax_vy.grid(True, alpha=0.3)
ax_vy.axhline(0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()
# Plot v-velocity profile on left wall at different times
fig_vy_evolution = plt.figure(figsize=(10, 7))
ax_vy_evo = fig_vy_evolution.add_subplot(111)

# Select time indices to plot (e.g., every 20% of simulation)
n_plots = 6
time_indices = np.linspace(0, len(t)-1, n_plots, dtype=int)

# Create colormap for different times
colors = plt.cm.viridis(np.linspace(0, 1, n_plots))

for idx, n_time in enumerate(time_indices):
    ax_vy_evo.plot(y*1000, U[n_time, 0, :, 1], 
                   color=colors[idx], linewidth=2,
                   label=f't = {t[n_time]*1000:.2f} ms')

ax_vy_evo.set_xlabel('y (mm)')
ax_vy_evo.set_ylabel('v-velocity (m/s)')
ax_vy_evo.set_title('v-velocity Profile Evolution on Left Wall')
ax_vy_evo.legend(loc='best')
ax_vy_evo.grid(True, alpha=0.3)
ax_vy_evo.axhline(0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# %% Species transport

# Species transport using advection-diffusion equation
# Species: N2, O2, CH4, H2O, CO2

@jit(nopython=True)
def iterate_species_vectorized(Y, n, U_field, dx, dy, dt, D):
    """
    Vectorized version: Solve species transport equation on 2D fields
    dY/dt + u*dY/dx + v*dY/dy = D*(d2Y/dx2 + d2Y/dy2)
    Using 1st-order upwind scheme for advection and central differences for diffusion
    """
    # Extract interior domain (avoiding boundaries)
    Y_curr = Y[n, 1:-1, 1:-1]
    u = U_field[1:-1, 1:-1, 0]
    v = U_field[1:-1, 1:-1, 1]
    
    # Pre-compute constants
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    # 1st order upwind scheme for advection - vectorized (STABLE)
    # For u-direction
    dYdx_forward = (Y[n, 2:, 1:-1] - Y_curr) * dx_inv  # Forward difference
    dYdx_backward = (Y_curr - Y[n, :-2, 1:-1]) * dx_inv  # Backward difference
    dYdx = np.where(u > 0, dYdx_backward, dYdx_forward)
    
    # For v-direction
    dYdy_forward = (Y[n, 1:-1, 2:] - Y_curr) * dy_inv  # Forward difference
    dYdy_backward = (Y_curr - Y[n, 1:-1, :-2]) * dy_inv  # Backward difference
    dYdy = np.where(v > 0, dYdy_backward, dYdy_forward)
    
    # Advection terms
    advection = -u * dYdx - v * dYdy
    
    # Diffusion terms (central difference) - vectorized
    d2Ydx2 = (Y[n, 2:, 1:-1] - 2*Y_curr + Y[n, :-2, 1:-1]) * dx2_inv
    d2Ydy2 = (Y[n, 1:-1, 2:] - 2*Y_curr + Y[n, 1:-1, :-2]) * dy2_inv
    diffusion = D * (d2Ydx2 + d2Ydy2)
    
    # Update interior points
    Y[n+1, 1:-1, 1:-1] = Y_curr + dt * (advection + diffusion)
    
    return Y

# Initialize mass fractions for each species
# Assuming air composition: N2 = 0.79, O2 = 0.21
# Resize to match actual time length from flow solver
Y_N2 = np.zeros((len(t), Nx, Ny))
Y_O2 = np.zeros((len(t), Nx, Ny))
Y_CH4 = np.zeros((len(t), Nx, Ny))
Y_H2O = np.zeros((len(t), Nx, Ny))
Y_CO2 = np.zeros((len(t), Nx, Ny))



""" # Initial conditions (air)
Y_N2[0, :, :] = 0.79  # N2 mass fraction
Y_O2[0, :, :] = 0.21  # O2 mass fraction
Y_CH4[0, :, :] = 0.0  # CH4 mass fraction
Y_H2O[0, :, :] = 0.0  # H2O mass fraction
Y_CO2[0, :, :] = 0.0  # CO2 mass fraction """

# Boundary conditions at inlets
Lslot_idx = int(Lslot/dx)
Lcoflow_idx = int((Lslot+Lcoflow)/dx)

# Slot inlet: air at bottom, methane at top
Y_N2[:, :Lslot_idx, 0] = 0.79
Y_O2[:, :Lslot_idx, 0] = 0.21
Y_CH4[:, :Lslot_idx, 0] = 0.0
Y_H2O[:, :Lslot_idx, 0] = 0.0
Y_CO2[:, :Lslot_idx, 0] = 0.0

Y_N2[:, :Lslot_idx, -1] = 0.0
Y_O2[:, :Lslot_idx, -1] = 0.0
Y_CH4[:, :Lslot_idx, -1] = 1.0
Y_H2O[:, :Lslot_idx, -1] = 0.0
Y_CO2[:, :Lslot_idx, -1] = 0.0

# Coflow inlet: N2 only at bottom and top
Y_N2[:, Lslot_idx:Lcoflow_idx, 0] = 1
Y_O2[:, Lslot_idx:Lcoflow_idx, 0] = 0
Y_CH4[:, Lslot_idx:Lcoflow_idx, 0] = 0.0
Y_H2O[:, Lslot_idx:Lcoflow_idx, 0] = 0.0
Y_CO2[:, Lslot_idx:Lcoflow_idx, 0] = 0.0

Y_N2[:, Lslot_idx:Lcoflow_idx, -1] = 1
Y_O2[:, Lslot_idx:Lcoflow_idx, -1] = 0
Y_CH4[:, Lslot_idx:Lcoflow_idx, -1] = 0.0
Y_H2O[:, Lslot_idx:Lcoflow_idx, -1] = 0.0
Y_CO2[:, Lslot_idx:Lcoflow_idx, -1] = 0.0


# Wall boundary conditions (no flux = Neumann BC)
# These will be maintained by the boundary treatment in iterate_species

# %% Simulation of an homogeneous reactor

def Q(Y_CH4,Y_O2,T):
    return A*rho**3*Y_CH4*Y_O2**2*np.exp(-Ta/T)/(WCH4*WO2**2)

def omega_pt_CH4(Q):
    return -WCH4*Q
def omega_pt_O2(Q):
    return -2*WO2*Q
def omega_pt_CO2(Q):
    return WCO2*Q
def omega_pt_H2O(Q):
    return 2*WH2O*Q

def omega_pt_T(omega_pt_k):
    return (-deltahCH4*omega_pt_k[0]/WCH4 - deltahO2*omega_pt_k[1]/WO2 + deltahCO2*omega_pt_k[2]/WCO2 + deltahH2O*omega_pt_k[3]/WH2O)
t0 = 0
tau = 0.01
tchem = np.linspace(t0,t0+tau,1000)

@jit(nopython=True)
def integrate_chemistry_vectorized(Y_CH4, Y_O2, Y_CO2, Y_H2O, T, n, dt_chem, n_substeps=10):
    """
    Integrate chemical source terms over a single time step using operator splitting
    Applies chemistry to all spatial points simultaneously (vectorized)
    
    Parameters:
    -----------
    Y_CH4, Y_O2, Y_CO2, Y_H2O : array (Nx, Ny)
        Current mass fraction fields at time step n
    T : array (Nx, Ny)
        Current temperature field at time step n
    n : int
        Current time step index
    dt_chem : float
        Chemistry time step (should be small for stiff chemistry)
    n_substeps : int
        Number of chemistry sub-steps for stability
    
    Returns:
    --------
    Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new : array (Nx, Ny)
        Updated mass fraction fields
    T_new : array (Nx, Ny)
        Updated temperature field
    """
    # Copy input fields to avoid modifying originals
    Y_CH4_new = Y_CH4[n].copy()
    Y_O2_new = Y_O2[n].copy()
    Y_CO2_new = Y_CO2[n].copy()
    Y_H2O_new = Y_H2O[n].copy()
    T_new = T[n].copy()
    
    dt_sub = 5e-8
    n_substeps = int(dt_chem / dt_sub)
    # Explicit Euler integration with sub-stepping
    for _ in range(n_substeps):
        # Compute reaction rate for all points (vectorized)
        Q_rate = A * rho**3 * Y_CH4_new * Y_O2_new**2 * np.exp(-Ta / T_new) / (WCH4 * WO2**2)
        
        """ print('Q',np.max(Q_rate)) """
        # Compute source terms (vectorized)
        omega_CH4 = -WCH4 * Q_rate
        omega_O2 = -2 * WO2 * Q_rate
        omega_CO2 = WCO2 * Q_rate
        omega_H2O = 2 * WH2O * Q_rate
        
        # Temperature source term
        omega_T = -(deltahCH4 * omega_CH4 / WCH4 +deltahO2 * omega_O2 / WO2 + 
                   deltahCO2 * omega_CO2 / WCO2 + deltahH2O * omega_H2O / WH2O)
        
        """ print('omega_T',np.max(omega_T)) """
        # Update all fields (vectorized)
        Y_CH4_new += dt_sub * omega_CH4 / rho
        Y_O2_new += dt_sub * omega_O2 / rho
        Y_CO2_new += dt_sub * omega_CO2 / rho
        Y_H2O_new += dt_sub * omega_H2O / rho
        T_new += dt_sub * omega_T / (rho * cp)
        
        
        # Clip mass fractions to physical bounds [0, 1]
        #Y_CH4_new = np.maximum(0.0, np.minimum(1.0, Y_CH4_new))
        #Y_O2_new = np.maximum(0.0, np.minimum(1.0, Y_O2_new))
        #Y_CO2_new = np.maximum(0.0, np.minimum(1.0, Y_CO2_new))
        #Y_H2O_new = np.maximum(0.0, np.minimum(1.0, Y_H2O_new))
    """ print('deltaT',np.max(T_new-T[n])) """
    return Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, T_new

# %% Temperature transport
# Initialize temperature field (resized to match actual time length)
T = np.zeros((len(t), Nx, Ny))
T[0, :, :] = Tcoflow  # Initial temperature everywhere is 300K
# Ignition zone: band around stagnation plane (x = Lx/2) with thickness δ = 0.5mm
delta_ignition = 0.5e-3
y_stagnation = Ly / 2
ignition_mask = np.abs(Y - y_stagnation) < delta_ignition / 2
T[0, ignition_mask] = 1000.0  # 1000K in ignition zone
# Boundary conditions for temperature
T[:, :Lslot_idx, -1] = Tslot  # Slot inlet temperature
T[:, Lslot_idx:Lcoflow_idx, 0] = Tcoflow  # Coflow inlet temperature (bottom)
T[:, Lslot_idx:Lcoflow_idx, -1] = Tcoflow  # Coflow inlet temperature (top)
T[:, :Lslot_idx, 0] = Tcoflow  # Bottom wall temperature
# Bottom wall temperature

print("Solving energy equation with chemistry and temperature transport...")
print(f"Flow solver completed with {len(t)} time steps (t_final = {t[-1]*1000:.2f} ms)")

# Number of chemistry sub-steps for stability
n_chem_substeps = 1000

# Time integration loop with operator splitting
for n in range(len(t)-1):
    if n % 100 == 0:
        print(f"Time step {n+1}/{len(t)-1}")
    
    # Step 1: Apply advection-diffusion for species (using velocity at time n)
    U_field = U[n]
    Y_CH4 = iterate_species_vectorized(Y_CH4, n, U_field, dx, dy, dt, D)
    Y_O2 = iterate_species_vectorized(Y_O2, n, U_field, dx, dy, dt, D)
    Y_N2 = iterate_species_vectorized(Y_N2, n, U_field, dx, dy, dt, D)
    Y_CO2 = iterate_species_vectorized(Y_CO2, n, U_field, dx, dy, dt, D)
    Y_H2O = iterate_species_vectorized(Y_H2O, n, U_field, dx, dy, dt, D)
    
    # Step 2: Apply advection-diffusion for temperature
    T = iterate_species_vectorized(T, n, U_field, dx, dy, dt, a)
    
    # Step 3: Apply chemistry to transported fields
    Y_CH4_chem, Y_O2_chem, Y_CO2_chem, Y_H2O_chem, T_chem = integrate_chemistry_vectorized(
        Y_CH4, Y_O2, Y_CO2, Y_H2O, T, n+1, dt, n_substeps=n_chem_substeps
    )
    
    # Update fields after chemistry step
    Y_CH4[n+1] = Y_CH4_chem
    Y_O2[n+1] = Y_O2_chem
    Y_CO2[n+1] = Y_CO2_chem
    Y_H2O[n+1] = Y_H2O_chem
    T[n+1] = T_chem  # Update temperature from chemistry
    """ print(f"  Max T after chemistry: {np.max(T[n+1]):.1f}K") """
    # Re-apply boundary conditions for all fields
    # Temperature BCs
    T[n+1, :Lslot_idx, -1] = Tslot
    T[n+1, Lslot_idx:Lcoflow_idx, 0] = Tcoflow
    T[n+1, Lslot_idx:Lcoflow_idx, -1] = Tcoflow
    T[n+1, :Lslot_idx, 0] = Tcoflow
    T[n+1, 0, :] = T[n+1, 1, :]  # Left wall (zero gradient)
    T[n+1, -1, :] = T[n+1, -2, :]  # Right wall (zero gradient)
    
    if t[n] < np.max(t)/2:
        T[n+1, ignition_mask] = 1000.0  # 1000K in ignition zone


    # Species BCs (same as before)
    # Slot inlet (bottom, y=0)
    Y_N2[n+1, :Lslot_idx, 0] = 0.79
    Y_O2[n+1, :Lslot_idx, 0] = 0.21
    Y_CH4[n+1, :Lslot_idx, 0] = 0.0
    Y_H2O[n+1, :Lslot_idx, 0] = 0.0
    Y_CO2[n+1, :Lslot_idx, 0] = 0.0
    
    # Slot inlet (top, y=Ly)
    Y_N2[n+1, :Lslot_idx, -1] = 0.0
    Y_O2[n+1, :Lslot_idx, -1] = 0.0
    Y_CH4[n+1, :Lslot_idx, -1] = 1.0
    Y_H2O[n+1, :Lslot_idx, -1] = 0.0
    Y_CO2[n+1, :Lslot_idx, -1] = 0.0
    
    # Coflow inlet (bottom, y=0)
    Y_N2[n+1, Lslot_idx:Lcoflow_idx, 0] = 1
    Y_O2[n+1, Lslot_idx:Lcoflow_idx, 0] = 0
    Y_CH4[n+1, Lslot_idx:Lcoflow_idx, 0] = 0.0
    Y_H2O[n+1, Lslot_idx:Lcoflow_idx, 0] = 0.0
    Y_CO2[n+1, Lslot_idx:Lcoflow_idx, 0] = 0.0
    
    # Coflow inlet (top, y=Ly)
    Y_N2[n+1, Lslot_idx:Lcoflow_idx, -1] = 1
    Y_O2[n+1, Lslot_idx:Lcoflow_idx, -1] = 0
    Y_CH4[n+1, Lslot_idx:Lcoflow_idx, -1] = 0.0
    Y_H2O[n+1, Lslot_idx:Lcoflow_idx, -1] = 0.0
    Y_CO2[n+1, Lslot_idx:Lcoflow_idx, -1] = 0.0
    
    # Wall boundaries (zero gradient)
    Y_N2[n+1, 0, :] = Y_N2[n+1, 1, :]
    Y_O2[n+1, 0, :] = Y_O2[n+1, 1, :]
    Y_CH4[n+1, 0, :] = Y_CH4[n+1, 1, :]
    Y_H2O[n+1, 0, :] = Y_H2O[n+1, 1, :]
    Y_CO2[n+1, 0, :] = Y_CO2[n+1, 1, :]

    Y_N2[n+1, -1, :] = Y_N2[n+1, -2, :]
    Y_O2[n+1, -1, :] = Y_O2[n+1, -2, :]
    Y_CH4[n+1, -1, :] = Y_CH4[n+1, -2, :]
    Y_H2O[n+1, -1, :] = Y_H2O[n+1, -2, :]
    Y_CO2[n+1, -1, :] = Y_CO2[n+1, -2, :]

print("Energy equation with chemistry and temperature transport solved!")
# %% Visualization of results
# Calculate maximum strain rate on left wall
# Strain rate a = |dv/dy| on left wall (x=0)
dv_dy_left_wall = np.gradient(U[-1, 0, :, 1], y)
max_strain_rate = np.max(np.abs(dv_dy_left_wall))
print(f'Maximum strain rate on left wall: a = {max_strain_rate:.2f} 1/s')

# Measure diffusive zone thickness on left wall using N2
# Get N2 mass fraction on left wall (x=0) at final time
Y_N2_left_wall = Y_N2[-1, 0, :]

# Get reference values
Y_slot_N2 = 0.79  # N2 mass fraction in slot (air at bottom)
Y_10 = 0.1 * Y_slot_N2
Y_90 = 0.9 * Y_slot_N2

# Find indices where Y_N2 is between 10% and 90% of slot value
mask = (Y_N2_left_wall >= Y_10) & (Y_N2_left_wall <= Y_90)
indices = np.where(mask)[0]

if len(indices) > 0:
    # Calculate thickness as extent of this region
    y_min = y[indices[0]]
    y_max = y[indices[-1]]
    delta_diff = y_max - y_min
    print(f'Diffusive zone thickness on left wall: δ = {delta_diff*1000:.3f} mm')
    print(f'  Location: y ∈ [{y_min*1000:.3f}, {y_max*1000:.3f}] mm')
else:
    print('No diffusive zone detected with specified criteria')

# Plot N2 profile on left wall
fig_N2_wall = plt.figure(figsize=(8, 6))
ax_N2 = fig_N2_wall.add_subplot(111)
ax_N2.plot(y*1000, Y_N2_left_wall, 'b-', linewidth=2, label='Y_N2')
ax_N2.axhline(Y_10, color='r', linestyle='--', label=f'10% Y_slot = {Y_10:.3f}')
ax_N2.axhline(Y_90, color='g', linestyle='--', label=f'90% Y_slot = {Y_90:.3f}')
if len(indices) > 0:
    ax_N2.axvspan(y_min*1000, y_max*1000, alpha=0.3, color='yellow', 
                  label=f'δ = {delta_diff*1000:.3f} mm')
ax_N2.set_xlabel('y (mm)')
ax_N2.set_ylabel('Y_N2')
ax_N2.set_title('N2 Mass Fraction Profile on Left Wall')
ax_N2.legend()
ax_N2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print maximum temperature reached
max_temp = np.max(T)
print(f'Maximum temperature reached: {max_temp:.2f} K')

# Plot final species distribution
fig_final, axes_final = plt.subplots(2, 3, figsize=(15, 10))
fig_final.suptitle(f'Final Species Mass Fractions at t = {t[-1]*1000:.2f} ms', fontsize=14)

# CH4
im1 = axes_final[0, 0].pcolor(X, Y, Y_CH4[-1], cmap='hot')
axes_final[0, 0].set_title('CH4 (Fuel)')
axes_final[0, 0].set_xlabel('x (m)')
axes_final[0, 0].set_ylabel('y (m)')
axes_final[0, 0].set_aspect('equal')
fig_final.colorbar(im1, ax=axes_final[0, 0])

# O2
im2 = axes_final[0, 1].pcolor(X, Y, Y_O2[-1], cmap='Blues')
axes_final[0, 1].set_title('O2 (Oxidizer)')
axes_final[0, 1].set_xlabel('x (m)')
axes_final[0, 1].set_ylabel('y (m)')
axes_final[0, 1].set_aspect('equal')
fig_final.colorbar(im2, ax=axes_final[0, 1])

# N2
im3 = axes_final[0, 2].pcolor(X, Y, Y_N2[-1], cmap='Greens')
axes_final[0, 2].set_title('N2 (Inert)')
axes_final[0, 2].set_xlabel('x (m)')
axes_final[0, 2].set_ylabel('y (m)')
axes_final[0, 2].set_aspect('equal')
fig_final.colorbar(im3, ax=axes_final[0, 2])

# H2O
im4 = axes_final[1, 0].pcolor(X, Y, Y_H2O[-1], cmap='cool')
axes_final[1, 0].set_title('H2O (Product)')
axes_final[1, 0].set_xlabel('x (m)')
axes_final[1, 0].set_ylabel('y (m)')
axes_final[1, 0].set_aspect('equal')
fig_final.colorbar(im4, ax=axes_final[1, 0])

# CO2
im5 = axes_final[1, 1].pcolor(X, Y, Y_CO2[-1], cmap='plasma')
axes_final[1, 1].set_title('CO2 (Product)')
axes_final[1, 1].set_xlabel('x (m)')
axes_final[1, 1].set_ylabel('y (m)')
axes_final[1, 1].set_aspect('equal')
fig_final.colorbar(im5, ax=axes_final[1, 1])

# Temperature
im6 = axes_final[1, 2].pcolor(X, Y, T[-1], cmap='hot')
axes_final[1, 2].set_title('Temperature (K)')
axes_final[1, 2].set_xlabel('x (m)')
axes_final[1, 2].set_ylabel('y (m)')
axes_final[1, 2].set_aspect('equal')
fig_final.colorbar(im6, ax=axes_final[1, 2])

plt.tight_layout()
plt.show()

# Plot initial conditions
fig_init, axes_init = plt.subplots(2, 3, figsize=(15, 10))
fig_init.suptitle('Initial Conditions at t = 0', fontsize=14)

# CH4
im1_init = axes_init[0, 0].pcolor(X, Y, Y_CH4[0], cmap='hot')
axes_init[0, 0].set_title('CH4 (Fuel)')
axes_init[0, 0].set_xlabel('x (m)')
axes_init[0, 0].set_ylabel('y (m)')
axes_init[0, 0].set_aspect('equal')
fig_init.colorbar(im1_init, ax=axes_init[0, 0])

# O2
im2_init = axes_init[0, 1].pcolor(X, Y, Y_O2[0], cmap='Blues')
axes_init[0, 1].set_title('O2 (Oxidizer)')
axes_init[0, 1].set_xlabel('x (m)')
axes_init[0, 1].set_ylabel('y (m)')
axes_init[0, 1].set_aspect('equal')
fig_init.colorbar(im2_init, ax=axes_init[0, 1])

# N2
im3_init = axes_init[0, 2].pcolor(X, Y, Y_N2[0], cmap='Greens')
axes_init[0, 2].set_title('N2 (Inert)')
axes_init[0, 2].set_xlabel('x (m)')
axes_init[0, 2].set_ylabel('y (m)')
axes_init[0, 2].set_aspect('equal')
fig_init.colorbar(im3_init, ax=axes_init[0, 2])

# H2O
im4_init = axes_init[1, 0].pcolor(X, Y, Y_H2O[0], cmap='cool')
axes_init[1, 0].set_title('H2O (Product)')
axes_init[1, 0].set_xlabel('x (m)')
axes_init[1, 0].set_ylabel('y (m)')
axes_init[1, 0].set_aspect('equal')
fig_init.colorbar(im4_init, ax=axes_init[1, 0])

# CO2
im5_init = axes_init[1, 1].pcolor(X, Y, Y_CO2[0], cmap='plasma')
axes_init[1, 1].set_title('CO2 (Product)')
axes_init[1, 1].set_xlabel('x (m)')
axes_init[1, 1].set_ylabel('y (m)')
axes_init[1, 1].set_aspect('equal')
fig_init.colorbar(im5_init, ax=axes_init[1, 1])

# Temperature
im6_init = axes_init[1, 2].pcolor(X, Y, T[0], cmap='hot')
axes_init[1, 2].set_title('Temperature (K)')
axes_init[1, 2].set_xlabel('x (m)')
axes_init[1, 2].set_ylabel('y (m)')
axes_init[1, 2].set_aspect('equal')
fig_init.colorbar(im6_init, ax=axes_init[1, 2])

plt.tight_layout()
plt.show()
# %%
