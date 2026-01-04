import numpy as np
import matplotlib.pyplot as plt


mesh_sizes = np.array([70,50,60,80,90])

strain_rates = np.array([5366.58,
3591.81,
4456.52,
6354.51,
7783.32
])

delta = np.array([0.406,
0.408,
0.407,
0.38,
0.427
])

Tmax = np.array([2293.01,
2266.79,
2279.33,
2302,
2295.1
])

# Calculate mean and standard deviation
mean_strain = np.mean(strain_rates)
std_strain = np.std(strain_rates)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(mesh_sizes, strain_rates, 'o', label='Maximum strain rates')
""" plt.axhline(y=mean_strain, color='r', linestyle='--', label=f'Mean = {mean_strain:.2f}')
plt.axhline(y=mean_strain + std_strain, color='g', linestyle=':', label=f'Mean + std = {mean_strain + std_strain:.2f}')
plt.axhline(y=mean_strain - std_strain, color='g', linestyle=':', label=f'Mean - std = {mean_strain - std_strain:.2f}') """

plt.xlabel('Mesh Size', fontsize=14)
plt.ylabel('Strain Rate ($s^{-1}$)', fontsize=14)
plt.title('Strain Rate vs Mesh Size', fontsize=16)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.show()

# Calculate mean delta
mean_delta = np.mean(delta)

# Create the delta plot
plt.figure(figsize=(10, 6))
plt.plot(mesh_sizes, delta, 'o', label='Delta values')
plt.axhline(y=mean_delta, color='r', linestyle='--', label=f'Mean = {mean_delta:.3f}')

plt.xlabel('Mesh Size', fontsize=14)
plt.ylabel('Delta (mm)', fontsize=14)
plt.title('Delta vs Mesh Size', fontsize=16)
plt.ylim(0.3,0.5)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.show()

# Calculate mean Tmax
mean_Tmax = np.mean(Tmax)

# Create the Tmax plot
plt.figure(figsize=(10, 6))
plt.plot(mesh_sizes, Tmax, 'o', label='Maximum temperatures')
plt.axhline(y=mean_Tmax, color='r', linestyle='--', label=f'Mean = {mean_Tmax:.2f}')

plt.xlabel('Mesh Size', fontsize=14)
plt.ylabel('Maximum Temperature (K)', fontsize=14)
plt.ylim(2200,2350)
plt.title('Maximum Temperature vs Mesh Size', fontsize=16)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.show()