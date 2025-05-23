# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:01:10 2025

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Simulation configuration
t0, tend, dt = 0, 10, 1E-3
t = np.linspace(t0, tend, round((tend-t0)/dt) + 1)
u = np.sin(2*np.pi*0.25*t)  # 250 mHz (0.25 Hz) sine wave

# 4th order transfer function
def sistema_4orden(R1, C1, L1, R2, C2, L2):
    a4 = L1 * C1 * L2 * C2
    a3 = R1 * C1 * L2 * C2 + R2 * C2 * L1 * C1
    a2 = L1 * C1 + L2 * C2 + R1 * C1 * R2 * C2
    a1 = R1 * C1 + R2 * C2
    a0 = 1
    num = [L2*C1, 0, C1/C2]
    den = [a4, a3, a2, a1, a0]
    return ctrl.tf(num, den)


# Parameters
R1_n, C1_n, L1_n = 5, 0.02, 0.1
R2_n, C2_n, L2_n = 5, 0.02, 0.1
R1_h, C1_h, L1_h = 20, 0.01, 0.15
R2_h, C2_h, L2_h = 20, 0.01, 0.15


# Create systems and simulate
sys_normal = sistema_4orden(R1_n, C1_n, L1_n, R2_n, C2_n, L2_n)
sys_hpb = sistema_4orden(R1_h, C1_h, L1_h, R2_h, C2_h, L2_h)
_, y_normal = ctrl.forced_response(sys_normal, t, u)
_, y_hpb = ctrl.forced_response(sys_hpb, t, u)

# Adjust amplitudes:
y_normal = 1.5 * y_normal / np.max(np.abs(y_normal))  # Scale to ±1.5 V (3 Vpp) - Blue
y_hpb = 2 * y_hpb / np.max(np.abs(y_hpb))            # Scale to ±2 V (4 Vpp) - Red

# Plot configuration
plt.figure(figsize=(12, 6))
plt.plot(t, y_normal, color='blue', linewidth=2, label='Normal Case')
plt.plot(t, y_hpb, color='red', linewidth=2, label='HPB Case')

# Reference lines
plt.axhline(y=1.5, color='blue', linestyle=':', alpha=0.3)  # 3 Vpp upper limit
plt.axhline(y=-1.5, color='blue', linestyle=':', alpha=0.3) # 3 Vpp lower limit
plt.axhline(y=2.0, color='red', linestyle=':', alpha=0.3)   # 4 Vpp upper limit
plt.axhline(y=-2.0, color='red', linestyle=':', alpha=0.3)  # 4 Vpp lower limit

# Axis and style settings
plt.ylim(-2.5, 2.5)  # Additional margin for visualization
plt.yticks(np.arange(-2.5, 3.0, 0.5), fontsize=10)
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 1), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Voltage [V]', fontsize=12)
plt.title('System Response', fontsize=14)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

