#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.stats as stats

# Constants
R = 0.001987  # kcal/(mol*K)
T_0 = 298.15  # K

# Thermodynamic parameters for different proteins
protein_params = {
    1: {'ΔH': 70, 'ΔS': 0.2, 'ΔCp': 2.0, 'm': 2.5, 'D_F': 1e7, 'ΔE_a': 5},
    2: {'ΔH': 85, 'ΔS': 0.25, 'ΔCp': 2.5, 'm': 3.0, 'D_F': 1e8, 'ΔE_a': 6}
}

# Target stabilities
ΔG_targets = [-2, 0, 2, 4, 6]

# Temperature range
T_range = np.linspace(278, 338, 20)

# Denaturant concentration range
Den_range = np.linspace(0, 8, 30)

# Function to calculate ΔG
def delta_G(T, den, ΔH, ΔS, ΔCp, m):
    return ΔH - T * ΔS + ΔCp * (T - T_0 - T * np.log(T / T_0)) - m * den

# Function to find denaturant concentration for a given ΔG target
def find_den(T, ΔG_target, ΔH, ΔS, ΔCp, m):
    func = lambda den: delta_G(T, den, ΔH, ΔS, ΔCp, m) - ΔG_target
    den_solution, = fsolve(func, 1)
    return den_solution

# Main simulation function
def main():
    for protein_id, params in protein_params.items():
        ΔH, ΔS, ΔCp, m, D_F, ΔE_a = params.values()
        for ΔG_target in ΔG_targets:
            den_solutions = [find_den(T, ΔG_target, ΔH, ΔS, ΔCp, m) for T in T_range]
            plt.plot(1/T_range, np.log(den_solutions), label=f'Protein {protein_id}, ΔG_target { ΔG_target}')
    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(k)')
    plt.title('Stability-corrected Arrhenius Plots')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()