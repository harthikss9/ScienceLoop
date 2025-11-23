#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, log

# Constants
sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
c = 2.998e8  # Speed of light (m/s)
L_solar = 3.828e26  # Solar luminosity (W)
R_solar = 6.957e8  # Solar radius (m)
M_solar = 1.989e30  # Solar mass (kg)
T_eff = 38000  # Effective temperature for O6.5 III star (K)
R_star = 10.0 * R_solar  # Stellar radius for GHV-62024
Vinf_base = 1500 * 1e3  # Terminal wind velocity (m/s)
Z = 0.1  # Metallicity relative to solar

# Functions
def luminosity(R, T):
    return 4 * pi * sigma * R**2 * T**4

def wind_momentum(Mdot, Vinf, R):
    return Mdot * Vinf * sqrt(R)

def wind_velocity_law(Vinf, R, r, beta):
    return Vinf * (1 - R/r)**beta

def WLR(Dmom, L):
    return log(Dmom / (L/c))

def D_WLR(Z):
    Z_solar = 1  # Assuming Z_solar as unity for relative comparison
    return 0.6 * log(Z/Z_solar)

# Main simulation function
def main():
    # Step 2: Calculate stellar luminosity
    L = luminosity(R_star, T_eff)

    # Step 3-12: Parameter sweeps and calculations
    betas = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    Q_targets = np.array([1e-8, 5e-8, 1e-7]) * M_solar * 1e-6 / (1e3 * R_solar**1.5)
    results = []
    for Q_target in Q_targets:
        for beta in betas:
            Mdot = Q_target * Vinf_base * R_star**1.5
            Dmom = wind_momentum(Mdot, Vinf_base, R_star)
            WLR_deviation = WLR(Dmom, L) - (1.9 * log(L/L_solar) + D_WLR(Z))  # Using average x_WLR
            results.append((beta, Mdot, Dmom, WLR_deviation))

    # Step 11: Plotting results
    betas, Mdots, Dmoms, WLR_deviations = zip(*results)
    plt.figure()
    plt.plot(betas, Mdots, label='Mass-loss rate (Mdot)')
    plt.xlabel('Beta')
    plt.ylabel('Mdot (M_solar/yr)')
    plt.title('Mdot vs Beta')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()