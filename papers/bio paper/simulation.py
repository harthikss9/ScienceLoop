#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from math import exp, log

# Constants
R = 0.001987  # kcal/(mol*K)
T0 = 298.15  # K
DH = -50  # kcal/mol, assumed average
DS = -0.15  # kcal/(mol*K)
DCp = -1.15  # kcal/(mol*K)
m = 2.25  # kcal/(mol*M)
DH_U2TS = 20  # kcal/mol
DS_U2TS = 0.06  # kcal/(mol*K)
DCp_U2TS = -0.45  # kcal/(mol*K)
m_U2TS = 0.9  # kcal/(mol*M)
DH_N2TS = 65  # kcal/mol
DS_N2TS = 0.20  # kcal/(mol*K)
DCp_N2TS = 0.7  # kcal/(mol*K)
m_N2TS = 1.35  # kcal/(mol*M)
D_fold = 1e7  # s^-1
D_unfold = 1e7  # s^-1
DG_target = -6.5  # kcal/mol

# Temperature range
T = np.linspace(278.15, 338.15, 25)

# Function definitions
def k_fold(T, gnd):
    return D_fold * np.exp(-((DH_U2TS - T*DS_U2TS + DCp_U2TS*(T - T0 - T*np.log(T/T0)) - m_U2TS*gnd) / (R*T)))

def k_unfold(T, gnd):
    return D_unfold * np.exp(-((DH_N2TS - T*DS_N2TS + DCp_N2TS*(T - T0 - T*np.log(T/T0)) - m_N2TS*gnd) / (R*T)))

def DG_fold(T, gnd):
    return DH - T*DS + DCp*(T - T0 - T*np.log(T/T0)) - m*gnd

def gnd_constant_stability(T):
    return (DG_target - DH + T*DS - DCp*(T - T0 - T*np.log(T/T0))) / m

# Main simulation function
def main():
    # Condition 1: Constant denaturant concentration
    gnd_constants = [0, 1, 2, 3]
    for gnd in gnd_constants:
        k_folds = [k_fold(temp, gnd) for temp in T]
        plt.plot(1/T, np.log(k_folds), label=f'gnd={gnd} M')

    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(k_fold)')
    plt.title('Arrhenius Plot for Condition 1')
    plt.legend()
    plt.savefig('condition1_arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Condition 2: Constant stability
    gnd_variables = [gnd_constant_stability(temp) for temp in T]
    k_folds_variable = [k_fold(T[i], gnd_variables[i]) for i in range(len(T))]
    plt.plot(1/T, np.log(k_folds_variable), label='Constant DG')
    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(k_fold)')
    plt.title('Arrhenius Plot for Condition 2')
    plt.legend()
    plt.savefig('condition2_arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()