#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import json

# Constants
R = 8.314  # J/(mol*K)
T0 = 295  # K
DF = 1e7  # s^-1, typical value
DH_U2TS = 40  # kcal/mol, typical value
DS_U2TS = 0.1  # kcal/(mol*K)
DCp_U2TS = 0.9  # kcal/(mol*K), typical value
m_U2TS = 0.5  # kcal/(mol*M)
DH = 60  # kcal/mol
DS = 0.2  # kcal/(mol*K)
DCp = 2.0  # kcal/(mol*K)
m = 1.5  # kcal/(mol*M)

# Simulation equations
def k(T, den):
    return DF * np.exp(-((DH_U2TS - T * DS_U2TS + DCp_U2TS * (T - T0 - T * np.log(T/T0)) + m_U2TS * den) / (R * T)))

def DG(T, den):
    return DH - T * DS + DCp * (T - T0 - T * np.log(T/T0)) - m * den

# Procedure steps
def main():
    # Step 2: Create temperature array
    T = np.linspace(278, 323, 25)
    results = []
    for temp in T:
        # Step 3: Compute k(T, den) for den=0 M
        rate = k(temp, 0)
        # Step 4: Solve for den that maintains constant DG
        DG_target = 0  # Target DG
        den_solve = lambda den: DG(temp, den) - DG_target
        den, = fsolve(den_solve, 0.5)
        # Step 5: Compute k(T, den) for constant DG
        rate_corrected = k(temp, den)
        results.append({'Temperature': temp, 'Denaturant': den, 'Rate': rate, 'Rate_corrected': rate_corrected})

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print('Results saved to results.csv')

if __name__ == '__main__':
    main()