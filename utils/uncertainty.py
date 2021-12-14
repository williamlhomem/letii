'''
Módulo responsável por auxiliar no cálculo das incertezas expandidas experimentais.
'''
from typing import Dict, Any

import numpy as np


def partial_diff(f: Any, h: float = 0.1, args=Dict) -> np.ndarray:
    '''
    Calcula a derivada parcial de uma função.
    '''
    diff = []
    for key, value in args.items():
        right = args.copy()
        right.update({key: value + h})

        left = args.copy()
        left.update({key: value - h})

        
        diff.append(
                (f(**right) - f(**left)) / (2*h)
            )

    return np.array(diff)

def uncertainty(grad: np.ndarray, uncertain: np.ndarray) -> float:
    '''
    Calcula a incerteza expandida de uma função.
    '''
    unc = 0
    for diff, u in zip(grad, uncertain):
        unc += diff**2 * u**2

    return np.sqrt(unc)
