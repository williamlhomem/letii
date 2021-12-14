'''
Script responsável por processar os dados oriundos do experimento sobre bombas hidráulicas
centrífugas. O objetivo desse script é gerar insights para a execução do relatório experimental
requerido pela disciplia de Laboratório de Sistemas Mecânicos II, ofertado pela Universidade
Federal do Espírito Santo.

Vitória,
2021.
'''

# %% Importações
from utils.uncertainty import uncertainty, partial_diff

import pandas as pd
import numpy as np

import json

# %% Carregamento dos dados experimentais
with open('data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['data'])
uncertainty_df = pd.Series(data['uncertainty'])

# %% Definição das funções referentes ao experimento 
rho = 997 # [kg/m^3]
n_prime = df['omega'].mean() # [rpm]
g = 9.81 # [m/s^2]

h_man = lambda q, p_suc, p_des, w, n, u, i: (p_des*10000 - p_suc + 90 + 1800 - 270 - 180)/1000
h_man_cor = lambda q, p_suc, p_des, w, n, u, i: (n_prime/n)**2 * h_man(q, p_suc, p_des, w, n, u, i)
q_cor = lambda q, p_suc, p_des, w, n, u, i: n_prime*q/n
w_cor = lambda q, p_suc, p_des, w, n, u, i: (n_prime/n)**3 * w
fp = lambda q, p_suc, p_des, w, n, u, i: w_cor(q, p_suc, p_des, w, n, u, i)/(u*i)
nu_g = lambda q, p_suc, p_des, w, n, u, i:\
     100*rho*g*(q_cor(q, p_suc, p_des, w, n, u, i) / (3600*1000))*h_man_cor(q, p_suc, p_des, w, n, u, i)/w_cor(q, p_suc, p_des, w, n, u, i)

# %% Cálculo das incertezas
cols = ['h_man', 'h_man_cor', 'q_cor', 'w_cor', 'fp', 'nu_g']
u_exp_df = pd.DataFrame(columns=cols)
calc_df = u_exp_df.copy()

uncertainty_df_cor = uncertainty_df.copy()
uncertainty_df_cor['P_suc'] /= 1000
uncertainty_df_cor['P_des'] /= 10
uncertainty_df_cor['W'] /= 1000



for index in df.index.values:
    args = df.loc[index]
    args.index = ['q', 'p_suc', 'p_des', 'w', 'n', 'u', 'i']
    args = dict(args)

    grads = [
        partial_diff(h_man, args=args),
        partial_diff(h_man_cor, args=args),
        partial_diff(q_cor, args=args),
        partial_diff(w_cor, args=args),
        partial_diff(fp, args=args),
        partial_diff(nu_g, args=args),
    ]

    calc_values = [
        h_man(**args),
        h_man_cor(**args),
        q_cor(**args),
        w_cor(**args),
        fp(**args),
        nu_g(**args)
    ]

    u_exp = np.array([uncertainty(grad, uncertainty_df_cor.values) for grad in grads])
    u_exp_df.loc[index] = u_exp

    calc_df.loc[index] = calc_values
# %%
