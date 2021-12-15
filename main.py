'''
Script responsável por processar os dados oriundos do experimento sobre bombas hidráulicas
centrífugas. O objetivo desse script é gerar insights para a execução do relatório experimental
requerido pela disciplia de Laboratório de Sistemas Mecânicos II, ofertado pela Universidade
Federal do Espírito Santo.

Vitória,
2021.
'''

# %% Importações
from logging import error
from utils.uncertainty import uncertainty, partial_diff
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

h_man = lambda **kwargs: (kwargs['p_des']*10000 - kwargs['p_suc'] + 90 + 1800 - 270 - 180)/1000
h_man_cor = lambda **kwargs: (n_prime/kwargs['n'])**2 * h_man(**kwargs)
h_ratio = lambda **kwargs: h_man_cor(**kwargs)/h_man(**kwargs)

q_cor = lambda **kwargs: n_prime*kwargs['q']/kwargs['n']
q_ratio = lambda **kwargs: q_cor(**kwargs)/kwargs['q']

w_cor = lambda **kwargs: (n_prime/kwargs['n'])**3 * kwargs['w']
w_ratio = lambda **kwargs: w_cor(**kwargs)/kwargs['w']

fp = lambda **kwargs: w_cor(**kwargs)/(kwargs['u']*kwargs['i'])
nu_g = lambda **kwargs:100*rho*g*(q_cor(**kwargs) / (3600*1000))*h_man_cor(**kwargs)/w_cor(**kwargs)

# %% Cálculo das incertezas
cols = [
    'h_man', 'h_man_cor', 'h_ratio', 
    'q_cor', 'q_ratio', 'w_cor',
    'w_ratio', 'fp', 'nu_g'
]
u_exp_df = pd.DataFrame(columns=cols)
calc_df = u_exp_df.copy()

for index in df.index.values:
    args = df.loc[index]
    args.index = ['q', 'p_suc', 'p_des', 'w', 'n', 'u', 'i']
    args = dict(args)

    grads = [
        partial_diff(h_man, args=args),
        partial_diff(h_man_cor, args=args),
        partial_diff(h_ratio, args=args),
        partial_diff(q_cor, args=args),
        partial_diff(q_ratio, args=args),
        partial_diff(w_cor, args=args),
        partial_diff(w_ratio, args=args),
        partial_diff(fp, args=args),
        partial_diff(nu_g, args=args),
    ]

    calc_values = [
        h_man(**args),
        h_man_cor(**args),
        h_ratio(**args),
        q_cor(**args),
        q_ratio(**args),
        w_cor(**args),
        w_ratio(**args),
        fp(**args),
        nu_g(**args)
    ]

    u_exp = np.array([uncertainty(grad, uncertainty_df.values) for grad in grads])
    u_exp_df.loc[index] = u_exp

    calc_df.loc[index] = calc_values

calc_df.fillna(0, inplace=True)
u_exp_df.fillna(0, inplace=True)
# %%
# Plots
# Altura manométrica
x = np.c_[calc_df['q_cor'], calc_df['q_cor']**2]
y = calc_df['h_man_cor']
reg = LinearRegression().fit(x, y)
preds_h = reg.predict(x)

traces = [
    px.scatter(
        data_frame=calc_df,
        x='q_cor',
        y='h_man_cor',
        error_x=u_exp_df['q_cor'],
        error_y=u_exp_df['h_man_cor'],
    ).data[0],
    px.line(
        x=calc_df['q_cor'],
        y=preds_h
    ).data[0]
]

layout = {
    'xaxis.title': 'Vazão Volumétrica [L/hr]',
    'yaxis.title': 'Altura Manométrica [mca]',
    'title': 'Curva Experimental: Altura Manomética',
}

fig = go.Figure(traces, layout=layout)
fig.show()
# %%
# Potência
x = np.c_[calc_df['q_cor'], calc_df['q_cor']**2]
y = calc_df['w_cor']
reg = LinearRegression().fit(x, y)
preds_w = reg.predict(x)

traces = [
    px.scatter(
        data_frame=calc_df,
        x='q_cor',
        y='w_cor',
        error_x=u_exp_df['q_cor'],
        error_y=u_exp_df['w_cor'],
    ).data[0],
    px.line(
        x=calc_df['q_cor'],
        y=preds_w
    ).data[0]
]

layout = {
    'xaxis.title': 'Vazão Volumétrica [L/hr]',
    'yaxis.title': 'Potência [W]',
    'title': 'Curva Experimental: Potência Demandada',
}

fig = go.Figure(traces, layout=layout)
fig.show()

# %%
# Rendimento global
x = np.c_[calc_df['q_cor'], calc_df['q_cor']**2]
y = calc_df['nu_g']
reg = LinearRegression().fit(x, y)
preds_nu = reg.predict(x)

traces = [
    px.scatter(
        data_frame=calc_df,
        x='q_cor',
        y='nu_g',
        error_x=u_exp_df['q_cor'],
        error_y=u_exp_df['nu_g'],
    ).data[0],
    px.line(
        x=calc_df['q_cor'],
        y=preds_nu
    ).data[0]
]

layout = {
    'xaxis.title': 'Vazão Volumétrica [L/hr]',
    'yaxis.title': 'Eficiência [%]',
    'title': 'Curva Experimental: Eficiência Global',
}

fig = go.Figure(traces, layout=layout)
fig.show()
# %%
# Potência por altura manométrica
fig = make_subplots(specs=[[{'secondary_y': True}]])

fig.add_trace(
    go.Scatter(
        x=calc_df['q_cor'],
        y=preds_h,
        name='Altura Manomética',
        error_x={
            'type': 'data',
            'array': u_exp_df['q_cor'].values,
        },
        error_y={
            'type': 'data',
            'array': u_exp_df['h_man_cor'].values
        }
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=calc_df['q_cor'],
        y=preds_w,
        name='Potência',
        error_x={
            'type': 'data',
            'array': u_exp_df['q_cor'].values,
        },
        error_y={
            'type': 'data',
            'array': u_exp_df['w_cor'].values
        }
    ),
    secondary_y=True,
)

layout = {
    'xaxis.title': 'Vazão Volumétrica [L/hr]',
    'title': 'Comparativo: Altura Manométrica x Potência',
}

fig.update_layout(layout)
fig.update_yaxes(title_text='Potência [W]', secondary_y=True)
fig.update_yaxes(title_text='Altura Manométrica [mca]', secondary_y=False)

fig.show()
# %%
# Potência por altura manométrica
fig = make_subplots(specs=[[{'secondary_y': True}]])

fig.add_trace(
    go.Scatter(
        x=calc_df['q_cor'],
        y=preds_h,
        name='Altura Manométrica',
        error_x={
            'type': 'data',
            'array': u_exp_df['q_cor'].values,
        },
        error_y={
            'type': 'data',
            'array': u_exp_df['h_man_cor'].values
        }
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=calc_df['q_cor'],
        y=preds_nu,
        name='Eficiência',
        error_x={
            'type': 'data',
            'array': u_exp_df['q_cor'].values,
        },
        error_y={
            'type': 'data',
            'array': u_exp_df['nu_g'].values
        }
    ),
    secondary_y=True,
)

layout = {
    'xaxis.title': 'Vazão Volumétrica [L/hr]',
    'title': 'Comparativo: Altura Manométrica x Eficiência',
}

fig.update_layout(layout)
fig.update_yaxes(title_text='Eficiêcia [%]', secondary_y=True)
fig.update_yaxes(title_text='Altura Manométrica [mca]', secondary_y=False)

fig.show()
# %%
