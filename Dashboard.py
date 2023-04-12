#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:31:00 2023

@author: javierlechuga
"""

import requests 
import datetime as dt
import yfinance as yf
import pandas as pd
import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import matplotlib.ticker as mtick
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask import Flask, render_template, request
import plotly.express as px



app = Flask(__name__)
server=app.server

#lIMPIEZA CETES
cete=pd.read_csv('Cetes.csv')
cete.rename(columns = {'S&P/BMV Government CETES Bond Index':'Cete'}, inplace = True)
cete['FECHA'] = pd.to_datetime(cete['FECHA'], format='%Y%m%d')
cete = cete[~(cete['FECHA'] <=  '2019-12-31')]
cete.set_index('FECHA',inplace=True)

# LIMPIEZA FONDOS
fondos=pd.read_csv('Fondos 1.csv')
fondos2=pd.read_csv('Fondos2.csv')
fondos = pd.concat([fondos, fondos2], ignore_index=True)
fondos = fondos[~fondos.index.duplicated(keep='first')]
fondos = fondos[['FECHA','INSTRUMENTO','EMISORA', 'SERIE','PRECIO SUCIO']]
fondos.rename(columns = {'PRECIO SUCIO':'PRECIOSUCIO'}, inplace = True)
fondos = fondos.replace(0, np.nan)

# LIMPIEZA FONDOS CON SERIES B,B1,B-1
series=['B','B-1','B1']
fondos_b= fondos[fondos['SERIE'].isin(series)]

# Removemos duplicados del DataFrame
fondos_b = fondos_b.groupby(['FECHA', 'EMISORA', 'INSTRUMENTO', 'SERIE']).mean().reset_index()

# Pivot del DataFrame
fondos_b=fondos_b.pivot(index='FECHA', columns=['EMISORA','INSTRUMENTO','SERIE'], values='PRECIOSUCIO')
fondos_b.columns=fondos_b.columns.droplevel(['SERIE','INSTRUMENTO'])
fondos_b.index = pd.to_datetime(fondos_b.index)

fondos_b=fondos_b.shift(-1)
empresas_repetidasb = [grupo for _, grupo in fondos_b.groupby(fondos_b.columns, axis=1) if len(grupo.columns) > 1]
for grupo in empresas_repetidasb:
    empresa = grupo.columns[0]
    union_empresa = grupo.ffill(axis=1).iloc[:, -1]  # Llenamos NAN datos y tomamos ultima columna 
    fondos_b = fondos_b.drop(columns=grupo.columns)
    fondos_b[empresa] = union_empresa



#Limpieza FONDOS ACTINVER SERIE B,B1,B-1
empresas_actinver=['ACTIREN','ACTIGOB','ACTIMED','ALTERNA','ACTICOB','ESCALA','ACTIPLU','ACTOTAL','ACTIVAR','MAYA','OPTIMO','ACTINMO','OPORT1','SNX','ACTICRE','ACTI500','ESFERA','SALUD','ROBOTIK','DINAMO','IMPULSA','PROTEGE','ACTVIDA','EVEREST','MAXIMO','ACTIG+','ACTIG+2','ACTDUAL','ALTERN','DIGITAL','ECOFUND','TEMATIK','OPORT1','ACT2025','ACT2030','ACT2035','ACT2040','ACT4560','SURASIA']
fondos_actinver=fondos[fondos['EMISORA'].isin(empresas_actinver)]
fondos_actinverb=fondos_actinver[fondos_actinver['SERIE'].isin(series)]

# Removemos duplicados
fondos_actinverb = fondos_actinverb.groupby(['FECHA', 'EMISORA', 'INSTRUMENTO', 'SERIE']).mean().reset_index()

fondos_actinverb=fondos_actinverb.pivot(index='FECHA', columns=['EMISORA','INSTRUMENTO','SERIE'], values='PRECIOSUCIO')
fondos_actinverb.columns=fondos_actinverb.columns.droplevel(['SERIE','INSTRUMENTO'])
fondos_actinverb.index = pd.to_datetime(fondos_b.index)
fondos_actinverb=fondos_actinverb.shift(-1)
empresas_repetidasactb = [grupo for _, grupo in fondos_actinverb.groupby(fondos_actinverb.columns, axis=1) if len(grupo.columns) > 1]

#UNIMOS COLUMNAS REPETIDAS Y REMOVEMOS LAS ORIGINALES
for grupo in empresas_repetidasactb:
    empresa = grupo.columns[0]
    union_empresa = grupo.ffill(axis=1).iloc[:, -1]  # Llenamos NAN datos y tomamos ultima columna 
    fondos_actinverb = fondos_actinverb.drop(columns=grupo.columns)
    fondos_actinverb[empresa] = union_empresa
    
#LIMPIEZA S&P500, AGG, ^MXX
start=dt.date(2020,1,2)
end=dt.date(2023,3,27)
idx_tickers=['^GSPC','AGG','^MXX','MXN=X']
idx_data=yf.download(idx_tickers,start,end)
idx_data.drop(["Open","High","Low","Volume","Close"], axis=1, inplace=True)
idx_data['AGG']=idx_data['Adj Close','AGG']*idx_data['Adj Close','MXN=X']
idx_data['GSPC']=idx_data['Adj Close','^GSPC']*idx_data['Adj Close','MXN=X']
idx_data['MXX']=idx_data['Adj Close','^MXX']
idx_data = idx_data.iloc[:,[4,5,6]].copy()
idx_data = pd.DataFrame([idx_data.AGG, idx_data.GSPC, idx_data.MXX,cete.Cete]).transpose()
#Llenamos nan valores con el valor anterior
idx_data = idx_data.fillna(method='ffill').loc[:'2023-03-27']


# Rendimientos S&P500, AGG, ^MXX y CETES
idx_datarend= idx_data.pct_change()

#RENDIMIENTO DE TODOS LOS FONDOS CON SERIES B, B1, B-1
fondos_rendb=fondos_b.pct_change()

#RENDIMIENTOS FONDOS ACTINVER SERIES B, B1, B-1
fondos_rendactb=fondos_actinverb.pct_change()

#CORRELACIÓN FONDOS SERIES B,B1,B-1 VS AGG,SP500,MXX Y CETES
#Concatinamos el idx_datarend y fondod_rendactb
fondos_bcorr=pd.concat([idx_datarend, fondos_rendb], axis=1, join='inner')
# Calculamos las correlaciones y las guardamos en un nuevo DataFrame
fondos_bcorr = fondos_bcorr.corr()
#Extraemos las correlaciones de las columnas idx_datarend con las columnas fondos_rendactb
fondos_bcorr = fondos_bcorr.loc[idx_datarend.columns, fondos_rendb.columns]
fondos_bcorr=fondos_bcorr.transpose()
fondos_bcorr = fondos_bcorr.dropna()
#ELEGIMOS LA CORRELACIÓN MAYOR PARA CADA ACTIVO DE LOS FONDOS  SERIE B,B1,B-1
correlacion_mayorb = fondos_bcorr.idxmax(axis=1)
correlacion_mayorb = pd.DataFrame({'Correlación Mayor': correlacion_mayorb.values,'Index': correlacion_mayorb.index})

#Creamos listas de correlación para AGG, GSPC, MXX, Cete
correlacion_mayorb = correlacion_mayorb.dropna()
agg_b = correlacion_mayorb.loc[correlacion_mayorb['Correlación Mayor'] == 'AGG', 'Index'].tolist()
gspc_b = correlacion_mayorb.loc[correlacion_mayorb['Correlación Mayor'] == 'GSPC', 'Index'].tolist()
mxx_b = correlacion_mayorb.loc[correlacion_mayorb['Correlación Mayor'] == 'MXX', 'Index'].tolist()
cete_b = correlacion_mayorb.loc[correlacion_mayorb['Correlación Mayor'] == 'Cete', 'Index'].tolist()

#Retornos de cada lista fondos externos
all_returns = pd.concat([idx_datarend, fondos_rendb], axis=1)
agg_returns = all_returns.loc[:, agg_b]
gspc_returns = all_returns.loc[:, gspc_b]
mxx_returns = all_returns.loc[:, mxx_b]
cete_returns = all_returns.loc[:, cete_b]

agg_total_return = agg_returns.sum(axis=0)
gspc_total_return = gspc_returns.sum(axis=0)
mxx_total_return = mxx_returns.sum(axis=0)
cete_total_return = cete_returns.sum(axis=0)

# Ordenamos los retornos de mayor a menor
agg_total_return = agg_total_return.sort_values(ascending=False)
gspc_total_return = gspc_total_return.sort_values(ascending=False)
mxx_total_return = mxx_total_return.sort_values(ascending=False)
cete_total_return = cete_total_return.sort_values(ascending=False)

#Listas top 10 y top 5 de fondos por retorno
top_5_agg_b = list(agg_total_return.head(5).index)
top_10_agg_b = list(agg_total_return.head(10).index)

top_5_gspc_b = list(gspc_total_return.head(5).index)
top_10_gspc_b = list(gspc_total_return.head(10).index)

top_5_mxx_b = list(mxx_total_return.head(5).index)
top_10_mxx_b = list(mxx_total_return.head(10).index)

top_5_cete_b = list(cete_total_return.head(5).index)
top_10_cete_b = list(cete_total_return.head(10).index)

#CORRELACIÓN FONDOS ACTINVER VS AGG,SP500,MXX Y CETES
#Concatinamos el idx_datarend y fondod_rendactb
fondos_actbcorr=pd.concat([idx_datarend, fondos_rendactb], axis=1, join='inner')
# Calculamos las correlaciones y las guardamos en un nuevo DataFrame
fondos_actbcorr = fondos_actbcorr.corr()
#Extraemos las correlaciones de las columnas idx_datarend con las columnas fondos_rendactb
fondos_actbcorr = fondos_actbcorr.loc[idx_datarend.columns, fondos_rendactb.columns]
fondos_actbcorr=fondos_actbcorr.transpose()
#ELEGIMOS LA CORRELACIÓN MAYOR PARA CADA ACTIVO DE LOS FONDOS ACTINVER SERIE B,B1,B-1
correlacion_mayoractb = fondos_actbcorr.idxmax(axis=1)
correlacion_mayoractb = pd.DataFrame({'Correlación Mayor': correlacion_mayoractb.values,'Index': correlacion_mayoractb.index})
#Creamos listas de empresas más correlacionadas para AGG, GSPC, MXX, Cete
agg_actb=['ESCALA','ACTICOB','ACTOTAL','ACTIPLU']
gspc_actb=['SNX','ACTI500','ROBOTIK','ESFERA','SALUD','DIGITAL','DINAMO','TEMATIK','EVEREST','ECOFUND',
'ACT2025','ACT2040','ACT4560','ACT2035']
mxx_actb=['ACTICRE','ACTIVAR','MAYA','OPTIMO','ACTINMO','OPORT1']
cete_actbb=['ACTIREN','ACTIGOB','ACTIMED','ALTERN','ACTIG+','ACTIG+2','ACTDUAL','MAXIMO','IMPULSA','PROTEGE','ACTVIDA']



# Inicializamos app
app = dash.Dash(__name__)
server = app.server

# Definimos colores
colors = {
    'background': '#F0F0F0',
    'text': '#24425C',
    'text_inverse': '#D3D3D3',
    'line': '#1E3A5F',
    'button': '#1E3A5F',
    'button_text': '#FFFFFF'
}
color_scale = px.colors.sequential.Blues_r
index_color = '#DAA520'
asset_colors = color_scale[1:]
color_scale_extern = px.colors.sequential.Plotly3



## Definimos layout
app.layout = html.Div(
    style={'backgroundColor': colors['background'], 'color': colors['text'], 'font-family': 'Helvetica Neue'},
    children=[        html.Div([            html.Iframe(src='https://www.actinver.com/o/the-example/images/logo-actinver.svg', style={'height': '50px', 'opacity': '1'}),            html.H1('Fondos', style={'font-family': 'Helvetica', 'color': colors['text'], 'margin-left': '20px', 'margin-top': '5px', 'margin-bottom': '5px', 'padding-left': '10px', 'border-left': '3px solid ' + colors['line']})
        ], style={'display': 'flex', 'align-items': 'center', 'padding': '20px 50px'}),
        html.Div([            html.Label('Seleccione rango de fechas: ', style={'font-weight': 'bold'}),            html.Div([                dcc.DatePickerRange(                    id='date-picker-range',                    min_date_allowed=datetime.datetime(2020, 1, 1),                    max_date_allowed=datetime.datetime.today(),                    start_date=datetime.datetime(2022, 1, 1),                    end_date=datetime.datetime.today(),                    style={'font-family': 'Helvetica Neue', 'width': '200px'}                ),                html.Div([                    html.Button('Submit', id='submit-button', style={'background-color': colors['button'], 'color': colors['button_text'], 'border': 'none', 'padding': '10px', 'cursor': 'pointer', 'display': 'inline-block', 'float': 'left'})
                ]),
                html.Label('Seleccione top activos externos a visualizar:', style={'font-weight': 'bold', 'margin-left': '10px'}),
                dcc.Dropdown(
                    id='top-dropdown',
                    options=[
                        {'label': 'Top 5', 'value': 5},
                        {'label': 'Top 10', 'value': 10},
                        {'label': 'Ninguno', 'value': 0}
                    ],
                    value=0,
                    style={'font-family': 'Helvetica Neue', 'width': '150px', 'margin-right': '10px'}
                ),
                html.Div([
                    html.Button('Submit', id='top-submit-button', style={'background-color': colors['button'], 'color': colors['button_text'], 'border': 'none', 'padding': '10px', 'cursor': 'pointer', 'display': 'inline-block', 'float': 'right'})
                ]),
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
        html.Div([dcc.Graph(id='graph-1')], style={'margin-bottom': '20px'}),
        html.Div([dcc.Graph(id='graph-2')], style={'margin-bottom': '20px'}),
        html.Div([dcc.Graph(id='graph-3')], style={'margin-bottom': '20px'}),
        html.Div([dcc.Graph(id='graph-4')], style={'margin-bottom': '20px'}),
    ]
)
@app.callback(
    [dash.dependencies.Output('graph-1', 'figure'),
     dash.dependencies.Output('graph-2', 'figure'),
     dash.dependencies.Output('graph-3', 'figure'),
     dash.dependencies.Output('graph-4', 'figure')],
    [dash.dependencies.Input('submit-button', 'n_clicks'),
     dash.dependencies.Input('top-submit-button', 'n_clicks')],
    [dash.dependencies.State('date-picker-range', 'start_date'),
     dash.dependencies.State('date-picker-range', 'end_date'),
     dash.dependencies.State('top-dropdown', 'value')]
)
def update_graphs(n_clicks, top_n_clicks, start_date, end_date, top):
    #Obtenemos la fecha deseada
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    idx_fecha=idx_data[start:end]
    fondos_fecha=fondos_actinverb[start:end]   
    idx_daily_change = idx_fecha.apply(lambda x: x / x.iloc[0] - 1)
    fondos_daily_change = fondos_fecha.apply(lambda x: x / x.iloc[0] - 1)
    total_returns_sorted = {}
    for asset in fondos_fecha.columns:
        # Obtenemos el primer y ultimo index dispoinle
        first_index = fondos_fecha[asset].first_valid_index()
        last_index = fondos_fecha[asset].last_valid_index()
        if first_index is not None and last_index is not None:
            total_returns_sorted[asset] = (fondos_fecha.loc[last_index, asset] / fondos_fecha.loc[first_index, asset]) - 1


    # Creamos las gráficas
    graphs = []
    for i, index in enumerate(['AGG', 'GSPC', 'MXX', 'Cete']):
        #Obtenemos los datos
        index_data = idx_daily_change[index]
        index_return = (idx_fecha[index].iloc[-1] / idx_fecha[index].iloc[0]) - 1
        if index == 'AGG':
            asset_data = fondos_daily_change[agg_actb]
            return_data = fondos_fecha[agg_actb]
        elif index == 'GSPC':
            asset_data = fondos_daily_change[gspc_actb]
            return_data = fondos_fecha[gspc_actb]
        elif index == 'MXX':
            asset_data = fondos_daily_change[mxx_actb]
            return_data = fondos_fecha[mxx_actb]
        elif index == 'Cete':
            asset_data = fondos_daily_change[cete_actbb]
            return_data = fondos_fecha[cete_actbb]
            
        #Seleccionamos los top assets
        if top == 5:
            top_assets = top_5_agg_b if index == 'AGG' else top_5_gspc_b if index == 'GSPC' else top_5_mxx_b if index == 'MXX' else top_5_cete_b
        elif top == 10:
            top_assets = top_10_agg_b if index == 'AGG' else top_10_gspc_b if index == 'GSPC' else top_10_mxx_b if index == 'MXX' else top_10_cete_b
        else:
            top_assets = []
        # Información fondos externos para la fecha seleccionada
        fondos_b_fecha = fondos_b.loc[start_date:end_date, top_assets] if len(top_assets) > 0 else pd.DataFrame()
        # Calculamos retornos diarios fondos externos
        fondos_b_daily_change = fondos_b_fecha.apply(lambda x: x / x.iloc[0] - 1)
        
        # Creamos figure
        fig = go.Figure()
        fig.add_shape(type="line", x0=index_data.index[0], y0=0, x1=index_data.index[-1], y1=0, line=dict(color="#24425C", width=1, dash="longdash"))
        fig.add_trace(go.Scatter(x=index_data.index, y=index_data, name=f"<b>{index}</b> ({index_return:.2%})",  line=dict(width=3, color=index_color)))
        
        
        

        total_returns = {}
        for asset in return_data.columns:
            # Obtenemos el primer y ultimo index dispoinle
            first_index = return_data[asset].first_valid_index()
            last_index = return_data[asset].last_valid_index()
            if first_index is not None and last_index is not None:
                #Calculamos el retorno total
                total_returns[asset] = (return_data.loc[last_index, asset] / return_data.loc[first_index, asset]) - 1

        
        # Ordenamos total_returns por valor
        total_returns_sorted = {k: v for k, v in sorted(total_returns.items(), key=lambda item: item[1], reverse=True)}
        
        # Añadimos leyenda de Fondos Actinver
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Fondos Actinver', showlegend=True, legendgroup="Fondos Actinver", marker=dict(color='rgba(255, 255, 255, 0)', size=1)))

        
        for i, (asset, total_return) in enumerate(total_returns_sorted.items()):
            color = index_color if asset == index else asset_colors[i % len(asset_colors)]
            # Calculamos el primer y ultimo valor de cada activo
            asset_values = return_data[asset].dropna()
            first_value = asset_values.iloc[0]
            last_value = asset_values.iloc[-1]
            total_return = (last_value / first_value) - 1
            fig.add_trace(go.Scatter(x=asset_values.index, y=asset_data[asset], name=f"{asset} ({total_return:.2%})", line=dict(color=color), legendgroup="Fondos Actinver"))
            
        if len(top_assets) > 0:
            # AÑadimos titulo de la leyenda de Fondos Externos
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Fondos Externos', showlegend=True, legendgroup="Fondos Externos", marker=dict(color='rgba(255, 255, 255, 0)', size=1)))
            external_fund_total_returns = {}
            for asset in fondos_b_daily_change.columns:
                asset_values = fondos_b_fecha[asset].dropna()
                if len(asset_values) == 0:
                    continue
                first_value = asset_values.iloc[0]
                last_value = asset_values.iloc[-1]
                external_fund_total_returns[asset] = (last_value / first_value) - 1
                
            sorted_external_funds = sorted(external_fund_total_returns.items(), key=lambda x: x[1], reverse=True)

            for asset, total_return in sorted_external_funds:
                color = color_scale_extern[top_assets.index(asset) % len(color_scale_extern)]
                fig.add_trace(go.Scatter(x=fondos_b_daily_change.index, y=fondos_b_daily_change[asset], name=f"{asset} ({total_return:.2%})*", line=dict(color=color, width=1.3), opacity=0.7, legendgroup="External Funds"))

            
        fig.update_layout(
                title=index + " vs fondos",
                yaxis_tickformat='%',
                plot_bgcolor='white',
                paper_bgcolor='#24425C',
                font=dict(
                    family='Lato',
                    color='#D3D3D3'
                    ),
                xaxis=dict(
                    tickfont=dict(
                        color='#D3D3D3'
                        ),
                    gridcolor='#7f7f7f',
                    gridwidth=1
                    ),
                yaxis=dict(
                    tickfont=dict(
                        color='#D3D3D3'
                        ),
                    side='right',
                    gridcolor='#7f7f7f',
                    gridwidth=1
                    ),
                legend=dict(
                    font=dict(
                        color='#D3D3D3'
                        )
                    )
                )
        graphs.append(fig)
    return graphs


    
    
    
