import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

@st.cache
def get_data():
    return pd.read_csv('data.csv')

def train_model():
    data = get_data()
    X = data.drop('MEDV', axis = 1)
    y = data['MEDV']
    rf_regressor = RandomForestRegressor(n_estimators = 200, max_depth = 7, max_features= 3)
    rf_regressor.fit(X, y)
    return rf_regressor

data = get_data()

model = train_model()

st.title('Data App - Prevendo Valores de Imóveis')

st.markdown('Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imoveis de Boston.')

st.subheader('Selecionando apenas um pequeno conjunto de atributos')

defaultcols = ['RM', 'PTRATIO', 'LSTAT', 'MEDV']

cols = st.multiselect('Atributos', data.columns.tolist(), default = defaultcols)

st.dataframe(data[cols].head(10))

st.subheader('Distribuição de imóveis por preço')

faixa_valores = st.slider('Faixa de preço', float(data.MEDV.min()), 150., (10.0, 100.0))

dados = data[data['MEDV'].between(left = faixa_valores[0], right = faixa_valores[1])]

f = px.histogram(dados, x = 'MEDV', nbins = 100, title = 'Distribuição de Preços')
f.update_xaxes(title='MEDV')
f.update_yaxes(title='Total Imóveis')
st.plotly_chart(f)

st.sidebar.subheader('Defina os atributos do imóvel para predicao')

crim = st.sidebar.number_input('Taxa de criminalidade', value = data.CRIM.mean())
indus = st.sidebar.number_input('Proporção de Hectares de Negocio', value= data.CRIM.mean())
chas = st.sidebar.selectbox('Faz limite com o rio:?', ('Sim', 'Não'))

chas = 1 if chas == 'Sim' else 0

nox = st.sidebar.number_input('Concentração de óxido nitrico', value = data.NOX.mean())

rm = st.sidebar.number_input('Numero de Quartos', value = 1)

ptratio = st.sidebar.number_input('Indice de alunos para professores', value = data.PTRATIO.mean())

b = st.sidebar.number_input('Proporção de pessoas com descendencia afro-americana', value = data.B.mean())

lstat = st.sidebar.number_input('Porcentagem de status baixo', value = data.LSTAT.mean())

btn_predict = st.sidebar.button('Realizar predição')

if btn_predict:
    result = model.predict([[crim, indus, chas, nox, rm, ptratio, b, lstat]])
    st.subheader('O valor previsto para o imóvel é: ')
    result = 'US $ ' + str(round(result[0]*10,2))
    st.write(result)

#streamlit run app.py
