# ------------------------- Librerias requeridas --------------------------------------------

# Tratamiento de datos
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

# Visualización de datos
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Selección de variables y medición del desempeño
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV 

# Modelos Candidatos 
from sklearn.linear_model import LogisticRegression      # Regresión logística
from sklearn.ensemble import RandomForestClassifier      # Clasificador bosques aleatorios
from xgboost import XGBClassifier                        # XGBoost 


# ------------------------- Exploración inicial de los datos -----------------------------------------------------------------

# Importación de los datos
features = ('https://raw.githubusercontent.com/Jorge-Roriguez/Prueba-Tecnica/main/Datos/varaibles_transaccionales_nequi.csv')
df_features = pd.read_csv(features)
df_features = df_features.drop('Unnamed: 0', axis = 1)
df_features.head(15)

# Información general de los datos 
df_features.info()
df_features.columns

# Verificación de Nulos y duplicados
df_features.isna().sum()           # No hay datos nulos
df_features.duplicated().sum()     # No hay datos duplicados

# Fechas_cruce
df_features['Fechas_cruce'].unique()     # los datos se encuentran para cada fin de mes de los años 2022 y 2023
meses_2022 = ['2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30',
              '2022-05-31', '2022-06-30', '2022-07-31', '2022-08-31',
              '2022-09-30', '2022-10-31', '2022-11-30', '2022-12-31']
meses_2023 = ['2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30',
              '2023-05-31', '2023-06-30', '2023-07-31', '2023-08-31',
              '2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31']

df_2022 = pd.DataFrame()
for i in range(len(meses_2022)):
    x = df_features[df_features['Fechas_cruce'] == meses_2022[i]]
    df_2022 = pd.concat([df_2022, x], axis = 0)
df_2022 # Datos de las personas del 2022

df_2023 = pd.DataFrame()
for i in range(len(meses_2023)):
    x = df_features[df_features['Fechas_cruce'] == meses_2023[i]]
    df_2023 = pd.concat([df_2023, x], axis = 0)   
df_2023 # Datos de las personas del 2023

# Gráfico de las personas que se encuentran en mora 
fig = make_subplots(rows = 1, cols = 1)
fig.add_trace(
    go.Histogram(x = df_features['target'], name = 'Estado de mora de los clientes', marker_color = 'cornflowerblue', xbins=dict(size=0.5,)),
    row = 1, col = 1
)
fig.update_layout(
    title_text = "Estado de los clientes con sus obligaciones",
    template = 'simple_white',
    xaxis = dict(
        ticktext = ['Clientes al día', 'Clientes en mora'],
        tickvals = [0, 1],
      )
)
fig.show();
print("Cantidad de cientes en mora: ", df_features['target'].sum(), 
      "\nLo que equivale a: ", round(df_features['target'].sum()/len(df_features) * 100, 2), 
      "% del total de los clientes")

# Veamos ahora los clientes en mora por los años 2022 y 2023
fig = make_subplots(rows = 1, cols = 2)
fig.add_trace(
    go.Histogram(x = df_2022['target'], name = 'Clientes 2022', marker_color = 'darkmagenta', xbins=dict(size=0.5,)),
    row = 1, col = 1
)
fig.add_trace(
    go.Histogram(x = df_2023['target'], name = 'Clientes 2023', marker_color = 'deeppink', xbins=dict(size=0.5,)),
    row = 1, col = 2
)
fig.update_layout(
    title_text = "Estado de los clientes con sus obligaciones",
    template = 'simple_white',
    xaxis1 = dict( # Para el primer gráfico
        ticktext = ['Cliente al día', 'Cliente en mora'],
        tickvals = [0, 1],
      ),
    xaxis2 = dict( # Para el segundo gráfico
        ticktext = ['Cliente al día', 'Cliente en mora'], 
        tickvals = [0, 1],
      ),
)
fig.show();
print('Porcentaje de clientes en mora 2022: ',round(df_2022['target'].sum()/len(df_2022) * 100, 2), '%',
      '\nPorcentaje de clientes en mora 2023: ',round(df_2023['target'].sum()/len(df_2023) * 100, 2), '%')

# Distribución de las transacciones de las personas en mora
base1 = df_features[df_features['target'] == 1]
base2 = df_features[df_features['target'] == 0]
fig = make_subplots(rows = 1, cols = 2)
fig.add_trace(
    go.Histogram(x = base1['TransactionValue_PSE'], name = 'Clientes en mora', marker_color = 'darkmagenta'),
    row = 1, col = 1
)
fig.add_trace(
    go.Histogram(x = base2['TransactionValue_PSE'], name = 'Clientes al día', marker_color = 'deeppink'),
    row = 1, col = 2
)
fig.update_layout(
    title_text = "Distribución de las transacciones de los clientes",
    template = 'simple_white')
fig.show();

# ------------------------- terminar de normalizar datos ----------------------------------------------

# Separamos variable objetivo y variables explicativas
y = df_features.target
x = df_features.loc[:, ~df_features.columns.isin(['Fechas_cruce', 'TransactionValue_PSE', 'target'])]
x1 = pd.DataFrame(df_features['TransactionValue_PSE'])

scaler = StandardScaler()
scaler.fit(x1)
x11 = scaler.transform(x1)

x['TransactionValue_PSE'] =  x11
x # Datos totalmente normalizados


# ------------------------- Selección de variables (Método Wrapper) -------------------------

# Modelos de clasificación binaria - Regresión logísitica, Random Forest y XGBoost
m_lr = LogisticRegression()  
m_rf = RandomForestClassifier()
m_xgb = XGBClassifier()

modelos = list([m_lr, m_rf, m_xgb])

# Función de selección de variables por RFE
def funcion_rfe(modelos,X,y, num_variables, paso):
  resultados = {}
  for modelo in modelos: 
    rfemodelo = RFE(modelo, n_features_to_select = num_variables, step = paso)
    fit = rfemodelo.fit(X,y)
    var_names = fit.get_feature_names_out()
    puntaje = fit.ranking_
    diccionario_importancia = {}
    nombre_modelo = modelo.__class__.__name__

    for i,j in zip(var_names,puntaje):
      diccionario_importancia[i] = j
      resultados[nombre_modelo] = diccionario_importancia
  
  return resultados

# Resultados selección
result = pd.DataFrame(funcion_rfe(modelos, x, y, 50, 1))
result.fillna('No incluida', inplace = True)
result.tail(40)

# Variables a seleccionar
var_names = ['normalized_col_trx3','normalized_col_trx10','normalized_col_trx14',
             'normalized_col_trx15','normalized_col_trx20','normalized_col_trx21',
             'normalized_col_trx25', 'normalized_col_trx38', 'normalized_col_trx39',
             'normalized_col_trx40', 'normalized_col_trx44','normalized_col_trx45',
             'normalized_col_trx48','normalized_col_trx52','normalized_col_trx54',
             'normalized_col_trx55','normalized_col_trx64','normalized_col_trx65',
             'normalized_col_trx67','normalized_col_trx74', 'normalized_col_trx78',
             'normalized_col_trx83', 'normalized_col_trx89','normalized_col_trx94',
             'normalized_col_trx96', 'normalized_cols3', 'normalized_cols15',
             'normalized_cols16','normalized_cols28', 'normalized_cols30','normalized_cols38',
             'normalized_cols39','normalized_cols43','normalized_cols44','normalized_cols49',
             'normalized_cols53', 'normalized_cols73','normalized_cols81', 'normalized_cols84',
             'normalized_cols89', 'normalized_cols90', 'normalized_col_trx19']

len(var_names) # Número de variables con significancia para los modelos

# Datos con variables selecccionadas
x_selec = x[var_names]


# ------------------------- Desempeño de los modelos -------------------------

# Función pra evaluar los modelos
def medir_modelos(modelos, scoring, X, y, cv):

    metric_modelos = pd.DataFrame()
    for modelo in modelos:
        scores = cross_val_score(modelo, X, y, scoring = scoring, cv = cv )
        pdscores = pd.DataFrame(scores)
        metric_modelos = pd.concat([metric_modelos,pdscores], axis = 1)
    
    metric_modelos.columns = ["logistic_r","rf_classifier","xgboost_classifier"]
    return metric_modelos

# Desempeño con todas las variables
accu_df = medir_modelos(modelos, 'accuracy', x, y, 5)
accu_df

# Desempeño con variables seleccionadas 
acc_df_sel = medir_modelos(modelos, 'accuracy', x_selec, y, 5)
acc_df_sel

# Distribución del desempeño
accu_df.plot(kind = 'box', title= 'Desempeño con todas las variables')
acc_df_sel.plot(kind = 'box', title= 'Desempeño con varibles seleccionadas')


# -------------------------------- Afinamiento de hiperparámetros xgboost -----------------------------------

# Parámetros
param_grid = [{'max_depth': [3,4,5,6,7], 'eta':[0.01, 0.09, 0.1, 0.2, 0.4],'subsample': [0,3,0.4,0.5,0.7]}]

research = RandomizedSearchCV(m_xgb, param_distributions = param_grid,
                              n_iter = 5, scoring='accuracy')
research.fit(x_selec, y)

resultados = research.cv_results_
research.best_params_
df_resultados = pd.DataFrame(resultados)
df_resultados[["params","mean_test_score"]].sort_values(by = "mean_test_score", ascending = False)

# Modelo con los mejores parámetros
xg_final = research.best_estimator_


# -------------------------------- Predicciones ---------------------------------------------------------------------------------------------

predictions = cross_val_predict(xg_final, x_selec, y, cv = 5)
df_pred = pd.DataFrame(predictions, columns = ['pred']) 

# Se añade las predicciones a la base de datos inicial
df_final = pd.concat([df_features, df_pred], axis = 1)

# Matriz de confusión para validación
conf_matrix = confusion_matrix(df_features['target'], df_final['pred'])
plt.figure(figsize = (8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel('Valores predichos')
plt.ylabel('Valores reales observados')
plt.title('Matriz de confusión')
plt.show()

# Probabilidades de los clientes de estar o no en mora
probabilidades = xg_final.predict_proba(x_selec)
df_prob = pd.DataFrame(probabilidades, columns = ['Prob al dia', 'Prob en mora'])
df_final = pd.concat([df_features, df_pred, df_prob], axis = 1)

# Clientes con mayores probabilidades de estar al día con sus obligaciones
df_final[['Prob al dia']].sort_values(by = 'Prob al dia', ascending= False).head(10)

# Clientes con una probabilidad mayor al 80% (Alta)
df_final[df_final['Prob al dia'] >= 0.8]
print("Clientes con probabilidad alta de adquirir un préstamo: ",
      round(df_final[df_final['Prob al dia'] >= 0.8]['Prob al dia'].count()/ len(df_final),2)*100,'%')

# Clientes con una probabilidad entre 80% y 70% (Media)
print("Clientes con probabilidad media de adquirir un préstamo: ",
      round(df_final[(df_final['Prob al dia'] >= 0.7) & (df_final['Prob al dia'] < 0.8)]['Prob al dia'].count()/ len(df_final),2)*100, '%')

# Clientes con una probabilidad menos de 70% (Baja)
print("Clientes con probabilidad baja de adquirir un préstamo: ",
      round(df_final[df_final['Prob al dia'] < 0.7]['Prob al dia'].count()/ len(df_final),2)*100,'%')

# Presentación de los resultados en un diagrama de tortas
datos0 = [3, 12, 85]
colores = ['cornflowerblue', 'deeppink', 'darkmagenta']
labels = ["Probabilidad Alta", "Probabilidad Media", "Probabilidad Baja"]
plt.pie(datos0, labels = labels, colors = colores,  autopct="%0.0f%%")
plt.title('Segmentación de clientes para adquirir un crédito')
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------------------