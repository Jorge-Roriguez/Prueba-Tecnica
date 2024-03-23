# ------------------------- Librerias requeridas -------------------------

# Tratamiento de datos
import pandas as pd

# Modelos Candidatos 
from sklearn.linear_model import LogisticRegression      # Regresión logística
from sklearn.ensemble import RandomForestClassifier      # Clasificador bosques aleatorios
from xgboost import XGBClassifier                        # XGBoost 


# ------------------------- Exploración inicial de los datos -------------------------

# Importación de los datos
features = ('')
df_features = pd.read_csv(features)