import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator


from datetime import datetime, timedelta
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
import re
from sklearn.metrics import accuracy_score
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
from pycaret.classification import get_config
import warnings
warnings.filterwarnings("ignore")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email ':['enriquemejiagamarra@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_demo',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 17 * * *',
)



def GetDataKaggle(download_path = "/opt/airflow/dags/data/",user='albertstats1988',token = 'e1c23ee858429bae18536e17c0f41495'):
  # Configurar las credenciales de Kaggle
  os.environ['KAGGLE_USERNAME'] = user
  os.environ['KAGGLE_KEY'] = token
  
  # Crear instancia de la API de Kaggle y autenticar
  api = KaggleApi()
  api.authenticate()
  
  # Aceptar las reglas de la competencia (intenta este paso primero)
  competition_name = 'playground-series-s4e6'
  # Descargar los archivos de la competencia
  api.competition_download_files(competition_name, path=download_path)
  
  # Descomprimir los archivos descargados
  import zipfile
  for item in os.listdir(download_path):
    if item.endswith('.zip'):
      zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
      zip_ref.extractall(download_path)
      zip_ref.close()
      print(f"Unzipped {item}") 

  path1 = download_path + "train.csv"
  path2 = download_path + "test.csv"
  df = pd.read_csv(path1)
  prueba =  pd.read_csv(path2)

  ct = ["Marital status","Daytime/evening attendance","Nacionality","Mother's occupation","Father's occupation",
        "Displaced","Educational special needs","Debtor","Tuition fees up to date","Gender","Scholarship holder","International",
        "Application mode","Application order"]
  for k in ct:
    df[k] = df[k].astype("O")
    prueba[k] = prueba[k].astype("O")

  return df, prueba 


def AUTOML_PyCaret_preprocess_data(ti):
  df, prueba  = ti.xcom_pull(task_ids='GetDataKaggle')
  def prueba_kr(x):
    if x<=0.10:
      return 1
    else:
      return 0
      
  def criterion_(df,columns):
    for k in columns:
      df[k] = df[k].map(prueba_kr)
    df["criterio"] = np.sum(df.get(columns),axis=1)
    df["criterio"] = df.apply(lambda row: 1 if row["criterio"]==3 else 0,axis = 1)
    return df
    
  def nombre_(x):
    return "C"+str(x)
    
  def indicadora(x):
    if x==True:
      return 1
    else:
      return 0
      
  def label_tg(x):
    if x=="Dropout":
      return 0
    elif x=="Enrolled":
      return 1
    else:
      return 2
      
  def label_tg_inv(x):
    if x==0:
      return "Dropout"
    elif x==1:
      return "Enrolled"
    else:
      return "Graduate"

  # Formatos
  formato = pd.DataFrame({'Variable': list(df.columns), 'Formato': df.dtypes })
  # Cuantitativas
  cuantitativas = list(formato.loc[formato["Formato"]!="object","Variable"])
  cuantitativas = [x for x in cuantitativas if x not in ["id","Target"]]
  # Categóricas
  categoricas = list(formato.loc[formato["Formato"]=="O","Variable"])
  categoricas = [x for x in categoricas if x not in ["id","Target"]]
  # Variables al cuadrado
  base_cuadrado = df.get(cuantitativas).copy()
  base_cuadrado["Target"] = df["Target"].copy()
  var_names2, pvalue1, pvalue2, pvalue3 = [], [], [], []
  for k in cuantitativas:
    base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2
    # Prueba de Kruskal sin logaritmo
    mue1 = base_cuadrado.loc[base_cuadrado["Target"]=="Graduate",k+"_2"].to_numpy()
    mue2 = base_cuadrado.loc[base_cuadrado["Target"]=="Dropout",k+"_2"].to_numpy()
    mue3 = base_cuadrado.loc[base_cuadrado["Target"]=="Enrolled",k+"_2"].to_numpy()

    p1 = stats.kruskal(mue1,mue2)[1]
    p2 = stats.kruskal(mue1,mue3)[1]
    p3 = stats.kruskal(mue2,mue3)[1]

    # Guardar p values y variables
    var_names2.append(k+"_2")
    pvalue1.append(np.round(p1,2))
    pvalue2.append(np.round(p2,2))
    pvalue3.append(np.round(p3,2))

  pcuadrado1 = pd.DataFrame({'Variable2':var_names2,'p value':pvalue1,'p value 2':pvalue2,'p value 3':pvalue3})
  pcuadrado1 = criterion_(pcuadrado1,["p value","p value 2","p value 3"])

  # Interacciones cuantitativas
  lista_inter = list(combinations(cuantitativas,2))
  base_interacciones = df.get(cuantitativas).copy()
  var_interaccion, pv1, pv2, pv3 = [], [], [], []
  base_interacciones["Target"] = df["Target"].copy()

  for k in lista_inter:
    base_interacciones[k[0]+"__"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]

    # Prueba de Kruskal
    mue1 = base_interacciones.loc[base_interacciones["Target"]=="Graduate",k[0]+"__"+k[1]].to_numpy()
    mue2 = base_interacciones.loc[base_interacciones["Target"]=="Dropout",k[0]+"__"+k[1]].to_numpy()
    mue3 = base_interacciones.loc[base_interacciones["Target"]=="Enrolled",k[0]+"__"+k[1]].to_numpy()

    p1 = stats.kruskal(mue1,mue2)[1]
    p2 = stats.kruskal(mue1,mue3)[1]
    p3 = stats.kruskal(mue2,mue3)[1]

    var_interaccion.append(k[0]+"__"+k[1])
    pv1.append(np.round(p1,2))
    pv2.append(np.round(p2,2))
    pv3.append(np.round(p3,2))

  pxy = pd.DataFrame({'Variable':var_interaccion,'p value':pv1,'p value 2':pv2,'p value 3':pv3})
  pxy = criterion_(pxy,["p value","p value 2","p value 3"])

  # Razones
  raz1 = [(x,y) for x in cuantitativas for y in cuantitativas]
  base_razones1 = df.get(cuantitativas).copy()
  base_razones1["Target"] = df["Target"].copy()
  
  var_nm, pval, pval2, pval3 = [], [], [], []
  for j in raz1:
    if j[0]!=j[1]:
      base_razones1[j[0]+"__coc__"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]]+0.01)
      # Prueba de Kruskal
      mue1 = base_razones1.loc[base_razones1["Target"]=="Graduate",j[0]+"__coc__"+j[1]].to_numpy()
      mue2 = base_razones1.loc[base_razones1["Target"]=="Dropout",j[0]+"__coc__"+j[1]].to_numpy()
      mue3 = base_razones1.loc[base_razones1["Target"]=="Enrolled",j[0]+"__coc__"+j[1]].to_numpy()
      p1 = stats.kruskal(mue1,mue2)[1]
      p2 = stats.kruskal(mue1,mue3)[1]
      p3 = stats.kruskal(mue2,mue3)[1]

      # Guardar valores
      var_nm.append(j[0]+"__coc__"+j[1])
      pval.append(np.round(p1,2))
      pval2.append(np.round(p2,2))
      pval3.append(np.round(p3,2))
  prazones = pd.DataFrame({'Variable':var_nm,'p value':pval,'p value 2':pval2, 'p value 3':pval3})
  prazones = criterion_(prazones,["p value","p value 2","p value 3"])

  # Interacciones categóricas
  cb = list(combinations(categoricas,2))
  p_value, modalidades, nombre_var = [], [], []
  
  base2 = df.get(categoricas).copy()
  for k in base2.columns:
    base2[k] = base2[k].map(nombre_)
  base2["Target"] = df["Target"].copy()
  for k in range(len(cb)):
    # Variable con interacción
    base2[cb[k][0]] = base2[cb[k][0]]
    base2[cb[k][1]] = base2[cb[k][1]]

    base2[cb[k][0]+"__"+cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]

    # Prueba chi cuadrado
    c1 = pd.DataFrame(pd.crosstab(base2["Target"],base2[cb[k][0]+"__"+cb[k][1]]))
    pv = stats.chi2_contingency(c1)[1]

    # Número de modalidades por categoría
    mod_ = len(base2[cb[k][0]+"__"+cb[k][1]].unique())

    # Guardar p value y modalidades
    nombre_var.append(cb[k][0]+"__"+cb[k][1])
    modalidades.append(mod_)
    p_value.append(pv)
  pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})

  seleccion1 = list(pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=15),"Variable"])
  sel1 = base2.get(seleccion1)
  
  contador = 0
  for k in sel1:
    if contador==0:
        lb1 = pd.get_dummies(sel1[k],drop_first=True)
        lb1.columns = [k + "_" + x for x in lb1.columns]
    else:
        lb2 = pd.get_dummies(sel1[k],drop_first=True)
        lb2.columns = [k + "_" + x for x in lb2.columns]
        lb1 = pd.concat([lb1,lb2],axis=1)
    contador = contador + 1
    
  for k in lb1.columns:
    lb1[k] = lb1[k].map(indicadora)
  lb1["Target"] = df["Target"].copy()

  # Interacción cuantitativa categórica
  cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]
  v1, v2, pvalores_min, pvalores_max  = [], [], [], []
  for j in cat_cuanti:
    k1 = j[0]
    k2 = j[1]
    g1 = pd.get_dummies(df[k2])
    lt1 = list(g1.columns)
    for k in lt1:
      g1[k] = g1[k] * df[k1]
    g1["Target"] = df["Target"].copy()

    pvalues_c = []
    for y in lt1:
      mue1 = g1.loc[g1["Target"]=="Graduate",y].to_numpy()
      mue2 = g1.loc[g1["Target"]=="Dropout",y].to_numpy()
      mue3 = g1.loc[g1["Target"]=="Enrolled",y].to_numpy()

      try:
        pval = stats.kruskal(mue1,mue2)[1]<=0.20
      except ValueError:
        pval = 0
      try:
        pval2 = stats.kruskal(mue1,mue3)[1]<=0.20
      except ValueError:
        pval2 = 0
      try:
        pval3 = stats.kruskal(mue2,mue3)[1]<=0.20
      except ValueError:
        pval3 = 0

      pvalues_c.append(np.round(np.sum(np.array([pval,pval2,pval3])),2))
    min_ = np.min(pvalues_c) # Se calcula el mínimo p value por categoría
    max_ = np.max(pvalues_c)
    v1.append(k1)
    v2.append(k2)
    pvalores_min.append(np.round(min_,2))
    pvalores_max.append(np.round(max_,2))
  pc2 = pd.DataFrame({'Cuantitativa':v1,'Categórica':v2,'p value':pvalores_min, 'p value max':pvalores_max})

  v1 = list(pc2.loc[(pc2["p value"]==3) & (pc2["p value max"]==3),"Cuantitativa"])
  v2 = list(pc2.loc[(pc2["p value"]==3) & (pc2["p value max"]==3),"Categórica"])
  for j in range(len(v1)):
    if j==0:
      g1 = pd.get_dummies(df[v2[j]],drop_first=True)
      lt1 = list(g1.columns)
      for k in lt1:
        g1[k] = g1[k] * df[v1[j]]
      g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
    else:
      g2 = pd.get_dummies(df[v2[j]],drop_first=True)
      lt1 = list(g2.columns)
      for k in lt1:
        g2[k] = g2[k] * df[v1[j]]
      g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
      g1 = pd.concat([g1,g2],axis=1)
  g1["Target"] = df["Target"].copy()
  
  # Selección variables al cuadrado
  print("Selección Cuadrado")
  var_cuad = list(pcuadrado1["Variable2"])
  base_modelo1 = base_cuadrado.get(var_cuad+["Target"])
  base_modelo1["Target"] = base_modelo1["Target"].map(label_tg)
  cov = list(base_modelo1.columns)
  cov = [x for x in cov if x not in ["Target"]]
  X1 = base_modelo1.get(cov)
  y1 = base_modelo1.get(["Target"])
  modelo1 = XGBClassifier()
  modelo1 = modelo1.fit(X1,y1)
  importancias = modelo1.feature_importances_
  imp1 = pd.DataFrame({'Variable':X1.columns,'Importancia':importancias})
  imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
  imp1 = imp1.sort_values(["Importancia"],ascending=False)
  imp1.index = range(imp1.shape[0])

  # selección Interacciones cuatitativas
  print("Selección Interacciones cuantitativas")
  var_int = list(pxy["Variable"])
  base_modelo2 = base_interacciones.get(var_int+["Target"])
  base_modelo2["Target"] = base_modelo2["Target"].map(label_tg)
  cov = list(base_modelo2.columns)
  cov = [x for x in cov if x not in ["Target"]]
  X2 = base_modelo2.get(cov)
  y2 = base_modelo2.get(["Target"])
  modelo2 = XGBClassifier()
  modelo2 = modelo2.fit(X2,y2)
  importancias = modelo2.feature_importances_
  imp2 = pd.DataFrame({'Variable':X2.columns,'Importancia':importancias})
  imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
  imp2 = imp2.sort_values(["Importancia"],ascending=False)
  imp2.index = range(imp2.shape[0])

  # selección razones
  print("Selección razones")
  var_raz = list(prazones["Variable"])
  base_modelo3 = base_razones1.get(var_raz+["Target"])
  base_modelo3["Target"] = base_modelo3["Target"].map(label_tg)
  cov = list(base_modelo3.columns)
  cov = [x for x in cov if x not in ["Target"]]
  X3 = base_modelo3.get(cov)
  y3 = base_modelo3.get(["Target"])
  modelo3 = XGBClassifier()
  modelo3 = modelo3.fit(X3,y3)
  importancias = modelo3.feature_importances_
  imp3 = pd.DataFrame({'Variable':X3.columns,'Importancia':importancias})
  imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
  imp3 = imp3.sort_values(["Importancia"],ascending=False)
  imp3.index = range(imp3.shape[0])

  # selección interacciones categóricas
  print("Selección Interacciones categóricas")
  lb1["Target"] = lb1["Target"].map(label_tg)
  cov = list(lb1.columns)
  cov = [x for x in cov if x not in ["Target"]]
  X4 = lb1.get(cov)
  y4 = lb1.get(["Target"])
  modelo4 = XGBClassifier()
  modelo4 = modelo4.fit(X4,y4)
  importancias = modelo4.feature_importances_
  imp4 = pd.DataFrame({'Variable':X4.columns,'Importancia':importancias})
  imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
  imp4 = imp4.sort_values(["Importancia"],ascending=False)
  imp4.index = range(imp4.shape[0])

  # selección interacciones cuantitativas y categóricas
  print("Selección interacciones cuantitativas y categóricas")
  g1["Target"] = g1["Target"].map(label_tg)
  cov = list(g1.columns)
  cov = [x for x in cov if x not in ["Target"]]
  X5 = g1.get(cov)
  y5 = g1.get(["Target"])
  modelo5 = XGBClassifier()
  modelo5 = modelo5.fit(X5,y5)
  importancias = modelo5.feature_importances_
  imp5 = pd.DataFrame({'Variable':X5.columns,'Importancia':importancias})
  imp5["Importancia"] = imp5["Importancia"] * 100 / np.sum(imp5["Importancia"])
  imp5 = imp5.sort_values(["Importancia"],ascending=False)
  imp5.index = range(imp5.shape[0])

  # variables más importantes
  c2 = list(imp1.iloc[0:3,0]) # Variables al cuadrado
  cxy = list(imp2.iloc[0:3,0]) # Interacciones cuantitativas
  razxy = list(imp3.iloc[0:3,0]) # Razones
  catxy = list(imp4.iloc[0:3,0]) # Interacciones categóricas
  cuactxy = list(imp5.iloc[0:3,0]) # Interacción cuantitativa y categórica

  # Preparación Datos
  D1 = df.get(cuantitativas).copy()
  D2 = df.get(categoricas).copy()
  for k in categoricas:
    D2[k] = D2[k].map(nombre_)
  D4 = D2.copy()
  # Variables al cuadrado (Activar D1)
  cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
  cuadrado = [x[0] for x in cuadrado]
  for k in cuadrado:
    D1[k+"_2"] = D1[k] ** 2
  # Interacciones cuantitativas (Activar D1)
  result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]
  for k in result:
    D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
  # Razones
  result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
  for k in result2:
    k2 = k[0]
    D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
  # Interacciones categóricas
  result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
  for k in result3:
    D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
  # Interacción cuantitativa vs categórica
  D5 = df.copy()
  result4 = [re.search(r'(.+?)_(.+?)_\d+', item).groups() for item in cuactxy]
  contador = 0
  for k in result4:
    col1, col2 = k[1], k[0] # categórica, cuantitativa
    if contador == 0:
      D51 = pd.get_dummies(D5[col1],drop_first=True)
      for j in D51.columns:
        D51[j] = D51[j] * D5[col2]
      D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
    else:
      D52 = pd.get_dummies(D5[col1],drop_first=True)
      for j in D52.columns:
        D52[j] = D52[j] * D5[col2]
      D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
      D51 = pd.concat([D51,D52],axis=1)
    contador = contador + 1
  # Base Final
  B1 = pd.concat([D1,D4],axis=1)
  base_modelo = pd.concat([B1,D51],axis=1)
  base_modelo["Target"] = df["Target"].copy()
  base_modelo["Target"] = base_modelo["Target"].map(label_tg)
  return base_modelo, cuantitativas, categoricas, cuadrado, result, result2, result3, result4


def AUTOML_PyCaret_train_tunning_predict(ti):
  base_modelo, cuantitativas, categoricas, cuadrado, result, result2, result3, result4 = ti.xcom_pull(task_ids='AUTOML_PyCaret_preprocess_data')
  df, prueba  = ti.xcom_pull(task_ids='GetDataKaggle')

  def prueba_kr(x):
    if x<=0.10:
      return 1
    else:
      return 0
      
  def criterion_(df,columns):
    for k in columns:
      df[k] = df[k].map(prueba_kr)
    df["criterio"] = np.sum(df.get(columns),axis=1)
    df["criterio"] = df.apply(lambda row: 1 if row["criterio"]==3 else 0,axis = 1)
    return df
    
  def nombre_(x):
    return "C"+str(x)
    
  def indicadora(x):
    if x==True:
      return 1
    else:
      return 0
      
  def label_tg(x):
    if x=="Dropout":
      return 0
    elif x=="Enrolled":
      return 1
    else:
      return 2
      
  def label_tg_inv(x):
    if x==0:
      return "Dropout"
    elif x==1:
      return "Enrolled"
    else:
      return "Graduate"

  formatos = pd.DataFrame(base_modelo.dtypes).reset_index()
  formatos.columns = ["Variable","Formato"]
  cuantitativas_bm = list(formatos.loc[formatos["Formato"]!="object",]["Variable"])
  categoricas_bm = list(formatos.loc[formatos["Formato"]=="object",]["Variable"])
  cuantitativas_bm = [x for x in cuantitativas_bm if x not in ["Target"]]
  categoricas_bm = [x for x in categoricas_bm if x not in ["Target"]]

  exp_clf101 = setup(data=base_modelo,target='Target',session_id=123,train_size=0.7,
  numeric_features = cuantitativas_bm,categorical_features = categoricas_bm)

  best_model = compare_models(include=['lightgbm', 'xgboost', 'lr'])

  if best_model.__class__.__name__ == 'LGBMClassifier':
    dt = create_model("lightgbm")
    param_grid_bayesian = {
    'n_estimators': [50,100,200],
    'max_depth': [3,5,7],
    'min_child_samples': [50,150,200]}
    # Perform Bayesian Search
    tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)

  if best_model.__class__.__name__ == 'XGBClassifier':
    dt = create_model("xgboost")
    param_grid_bayesian = {
    'n_estimators': [50,100,200],
    'max_depth': [3,5,7]}
    # Perform Bayesian Search
    tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)

  if best_model.__class__.__name__ == 'LogisticRegression':
    dt = create_model("lr")
    param_grid_bayesian = {
    'C': [x for x in np.arange(0.1,5.1,0.1)]}
    # Perform Bayesian Search
    tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
  
  # Modelo final y completo
  final_dt = finalize_model(tuned_dt)
  
  # Preparación datos prueba
  D1 = prueba.get(cuantitativas).copy()
  # Variables categóricas
  D2 = prueba.get(categoricas).copy()
  for k in categoricas:
    D2[k] = D2[k].map(nombre_)
  D4 = D2.copy()
  # Variables al cuadrado (Activar D1)
  for k in cuadrado:
    D1[k+"_2"] = D1[k] ** 2
  # Interacciones cuantitativas (Activar D1)
  for k in result:
    D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
  # Razones
  for k in result2:
    k2 = k[0]
    D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
    
  # Interacciones categóricas
  for k in result3:
    D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
  # Interacción cuantitativa vs categórica
  D5 = prueba.copy()
  contador = 0
  for k in result4:
    col1, col2 = k[1], k[0] # categórica, cuantitativa
    if contador == 0:
      D51 = pd.get_dummies(D5[col1],drop_first=True)
      for j in D51.columns:
        D51[j] = D51[j] * D5[col2]
      D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
    else:
      D52 = pd.get_dummies(D5[col1],drop_first=True)
      for j in D52.columns:
        D52[j] = D52[j] * D5[col2]
      D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
      D51 = pd.concat([D51,D52],axis=1)
    contador = contador + 1

  B1 = pd.concat([D1,D4],axis=1)
  base_modelo2 = pd.concat([B1,D51],axis=1)
  df_test = base_modelo2.copy()
  predictions = predict_model(final_dt, data=df_test)

  result_k = pd.DataFrame({'id': prueba["id"],'Target': predictions['prediction_label']})
  result_k["Target"] = result_k["Target"].map(label_tg_inv)
  file_path = '/opt/airflow/dags/submissions/submit_albert0.csv'
  result_k.to_csv(file_path, index=False,sep=",")
  return file_path

def Subbmit_kaggle(ti):
  file_path = ti.xcom_pull(task_ids='AUTOML_PyCaret_train_tunning_predict')
  def prueba_kr(x):
    if x<=0.10:
      return 1
    else:
      return 0
      
  def criterion_(df,columns):
    for k in columns:
      df[k] = df[k].map(prueba_kr)
    df["criterio"] = np.sum(df.get(columns),axis=1)
    df["criterio"] = df.apply(lambda row: 1 if row["criterio"]==3 else 0,axis = 1)
    return df
    
  def nombre_(x):
    return "C"+str(x)
    
  def indicadora(x):
    if x==True:
      return 1
    else:
      return 0
      
  def label_tg(x):
    if x=="Dropout":
      return 0
    elif x=="Enrolled":
      return 1
    else:
      return 2
      
  def label_tg_inv(x):
    if x==0:
      return "Dropout"
    elif x==1:
      return "Enrolled"
    else:
      return "Graduate"
  # Configurar las credenciales de Kaggle
  os.environ['KAGGLE_USERNAME'] = 'albertstats1988'
  os.environ['KAGGLE_KEY'] = 'e1c23ee858429bae18536e17c0f41495'

  
  # Crear instancia de la API de Kaggle y autenticar
  api = KaggleApi()
  api.authenticate()
  
  # Aceptar las reglas de la competencia (intenta este paso primero)
  competition_name = 'playground-series-s4e6'
  # Descargar los archivos de la competencia
  download_path = '/opt/airflow/dags/data/'
  api.competition_download_files(competition_name, path=download_path)
  
  # Descomprimir los archivos descargados
  import zipfile
  for item in os.listdir(download_path):
    if item.endswith('.zip'):
      zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
      zip_ref.extractall(download_path)
      zip_ref.close()
      print(f"Unzipped {item}") 
  api.competition_submit(file_name=file_path,
  message="First submission",competition="playground-series-s4e6")
  return True

GetDataKaggle_task = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    dag=dag,
)

AUTOML_PyCaret_preprocess_task = PythonOperator(
    task_id='AUTOML_PyCaret_preprocess_data',
    python_callable=AUTOML_PyCaret_preprocess_data,
    dag=dag,
)

AUTOML_PyCaret_model_task = PythonOperator(
    task_id='AUTOML_PyCaret_train_tunning_predict',
    python_callable=AUTOML_PyCaret_train_tunning_predict,
    dag=dag,
)

Subbmit_kaggle_task = PythonOperator(
    task_id='Subbmit_kaggle',
    python_callable=Subbmit_kaggle,
    dag=dag,
)


GetDataKaggle_task >> AUTOML_PyCaret_preprocess_task >> AUTOML_PyCaret_model_task >> Subbmit_kaggle_task
