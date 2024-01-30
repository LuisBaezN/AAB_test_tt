import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/logs_exp_us.csv',sep='\t')

print(data.head(3))
data.info()
print('> Duplicated rows: ', data.duplicated().sum())
data.describe()

## Preprocesamiento
'''
Cambiamos los nombres de las columnas.
'''
data.columns = ['event', 'id', 'time', 'group']

'''
Cambiamos el formato de fecha y extraemos los nanosegundos.
'''
data['time'] = pd.to_datetime(data['time']).dt.microsecond * 1000 + pd.to_datetime(data['time']).dt.nanosecond

'''
Cambiamos la representación de los grupos a letras.
'''

data['group'] = data['group'].astype('str').replace({'246':'A1', '247':'A2', '248':'B'})

'''
Removemos datos duplicados.
'''
data = data[~data.duplicated()]

## EDA
'''
Veamos los eventos que hay dentro del experimento.
'''
print(data['event'].value_counts())
print('> Number of events:', data['event'].nunique())

'''
Vizualicemos ahora cuántos usuarios hay en el estudio.
'''

print('> Total users:', data['id'].nunique())

'''
Calculemos el promedio de acciones por usuario.
'''
events_user = data.groupby('id', as_index=False)['event'].count()

print('> Event mean per user:', events_user['event'].mean())

'''
Ahora grafiquemos esta actividad.
'''
events_user['event'].hist()
plt.show()

'''
Notamos que hay algunos usuarios que usan la plataforma con mucha más frecuencia que los demás usuarios.
'''

print(np.percentile(events_user['event'], [95, 97, 99]))

'''
El 5% de los usuarios hacen más de 89 acciones dentro de la plataforma.
'''
