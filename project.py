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
data.columns = ['event', 'id', 'time_stamp', 'group']

'''
Cambiamos el formato de la fecha y extraemos el dia, el mes y el año en una columna nueva.
'''
data['time_stamp'] = data['time_stamp'].map(lambda x: pd.Timestamp(x, unit='s', tz='US/Pacific'))

data['date'] = data['time_stamp'].dt.date

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

Revisemos como el tiempo en el que transcurre cada evento.
'''

data['time_stamp'].plot()
plt.grid()
plt.show()

'''
Vemos que prácticamente los primeros siete días no hay eventos. Esto es, del 25 de junio al 31 del mismo mes (7 días). En los siete días 
posteriores tenemos la mayoría de los datos (hasta el 7 de agosto).

Eliminaremos los datos que corresponden a los días de julio y verificaremos que tengamos los datos necesarios para las pruebas posteriores.
'''

data = data[data['time_stamp'] >= '2019-08-01']

'''
Revisemos de nuevo el número de usuarios y eventos.
'''

print(data['event'].value_counts())
print('> Number of events:', data['event'].nunique())
print('> Total users:', data['id'].nunique())

'''
El número de usuarios y eventos disminuyó homogéneamente. Además, de los 7550 usuarios, solo se perdieron 50. Verifiquemos si hay un desvalance
en los grupos.
'''

groups = ['A1', 'A2', 'B']

for i in groups:
    print('> Users in group {0}: {1}'.format(i, data[data['group'] == i]['id'].nunique()))

'''
Observamos que hay un ligero desvalance en el grupo A1 respecto a los otros dos.
'''

## Funnel

'''
De nuestra inspección anterior sabemos que el evento con más acciones fue la pantalla de inicio y la que menos acciones tuvo, fue la del tutorial.
Los demás eventos tienen un comportamiento similar y se relaciona al proceso de compra (producto -> carrito -> pago)

Veamos estos eventos, respecto a usuarios.
'''

print(data.groupby('event', as_index=False)['id'].nunique().sort_values(by='id', ascending=False))

'''
Vemos que la proporción se conserva.

Ahora veamos la actividad individual de cada usuario.
'''

users_actions = data.groupby('id', as_index=False)['event'].count().sort_values(by='event').reset_index(drop=True)

print(users_actions)

'''
Se observa que hay usuarios que solo hicieron una acción. Veamos esta proporción.
'''

users_actions['event'].plot()
plt.grid()
plt.show()

'''
Aquí podemos notar que la gran mayoría de los usuarios hizo solo una acción.

Tomando en cuenta que se registraron más de tres mil ventas, se espera que haya usuarios que aportaron en gran medida a esta métrica.
Ya habíamos observado que hay usuarios que tienen una actividad muy alta de la plataforma. Se calculó que al rededor del 5% de usuarios
hacían más de 89 acciones.
'''