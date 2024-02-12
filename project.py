import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

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
Tenemos también la páigna de ofertas, la cual es la segunda más visitada, seguidas por la página del carro de compras y la pantalla de pago satisfactorio.
De estos eventos, solo los últimos dos pertenecen a una secuencia clara. 

Las posibles secuencias son las siguientes:

                    |---> Cart Screen   ---> Payment Screen
                    |           ^
                    |           |
    - Main Screen --|---> Offers Screen 
                    |           ^
                    |           |
                    |--->   Tutorial 

Veamos estos eventos, respecto a usuarios.
'''

funnel = data.groupby('event', as_index=False)['id'].nunique().sort_values(by='id', ascending=False).reset_index(drop=True)
print(funnel)

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
Ya habíamos observado que hay usuarios que tienen una actividad muy alta de la plataforma. Se calculó que al rededor de solo el 5% de 
usuarios hacían más de 89 acciones.

Del embudo sabemos que se pierden más usuarios desde la pantalla de inicio a la de ofertas o a la pantalla del carrito. Los porcentajes
son los siguientes:
'''

for i in range(len(funnel) - 1):
    print(f'> {(funnel['id'][i]/funnel['id'][0]*100):.2f}% of clients in : {funnel['event'][i]}')

for i in range(len(funnel) - 2):
    print(f'> {(1 - funnel['id'][i + 1]/funnel['id'][i])*100:.2f}% of clients lost from {funnel['event'][i]} to {funnel['event'][i + 1]}')

'''
Vemos que el 46.67% de los usuarios hace todo el viaje.
'''

## Results analisys

'''
Sabemos que el tamaño de los grupos es muy similar. Veamos de nuevo los datos:
'''

test_groups = {}

for i in groups:
    test_groups[i] = data[data['group'] == i]
    print('\n> Users in group {0}: {1}'.format(i, test_groups[i]['id'].nunique()))
    print('> Events in group {0}: {1}'.format(i, len(test_groups[i])))

'''
Eliminaremos los clientes atípicos en todos los grupos. Trataremos como clientes atípicos a aquellos que realizaron más de 89 acciones dentro
de la plataforma.

Identificamos a estos usuarios.
'''

for i in range(len(test_groups)):
    print('> Group {0}, Before: {1}'.format(groups[i], len(test_groups[groups[i]])))
    events_per_user = test_groups[groups[i]].groupby('id', as_index=False)['event'].count()['event']
    threshold = events_per_user <= 89
    outliers = test_groups[groups[i]].groupby('id', as_index=False)['event'].count()[~threshold]

    test_groups[groups[i]] = test_groups[groups[i]][~test_groups[groups[i]]['id'].isin(outliers['id'])]
    print('> Group {0}, After: {1}'.format(groups[i], len(test_groups[groups[i]])))

'''
Aplicado el filtro, el número de eventos es similar entre cada uno de los grupos.

Veamos la proporción de usuarios.
'''

for i in groups:
    print('\n> Users in group {0}: {1}'.format(i, test_groups[i]['id'].nunique()))

'''
La proporción de usuarios es sigue siendo similar, sin embargo, el grupo A1 sigue siendo el grupo que tiene menos usuarios.

Veamos si hay una diferencia estadísticamente significativa en la proporción de las muestras entre el grupo A1 Y A2.
'''

test_groups['A1'].isna().sum()

#len(test_groups[groups[0]]['id'].unique()) + len(test_groups[groups[1]]['id'].unique()) + len(test_groups[groups[2]]['id'].unique())
#len(np.unique(np.concatenate((np.concatenate((test_groups[groups[0]]['id'].unique(), test_groups[groups[1]]['id'].unique())), test_groups[groups[2]]['id'].unique()))))
