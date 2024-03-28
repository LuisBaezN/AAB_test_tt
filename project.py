import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

def two_mean_z(data_1: object, data_2: object, column: str, D_0: float = 0) -> float:
    '''
    Calculate the z score for mean comparision of two large populations
    '''
    d_0 = D_0
    x_1 = data_1[column].mean()
    x_2 = data_2[column].mean()
    s_1 = data_1[column].std()
    s_2 = data_2[column].std()
    n_1 = len(data_1)
    n_2 = len(data_2)

    return ((x_1 - x_2) - d_0) / (np.sqrt(s_1**2/n_1 + s_2**2/n_2))

def test_hyp(z_score: float, rejection_point: float, test_type: str = 't'):
    '''
    Verify the hypothesis with z-scores. The tests available is the two tail test.
    '''
    if test_type == 't':
        if z_score < -rejection_point or z_score > rejection_point:
            print('> The null hipothesis is rejected.')
        else:
            print('> The null hipothesis is accepted.')


def name(dfs: list, group_names: list, row: str, category: str, group: str, alpha=0.01, simple: bool = True) -> None | float:
    '''
    Make a non-parametric test with the Mann Whintey method.

    Info:
    -----
    Structure -> `test_groups['A1'][test_groups['A1']['event'] == 'MainScreenAppear'].groupby('id', as_index=False)['event'].count()['event']`
    '''

    results = st.mannwhitneyu(
        dfs[group_names[0]][dfs[group_names[0]][row] == category].groupby(group, as_index=False)[row].count()[row],
        dfs[group_names[1]][dfs[group_names[1]][row] == category].groupby(group, as_index=False)[row].count()[row]
    )

    if simple:
        if results.pvalue > alpha:
            print('Both groups have no statistical difference')
        else:
            print('Both groups have statistical difference')

        print('p-value: ', results.pvalue)

    else:
        return results.pvalue


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
La proporción de usuarios sigue siendo similar, sin embargo, el grupo A1 permanece como el grupo que tiene menos usuarios.

Veamos si hay una diferencia estadísticamente significativa en la proporción de las muestras entre el grupo A1 Y A2. Para ello,
verificaremos primero si nuestros grupos tienen una distribución normal usando una gráfica de probabilidad. un test de Shapiro-Wilk y uno de Anderson-Darling.
'''
print('='*20, 'A1 Case:', '='*20)
st.probplot(test_groups['A1'].groupby('id', as_index=False)['event'].count()['event'], dist="norm", plot=plt)
plt.title('A1 probability distribution')
plt.show()

results = st.shapiro(test_groups['A1'].groupby('id', as_index=False)['event'].count()['event'])
print(results)
if results.pvalue > 0.01:
    print('Both groups are statisticaly equal')
else:
    print('Both groups are statisticaly different')
print('Shapiro\'s p-value: ', results.pvalue)

results = st.anderson(test_groups['A1'].groupby('id', as_index=False)['event'].count()['event'])
if results[0]:
    print('Both groups are statisticaly equal')
else:
    print('Both groups are statisticaly different')

print('='*20, 'A2 Case:', '='*20)

st.probplot(test_groups['A2'].groupby('id', as_index=False)['event'].count()['event'], dist="norm", plot=plt)
plt.title('A2 probability distribution')
plt.show()

results = st.shapiro(test_groups['A2'].groupby('id', as_index=False)['event'].count()['event'])
if results.pvalue > 0.01:
    print('Both groups are statisticaly equal')
else:
    print('Both groups are statisticaly different')
print('Shapiro\'s p-value: ', results.pvalue)

results = st.anderson(test_groups['A2'].groupby('id', as_index=False)['event'].count()['event'])
if results[0]:
    print('Both groups are statisticaly equal')
else:
    print('Both groups are statisticaly different')

'''
De acuerdo a los resultados, solo la prueba de Anderson indica que nuestros grupos sigen una distribución normal en ambos grupos. 

Dados estos resultados, usaremos pruebas una prueba no paramétrica (Mann Whitney) para determinar si ambos grupos son estadísticamente diferentes.
'''

results = st.mannwhitneyu(test_groups['A1'].groupby('id', as_index=False)['event'].count()['event'], test_groups['A2'].groupby('id', as_index=False)['event'].count()['event'])

print('p-value: ', results.pvalue)

'''
Nuestro p-value es muy alto, por lo que no hay una diferencia estadísticamente significativa entre estos dos grupos, en otras palabras, los factores que hacen visible una diferencia entre ambos grupos
no afectará los análisis posteriores ya que es practicamente despreciable.

Nos enfocaremos ahora en el evento más popular, tal evento corresponde a la página principal (´MainScreenAppear´) y verificaremos hay una diferencia estadísticamente significativa entre el grupo
A1 y A2.
'''

name(test_groups, ['A1', 'A2'], 'event', 'MainScreenAppear', 'id')

'''
Dado a que el p-value tiene un valor alto, podemos aseverar que ambos grupos no tienen una diferencia estadísticamente significativa.

Realizaremos la misma prueba para los eventos restantes.
'''

categories = ['OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial']

for i in categories:
    print('='*70)
    print(f'\n>> For {i}:\n\n')
    name(test_groups, ['A1', 'A2'], 'event', i, 'id')

'''
Para cada categoría no se encontró una diferencia significativa, por lo que los grupos quedaron divididos correctamente. Note que debido al número de eventos 
solo la última categoría tiene un p-value más bajo que las demás, pero muy alejado de nuestro umbral de decisión.

Compararemos ahora cada una de las categorías, con el grupo B. Comenzamos la comparación con el grupo A1
'''

categories = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial']

for i in categories:
    print('='*70)
    print(f'\n>> For {i}:\n\n')
    name(test_groups, ['A1', 'B'], 'event', i, 'id')

'''
Notamos que tampoco existe una diferencia significativa entre el grupo A1 y el B. De todas las categorias, solo la correspondiente a la página del carro de
compras presenta un p-value bajo (0.033).

Continuamos con la comparación, cambiando el grupo A1 por el A2.
'''

for i in categories:
    print('='*70)
    print(f'\n>> For {i}:\n\n')
    name(test_groups, ['A2', 'B'], 'event', i, 'id')


'''
Nuevamente no se encuentra una diferencia significativa entre ambos grupos, a excepción del bajo valor del p-value correspondiente a la página del carro de compras

Combinaremos el grupo A1 y el A2 para hacer una última prueba.
'''

test_groups['A'] = pd.concat([test_groups['A1'], test_groups['A2']])

for i in categories:
    print('='*70)
    print(f'\n>> For {i}:\n\n')
    name(test_groups, ['A', 'B'], 'event', i, 'id')

'''
Aunque los resultados obtenidos son muy parecidos a los resultados obtenidos con los grupos separados, los p-values se hicieron más pequeños en cada una de las 
categorias. Para el caso de la página del carrito de compras bajó a 0.0102.

De esta forma, podemos ver que el cambio de fuente en toda la página, aparentemente solo presenta un cambio de comportamiento en los clientes en la parte donde se
muestra el carrito de compras.

Según la corrección de Bonferroni, sabemos que:

FWER = 1 - (1 - alpha)^k

Donde:
alpha: 0.01 (nivel de significación)
k: 10 (número de pruebas)

Note que de las 15 pruebas hechas, tomamos en cuenta 10, ya que las últimas 5 toman los mismos datos de las primeras 10. Planteados los datos que usaremos
calculemos el probabilidad de obtener al menos un falso positivo:
'''

alpha = 0.05
k = 10

fwer = 1 - (1 - alpha)**k

print(f'> False positive probability: {fwer * 100}%')

'''
Tenemos cerca del 10% de obtener un 10% de obtener un falso positivo, dado que no obtuvimos ninguna prueba positiva con el actual nivel de significancia
esta probabilidad es consistente con los resultados de mnuestras pruebas.

Sabemos sin embargo, que hay una etapa que tiene dos resultados que se encuentran muy cerca de nuestro nivel de significancia, por lo que si subimos el 
nivel de alpha a 0.05 estas pruebas se considerarán como positivas pero con una probabilidad de al rededor del 40% de que al menos uno de los resultados 
sea un falso positivo 
'''

# FDR alternative if you have multiples positive results and you want to minimize the false positives and false negatives, also, if you have a limited sample size. The effect(s) of interest are not very large/consistent

groups = ['A1', 'A2']
p_values = []

for g in groups:
    for c in categories:
        p_values.append(name(test_groups, [g, 'B'], 'event', c, 'id', simple=False))

st.false_discovery_control(p_values)

df_tmp = pd.DataFrame(p_values).sort_values(by=0).reset_index(drop=True)
df_tmp * len(df_tmp) / pd.DataFrame([i + 1 for i in range(10)])

pd.DataFrame(p_values).sort_values(by=0)

#len(test_groups[groups[0]]['id'].unique()) + len(test_groups[groups[1]]['id'].unique()) + len(test_groups[groups[2]]['id'].unique())
#len(np.unique(np.concatenate((np.concatenate((test_groups[groups[0]]['id'].unique(), test_groups[groups[1]]['id'].unique())), test_groups[groups[2]]['id'].unique()))))

#### LEARN ABOUT IFDR!!

rej_point = np.abs(st.norm.ppf(1 - 0.005))
print('> Percent point', rej_point)

z = two_mean_z(test_groups['A1'].groupby('id', as_index=False)['event'].count(), test_groups['A1'].groupby('id', as_index=False)['event'].count(), 'event')

test_hyp(z, rej_point)
