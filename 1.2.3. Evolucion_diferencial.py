## EVOLUCIÓN DIFERENCIAL FUNCIÓN DE ROSENBROCK
import numpy as np
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
import matplotlib.pyplot as plt
def obj_R(x): #Se define la función objetivo la cual se quiere optimizar
##Dentro de ella se realiza un condicional ya que a la función puede llegarle un arreglo de dos dimensiones o de una
    s=x.shape
    if len(s)>1:
        fitness=(1 + x[:,0])**2 + 100*(x[:,1] - x[:,0]**2)**2
    else:
        fitness=(1 + x[0])**2 + 100*(x[1] - x[0]**2)**2
    return fitness


# La función de mutación se encarga de sumar la multiplicación entre el factor de mutación y a diferencia entre dos candidatos, con un tercer candidatoo 
def F_mutacion(x, F):
    return x[0] + F * (x[1] - x[2])


# En esta función se verifica que la mutación esté dentro de los límites previamente establecidos
def Ver_lim(a_mutado, limites):
    lim_F_mutacion = [clip(a_mutado[i], limites[i, 0], limites[i, 1]) for i in range(len(limites))] #clip es una función que se encarga de reemplazar los valores menores al límite inferior por este valor, y los mayores al superior por dicho valor
    return lim_F_mutacion


# La función crossover realiza el cruce de datos en cada par de valores que entra a ella. Su funcionamiento
# se basa en un factor de cruce "cr" que se encarga de decidir que valor dejar y que valor cambiar.
# El if se encarga de dicha decisión, si el valor producido aleatoriamente es menor que cr se dejará el valor mutado de entrada
# de lo contrario se deja el que se encuentre en la variable objetivo
def crossover(a_mutado, p_objetivo, dims, cr):
    p = rand(dims)
    trial = [a_mutado[i] if p[i] < cr else p_objetivo[i] for i in range(dims)]
    return trial


def F_ED(pop_size, limites, iter, F, cr): #La función de la evolución diferencial se encarga de reunir todas las funciones en un orden para dar como resultado la optimización de la función
    # Primero, se inicializa una población de posibles soluciones, ésto se hace de manera aleatoria y teniendo en cuenta los límites
    pop = limites[:, 0] + (rand(pop_size, len(limites)) * (limites[:, 1] - limites[:, 0]))
    # Se evalua el desempeño de la población en la función
    obj_all = obj_R(pop)
    # Después, se encuentra el "individuo" que mejor desempeño tuvo, es decir, el número más pequeño 
    best_vector = pop[argmin(obj_all)]
    #También se almacena dicho valor
    best_obj = min(obj_all)
    prev_obj = best_obj
    vectores=vectores=np.empty([iter,2])
    objetivos=[]
    #El for se encarga de realizar las iteraciones solicitadas
    for i in range(iter):
        # Se realiza también iteraciones por cada individuo
        for j in range(pop_size):
            # Se escogen tres candidatos (a,b,c) de la población que sean diferentes al actual 
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # Se realiza la mutación
            a_mutado = F_mutacion([a, b, c], F)
            # Se verifica que los valores de la mutación estén dentro de los límites establecidos
            a_mutado = Ver_lim(a_mutado, limites)
            # Leugo se realiza el cruce
            trial = crossover(a_mutado, pop[j], len(limites), cr)
            trial=np.array(trial)
            # Se calcula la función objetivo con el indviduo actual de la población inicial
            obj_p_objetivo = obj_R(pop[j])
            # Ahora se calcula con el individuo mutado y cruzado
            obj_trial = obj_R(trial)
            # Se realiza una comparación para saber que individuo tuvo mejor resultado
            if obj_trial < obj_p_objetivo:
                # Si el individuo mutado y cruzado tuvo mejor resultado que el de la población inicial, se reemplaza el individuo inicial con el mutado 
                pop[j] = trial
                # Se guardan los valores en una variable para posteriormente encontrar el menos (mejor)
                obj_all[j] = obj_trial
        # Se encuentra el mejor desempeño
        best_obj = min(obj_all)
        # Se realiza un comparación entre el desempño después de los cambios y el inicial
        if best_obj < prev_obj:
            # De ser menor, se almacena el individuo que dió dicho valor
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # Se muestra el progreso por cada iteración
            print('Iteración: %d f(%s) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
        vectores[i,:]=best_vector
        objetivos.append(best_obj)
    solucion=[best_vector, best_obj]
    return vectores,objetivos,solucion


# Se define el tamaño de la población
pop_size = 20
# Se definen los límites
limites = asarray([(-2.048, 2.048), (-1.5, 2.048)])
# Número de iteraciones
iter = 100
# Se define factor de mutación
F = 0.5
# Se define factor de cruce
cr = 0.7

# Finalmente, se realiza la diferenciación evolutiva
vectores,objetivos,solucion = F_ED(pop_size, limites, iter, F, cr)
print('\nSolucion: f(%s) = %.5f' % (solucion[0],  solucion[1]))
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x1 = np.linspace(-2.048,2.048,250)
y1 = np.linspace(-1,2.048,250)
X, Y = np.meshgrid(x1, y1)
Z=(1 + X)**2 + 100*(Y - X**2)**2
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(vectores[:,0],vectores[:,1],objetivos,color = 'r', marker = '*', alpha = .4)
ax.set_title('Evolución diferencial 3D con {} iteraciones'.format(iter))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(vectores[:,0],vectores[:,1],color = 'r', marker = 'o')
ax.set_title('Evolución diferencial 2D con {} iteraciones'.format(iter))

plt.show()

#%%
##EVOLUCIÓN DIFERENCIAL FUNCIÓN GOLDSTEIN PRICE
## La evolución diferencial se realiza igual que con la anterior función, solo cambia la función objetivo
import numpy as np
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
import matplotlib.pyplot as plt

def obj_GP(x): ## FUNCIÓN GOLDSTEIN PRICE
    s=x.shape
    if len(s)>1:
        fact1a=(x[:,0] + x[:,1] + 1)**2
        fact1b=19 - (14*x[:,0]) + (3*(x[:,0]**2)) - (14*x[:,1]) + (6*x[:,0]*x[:,1]) + (3*(x[:,1]**2))
        fact1=1 + (fact1a*fact1b)
        fact2a=(2*x[:,0] - 3*x[:,1])**2
        fact2b=18 - 32*x[:,0] + 12*x[:,0]**2 + 48*x[:,1] - 36*x[:,0]*x[:,1] + 27*x[:,1]**2
        fact2=30 + fact2a*fact2b

    else:
        fact1a=(x[0] + x[1] + 1)**2
        fact1b=19 - (14*x[0]) + (3*(x[0]**2)) - (14*x[1]) + (6*x[0]*x[1]) + (3*(x[1]**2))
        fact1=1 + (fact1a*fact1b)
	
        fact2a=(2*x[0] - 3*x[1])**2
        fact2b=18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2
        fact2=30 + fact2a*fact2b
    return fact1*fact2

def F_mutacion(x, F):
    return x[0] + F * (x[1] - x[2])

def Ver_lim(a_mutado, limites):
    lim_F_mutacion = [clip(a_mutado[i], limites[i, 0], limites[i, 1]) for i in range(len(limites))]
    return lim_F_mutacion

def crossover(a_mutado, p_objetivo, dims, cr):
    p = rand(dims)
    trial = [a_mutado[i] if p[i] < cr else p_objetivo[i] for i in range(dims)]
    return trial


def F_ED(pop_size, limites, iter, F, cr): 
    
    pop = limites[:, 0] + (rand(pop_size, len(limites)) * (limites[:, 1] - limites[:, 0]))
    obj_all = obj_GP(pop)
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    vectores=vectores=np.empty([iter,2])
    objetivos=[]
    for i in range(iter):
        for j in range(pop_size):
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            a_mutado = F_mutacion([a, b, c], F)
            a_mutado = Ver_lim(a_mutado, limites)
            trial = crossover(a_mutado, pop[j], len(limites), cr)
            trial=np.array(trial)
            obj_p_objetivo = obj_GP(pop[j])
            obj_trial = obj_GP(trial)
            if obj_trial < obj_p_objetivo:
                pop[j] = trial
                obj_all[j] = obj_trial
        best_obj = min(obj_all)
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            print('Iteración: %d f(%s) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
        vectores[i,:]=best_vector
        objetivos.append(best_obj)
    solucion=[best_vector, best_obj]
    return vectores,objetivos,solucion


pop_size = 30
limites = asarray([(-2, 2), (-1.5, 2)])
iter = 100
F = 0.4
cr = 0.3

vectores,objetivos,solucion = F_ED(pop_size, limites, iter, F, cr)
print('\nSolucion: f(%s) = %.5f' % (solucion[0],  solucion[1]))
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x1 = np.linspace(-2,2,250)
y1 = np.linspace(-1,2,250)
X, Y = np.meshgrid(x1, y1)
Fact1a=(X + Y + 1)**2
Fact1b=19 - (14*X) + (3*(X**2)) - (14*Y) + (6*X*Y) + (3*(Y**2))
Fact1=1 + (Fact1a*Fact1b)
           
Fact2a=(2*X - 3*Y)**2
Fact2b=18 - 32*X + 12*X**2 + 48*Y - 36*X*Y + 27*Y**2
Fact2=30 + Fact2a*Fact2b
Z=Fact1*Fact2
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(vectores[:,0],vectores[:,1],objetivos,color = 'r', marker = '*', alpha = .4)
ax.set_title('Evolución diferencial 3D con {} iteraciones'.format(iter))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(vectores[:,0],vectores[:,1],color = 'r', marker = 'o')
ax.set_title('Evolución diferencial 2D con {} iteraciones'.format(iter))

plt.show()