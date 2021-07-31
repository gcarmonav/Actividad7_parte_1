## ENJAMBRE DE PARTICULAS FUNCIÓN ROSENBROCK

import random
import numpy as np 
import matplotlib.pyplot as plt
#Función a optimizar (Rosenbrock)
def Funcion_cost(part):
    return (1 + part[0])**2 + 100*(part[1] - part[0]**2)**2

n_iteraciones = 100
error_obj = 1e-5
n_particulas = 50
#W es un parámetro inercial que afecta la propagación del movimiento de las partículas dada por el último valor de velocidad.
W = 0.2
#c1 y c2 son coeficientes de aceleración. El valor C₁ da el "peso" del mejor valor personal y C₂ el "peso" del mejor valor social.
c1 = 0.3
c2 = 0.8

val_obj = 4
#Se crea el vector de particulas inicial
vec_part=np.empty([n_particulas,2])
x = np.random.uniform(-2.048,2.048,n_particulas) #Definición valores de x (x1)
y = np.random.uniform(-1,2.048,n_particulas)#Definición valores de y (x2)
#Se almacena las partículas iniciales en el vector previamente creado
vec_part[:,0],vec_part[:,1]=x,y
#pmejor_part se encarga de almacenar el mejor valor de cada partícula
pmejor_part = np.zeros((n_particulas,2))
#pmejor_fitness_val almacena el mejor valor de desempeño por partícula
pmejor_fitness_val = np.array([float('inf') for _ in range(n_particulas)])
#gmejor_fitness_val guarda el mejor valorr de desempeño global. Se inicializa con la variable 'inf' que significa infinito para realizar la comparación
gmejor_fitness_val = float('inf')
# gmejor_part se encarga de alamacenar la partícula que tuvo el mejor desempeño
gmejor_part = np.array([float('inf'), float('inf')])
#Se crea el vector de la velocidad de movimiento de las partículas
vel_vector = ([np.array([0, 0]) for _ in range(n_particulas)])
it_cont = 0 #Contador de las iteraciones
while it_cont < n_iteraciones: #El 'while' se encarga de ejecutar el codigo hata llegar a las iteraciones deseadas
    for i in range(n_particulas):
        #Primero, se calcula un candidato como valor "base" de la función de coste con una particula de la población inicial 
        fitness_cadidato = Funcion_cost(vec_part[i])
        #Después se realiza la comparación que permitirá evaluar el mejor valor que cada partícula ha adquirido
        #Sí el candidato calculado anteriormente es menor que el mejor calculado entonces éste se convertirá en el mejor calculado
        if(pmejor_fitness_val[i] > fitness_cadidato):
            pmejor_fitness_val[i] = fitness_cadidato
            pmejor_part[i] = vec_part[i] #Y se guarda el individuo que obtuvo dicho desempeño
        
        #También se realiza una comparación para saber el mejor valor global
        #Si el valor candidato es menor que el valor global actual y no es menor que el valor objetivo, entonces éste se convierte en el nuevo mejor valor global
        if(gmejor_fitness_val > fitness_cadidato and fitness_cadidato >= val_obj): 
            gmejor_fitness_val = fitness_cadidato
            gmejor_part = vec_part[i] #También se guarda el sujeto que obtuvo dicho resultado

    if(abs(gmejor_fitness_val - val_obj) < error_obj): #En caso de que se llegue a un error menor al solicitado se detienen las iteraciones
        break
    
    #Finalmente, se realiza el "avance" de las partículas calculando la nueva velocidad
    for i in range(n_particulas):
        #La velocidad de movimiento tiene influencia propia, de la posición más conocida y también de la posición más conocida globalmente
        new_velocity = (W*vel_vector[i]) + (c1*random.random()) * (pmejor_part[i] - vec_part[i]) + (c2*random.random()) * (gmejor_part-vec_part[i])
        new_part = new_velocity + vec_part[i] #Se suma la velocidad a cada sujeto
        vec_part[i] = new_part
    it_cont = it_cont + 1
    
print("Mejores valores  ", gmejor_part, "Con resultado de ", gmejor_fitness_val) #Por último, se muestra la partícula con mejor desempeño y el valor de ésta

get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x1 = np.linspace(-2.048,2.048,250)
y1 = np.linspace(-1,2.048,250)
X, Y = np.meshgrid(x1, y1)
Z=(1 + X)**2 + 100*(Y - X**2)**2
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(pmejor_part[:,0],pmejor_part[:,1],pmejor_fitness_val,color = 'r', marker = '*', alpha = .4)
ax.set_title('Enjambre de partículas 3D con {} generaciones'.format(n_iteraciones))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(pmejor_part[:,0],pmejor_part[:,1],color = 'r', marker = 'o')
ax.set_title('Enjambre de partículas 2D con {} iteraciones'.format(n_iteraciones))

plt.show()

#%%
## ENJAMBRE DE PARTICULAS FUNCIÓN GOLDSTEIN PRICE

import random
import numpy as np 
import matplotlib.pyplot as plt

def Funcion_costGP(part):
     fact1a=(part[0] + part[1] + 1)**2
     fact1b=19 - (14*part[0]) + (3*(part[0]**2)) - (14*part[1]) + (6*part[0]*part[1]) + (3*(part[1]**2))
     fact1=1 + (fact1a*fact1b)
           
     fact2a=(2*part[0] - 3*part[1])**2
     fact2b=18 - 32*part[0] + 12*part[0]**2 + 48*part[1] - 36*part[0]*part[1] + 27*part[1]**2
     fact2=30 + fact2a*fact2b
     return fact1*fact2

n_iteraciones = 200
error_obj = 1e-5
n_particulas = 20
W = 0.2
c1 = 0.3
c2 = 0.7
val_obj = 3.0
vec_part=np.empty([n_particulas,2])
x = np.random.uniform(-2,2,n_particulas) #Definición valores de x (x1)
y = np.random.uniform(-1.5,2,n_particulas)#Definición valores de y (x2)
#Se almacena las partículas iniciales en el vector previamente creado
vec_part[:,0],vec_part[:,1]=x,y
#pmejor_part se encarga de almacenar el mejor valor de cada partícula
pmejor_part = np.zeros((n_particulas,2))
#pmejor_fitness_val almacena el mejor valor de desempeño por partícula
pmejor_fitness_val = np.array([float('inf') for _ in range(n_particulas)])
#gmejor_fitness_val guarda el mejor valorr de desempeño global. Se inicializa con la variable 'inf' que significa infinito para realizar la comparación
gmejor_fitness_val = float('inf')
# gmejor_part se encarga de alamacenar la partícula que tuvo el mejor desempeño
gmejor_part = np.array([float('inf'), float('inf')])
#Se crea el vector de la velocidad de movimiento de las partículas
vel_vector = ([np.array([0, 0]) for _ in range(n_particulas)])
it_cont = 0 #Contador de las iteraciones
while it_cont < n_iteraciones: #El 'while' se encarga de ejecutar el codigo hata llegar a las iteraciones deseadas
    for i in range(n_particulas):
        #Primero, se calcula un candidato como valor "base" de la función de coste con una particula de la población inicial 
        fitness_cadidato = Funcion_costGP(vec_part[i])
        #Después se realiza la comparación que permitirá evaluar el mejor valor que cada partícula ha adquirido
        #Sí el candidato calculado anteriormente es menor que el mejor calculado entonces éste se convertirá en el mejor calculado
        if(pmejor_fitness_val[i] > fitness_cadidato):
            pmejor_fitness_val[i] = fitness_cadidato
            pmejor_part[i] = vec_part[i] #Y se guarda el individuo que obtuvo dicho desempeño
        
        #También se realiza una comparación para saber el mejor valor global
        #Si el valor candidato es menor que el valor global actual y no es menor que el valor objetivo, entonces éste se convierte en el nuevo mejor valor global
        if(gmejor_fitness_val > fitness_cadidato and fitness_cadidato >= val_obj): 
            gmejor_fitness_val = fitness_cadidato
            gmejor_part = vec_part[i] #También se guarda el sujeto que obtuvo dicho resultado

    if(abs(gmejor_fitness_val - val_obj) < error_obj): #En caso de que se llegue a un error menor al solicitado se detienen las iteraciones
        break
    
    #Finalmente, se realiza el "avance" de las partículas calculando la nueva velocidad
    for i in range(n_particulas):
        #La velocidad de movimiento tiene influencia propia, de la posición más conocida y también de la posición más conocida globalmente
        new_velocity = (W*vel_vector[i]) + (c1*random.random()) * (pmejor_part[i] - vec_part[i]) + (c2*random.random()) * (gmejor_part-vec_part[i])
        new_part = new_velocity + vec_part[i] #Se suma la velocidad a cada sujeto
        vec_part[i] = new_part
    it_cont = it_cont + 1

print("Mejores valores  ", gmejor_part, "Con resultado de ", gmejor_fitness_val)

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
ax.plot(pmejor_part[:,0],pmejor_part[:,1],pmejor_fitness_val,color = 'r', marker = '*', alpha = .4)
ax.set_title('Enjambre de partículas 3D con {} generaciones'.format(n_iteraciones))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(pmejor_part[:,0],pmejor_part[:,1],color = 'r', marker = 'o')
ax.set_title('Enjambre de partículas 2D con {} iteraciones'.format(n_iteraciones))

plt.show()