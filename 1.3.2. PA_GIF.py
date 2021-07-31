import random
import numpy as np 
import matplotlib.pyplot as plt
import os
import imageio
#Función a optimizar (Rosenbrock)
def Funcion_cost(part):
    return (1 + part[0])**2 + 100*(part[1] - part[0]**2)**2

filenames = []

n_iteraciones = 100
error_obj = 1e-5
n_particulas = 50
#W es un parámetro inercial que afecta la propagación del movimiento de las partículas dada por el último valor de velocidad.
W = 0.5
#c1 y c2 son coeficientes de aceleración. El valor C₁ da el "peso" del mejor valor personal y C₂ el "peso" del mejor valor social.
c1 = 0.5
c2 = 0.9

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
it_cont = 0 
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))
x1 = np.linspace(-2.048,2.048,250)
y1 = np.linspace(-1,2.048,250)
X, Y = np.meshgrid(x1, y1)
Z=(1 + X)**2 + 100*(Y - X**2)**2


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
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax = fig.add_subplot(1, 2, 2)
    ax1.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
    ax1.scatter(pmejor_part[:,0],pmejor_part[:,1],pmejor_fitness_val,color = 'r', marker = '*')
    ax1.set_title('Enjambre de partículas 3D con {} iteraciones'.format(it_cont))
    ax1.view_init(40,235)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax.contour(X,Y,Z, 50, cmap = 'jet')
    ax.scatter(pmejor_part[:,0],pmejor_part[:,1],color = 'r', marker = 'o')
    ax.set_title('Enjambre de partículas 2D con {} iteraciones'.format(it_cont))
    filename = f'{it_cont}.png'
    filenames.append(filename)
    plt.savefig(filename)
    it_cont = it_cont + 1
    plt.clf()


with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)





