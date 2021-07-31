#Gradiente Descendiente función Rosenbrock

#Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import random

##DEFINICIÓN DE FUNCIONES

#Función de Rosenbrock
def funcion_Rosenbrock(x1,x2):
    return (1 + x1)**2 + 100*(x2 - x1**2)**2

#Función del Grediente (primera derivada) de la funcieón de Rosenbrock
def Grad_Rosenbrock(x1,x2):
    g1 = -400*x1*x2 + 400*x1**3 + 2*x1 -2 #Derivada con respecto a x
    g2 = 200*x2 -200*x1**2 #Derivada con respecto a y
    return np.array([g1,g2])

#Función donde se calcula el gradiente descendiente
def Gradient_Descent(Grad,x,y, alpha = 0.00125, epsilon=0.0001, nMax = 10000 ):
    i = 0
    iter_x, iter_y = np.empty(0),np.empty(0) #Estas variables se van a encargar de guardar todos los valores que x1(x) y x2(y) irán tomando a lo largo del proceso
    error = 10 #Error inicial a partir del cual se realizará la comparación en el while
    X = np.array([x,y]) #Array que contiene las variables iniciales
    
    while np.linalg.norm(error) > epsilon and i < nMax: #Este while se encarga de ejecutar las iteraciones necesarias para que el error sea menos que "epsilon", también que no se pase del # máximos de iteraciones
        i +=1 # i irá aumentando de 1 en 1
        iter_x = np.append(iter_x,x)##
        iter_y = np.append(iter_y,y)## Indexación de los valores de x y y en los vectores  
        
        X_prev = X #Esta variable se crea con el fin de guardar el valor inicial para calcular posteriormente el error
        X = X - alpha * Grad(x,y) #Cálculo de aprendizaje
        error = X - X_prev #Cálculo error
        x,y = X[0], X[1] #Nuevos valores de x y y
    iter_count=i  #Variable que indica el # de iteraciones necesarias para que el error fuera menor que epsilon    
    print(X)
    return X, iter_x,iter_y, iter_count


x = np.linspace(-2.048,2.048,250) #Definición valores de x (x1)
y = np.linspace(-1,2.048,250)#Definición valores de y (x2)
root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Rosenbrock,random.choice(x),random.choice(y)) #Calculo del gradiente descendiente con entrada aleatoria
X, Y = np.meshgrid(x, y) #Esta línea de codigo nos permite crear una matri de coordenadas que nos servirá para graficar la función
Z = funcion_Rosenbrock(X, Y) #



anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1] ##Variables que definen la dirección de las flechas que indicarán el cambio de x,y con las iteraciones

get_ipython().run_line_magic('matplotlib', 'inline') #Esta línea de codigo permite mostrar las gráficas debajo de la linea ejecutada en caso de que se utilice un notebook
fig = plt.figure(figsize = (16,8)) #Creación de la figura que mostrará las graficas

#Gráfica 3D de la función Rosenbrock que muestra el recorrido del Gradiente Descendiente sobre ésta
ax = fig.add_subplot(1, 2, 1, projection='3d') 
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(iter_x,iter_y, funcion_Rosenbrock(iter_x,iter_y),color = 'r', marker = '*', alpha = .4)
ax.set_title('Gradiente Descendente 3D con {} iteraciones'.format(iter_count))
ax.view_init(40,235)
ax.set_xlabel('x')
ax.set_ylabel('y')


#Gráfica 2D de la parte específica de la función Rosenbrock que muestra el recorrido del Gradiente Descendiente sobre ésta
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(iter_x,iter_y,color = 'r', marker = 'o')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3) #Aquí se traza un campo de flechas que va indicando el cambio de x,y con las iteraciones
ax.set_title('Gradiente Descendente con {} iteraciones'.format(iter_count))

plt.show()

#%%
##Gradiente Descendiente funci�n Goldstein Price
#El codigo tiene la misma estructura que el de la funci�n Rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import random
def funcion_GoldsteinP(x1,x2): #Funci�n Goldstein Price
    fact1a=(x1 + x2 + 1)**2
    fact1b=19 - (14*x1) + (3*(x1**2)) - (14*x2) + (6*x1*x2) + (3*(x2**2))
    fact1=1 + (fact1a*fact1b)
	
    fact2a=(2*x1 - 3*x2)**2
    fact2b=18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2=30 + fact2a*fact2b
    return fact1*fact2

def Grad_GoldsteinP(x1,x2): #Funci�n del Grediente (primera derivada) de la funcieón Goldstein Price
    g1 = (12*x1**3)+((36*x1**2)*x2)-(24*x1**2)+(36*x1*(x2**2))-(48*x1*x2)-(12*x1)+(12*x2**3)+24-(24*x2**2)-(12*x2)
    g2 = (12*x1**3)+((36*x1**2)*x2)-(24*x1**2)+(36*x1*(x2**2))-(48*x1*x2)-(12*x1)+(12*x2**3)+24-(24*x2**2)-(12*x2)
    return np.array([g1,g2])

def Gradient_Descent(Grad,x,y, gamma = 0.0012, epsilon=0.0001, nMax = 10000 ):
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        
        X_prev = X
        X = X - gamma * Grad(x,y)
        error = X - X_prev
        x,y = X[0], X[1]
          
    print(X)
    return X, iter_x,iter_y, iter_count

x = np.linspace(-2,2,250)
y = np.linspace(-1.5,2,250)
X, Y = np.meshgrid(x, y)
Z = funcion_GoldsteinP(X, Y)

root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_GoldsteinP,random.choice(x),random.choice(y))

anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (16,8))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(iter_x,iter_y, funcion_GoldsteinP(iter_x,iter_y),color = 'r', marker = '*', alpha = .4)

ax.view_init(25,190) ##VISUALIZACI�N GRAFICA
ax.set_xlabel('x')
ax.set_ylabel('y')


ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))


plt.show()