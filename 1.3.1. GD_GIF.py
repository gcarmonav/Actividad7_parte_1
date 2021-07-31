import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
##DEFINICIÃN DE FUNCIONES

#FunciÃ³n de Rosenbrock
def funcion_Rosenbrock(x1,x2):
    return (1 + x1)**2 + 100*(x2 - x1**2)**2

#FunciÃ³n del Grediente (primera derivada) de la funcieÃ³n de Rosenbrock
def Grad_Rosenbrock(x1,x2):
    g1 = -400*x1*x2 + 400*x1**3 + 2*x1 -2 #Derivada con respecto a x
    g2 = 200*x2 -200*x1**2 #Derivada con respecto a y
    return np.array([g1,g2])

#FunciÃ³n donde se calcula el gradiente descendiente
def Gradient_Descent(Grad,x,y, alpha = 0.002, epsilon=0.0001, nMax = 10000 ):
    i = 0
    iter_x, iter_y = np.empty(0),np.empty(0) #Estas variables se van a encargar de guardar todos los valores que x1(x) y x2(y) irÃ¡n tomando a lo largo del proceso
    error = 10 #Error inicial a partir del cual se realizarÃ¡ la comparaciÃ³n en el while
    X = np.array([x,y]) #Array que contiene las variables iniciales
    while np.linalg.norm(error) > epsilon and i < nMax: #Este while se encarga de ejecutar las iteraciones necesarias para que el error sea menos que "epsilon", tambiÃ©n que no se pase del # mÃ¡ximos de iteraciones
        i +=1 # i irÃ¡ aumentando de 1 en 1
        iter_x = np.append(iter_x,x)##
        iter_y = np.append(iter_y,y)## IndexaciÃ³n de los valores de x y y en los vectores  
        
        X_prev = X #Esta variable se crea con el fin de guardar el valor inicial para calcular posteriormente el error
        X = X - alpha * Grad(x,y) #CÃ¡lculo de aprendizaje
        error = X - X_prev #CÃ¡lculo error
        x,y = X[0], X[1] #Nuevos valores de x y y
    iter_count=i  #Variable que indica el # de iteraciones necesarias para que el error fuera menor que epsilon    
    print(X)
    return X, iter_x,iter_y, iter_count

x = np.linspace(-2.048,2.048,250) #DefiniciÃ³n valores de x (x1)
y = np.linspace(-1,2.048,250)#DefiniciÃ³n valores de y (x2)
root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Rosenbrock,0.6,1.78) #Calculo del gradiente descendiente con entrada aleatoria
X, Y = np.meshgrid(x, y) #Esta lÃ­nea de codigo nos permite crear una matri de coordenadas que nos servirÃ¡ para graficar la funciÃ³n
Z = funcion_Rosenbrock(X, Y) #

anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]
filenames = []
get_ipython().run_line_magic('matplotlib', 'inline') 
fig = plt.figure(figsize = (16,8)) #CreaciÃ³n de la figura que mostrarÃ¡ las graficas
ax1 = fig.add_subplot(1, 2, 1, projection='3d') 
ax1.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
for i in range(80):
    ax1.scatter(iter_x[i],iter_y[i], funcion_Rosenbrock(iter_x[i],iter_y[i]),color = 'r', marker = '*', alpha = .4)
    ax1.set_title('Gradiente Descendente 3D con {} iteraciones'.format(i))
    ax1.view_init(40,235)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax.plot(iter_x[i],iter_y[i],color = 'r', marker = 'o')
    ax.set_title('Gradiente Descendente con {} iteraciones'.format(i))
    g=[iter_x[i],iter_y[i]]
    plt.suptitle('Cambio de x1 y x2 con la iteraciones {} '.format(g))
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)


with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)