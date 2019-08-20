    #------------------------------Dependencias------------------------------------#
from __future__ import division
import random
import math
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#--------------------------------Fitness---------------------------------------#
def funcao(x):
    xi,yi = int(x[0]),int(x[1])                                                #valores para otimizar
    hi=temp_h+yi                                                               #ponto+tamanho do template em y
    wi=temp_w+xi                                                               #ponto+tamanho do template em x
    img2=img[yi:hi,xi:wi]                                                      #recorta imagem original do tamanho do template
    fit = (np.square(img2 - template)).mean(axis=None)                         #calcula fitness -- MEAN SQUARED ERROR --

    if hi>img_h or wi>img_w:                                                   #RESTRICAO caso ponto+template fique fora da imagem, nesse caso o fitness recebe um valor alto
        fit=2*fit
    else:
        return fit
        
        
        

#Outros fitness
#fit = np.sum(np.absolute(img2 - template)) / (img2.shape[0]*img2.shape[1]) / 255             #calcula fitness -- DIFERENCA ENTRE PIXELS --
#fit, diff = compare_ssim(img2, template, full=True)                                          #calcula fitness -- DIFERENCA ESTRUTURAL --

#-----------------------------PARTICULA-PSO-----------------------------------#
class Particle:
    def __init__(self,x0):
        self.position_i=[]          #posicao da particula
        self.velocity_i=[]          #velocidade da particula
        self.pos_best_i=[]          #melhor posicao individual
        self.err_best_i=0           #melhor fitness individual 
        self.err_i=0                #fitness individual
        self.target=0               #melhor fitness

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))        #para cada particula adiciona uma velocidade random de -1 a 1
            self.position_i.append(x0[i])                       #adiciona posicao randomica gerada

    def evaluate(self,costFunc):                                #Avalia a fitness
        self.err_i=costFunc(self.position_i)

        if self.err_i < self.err_best_i or self.err_best_i==self.target:        #checa se a posicao atual é a melhor
            self.pos_best_i=self.position_i                                     #se for a melhor atualiza a melhor posicao e o erro
            self.err_best_i=self.err_i
            

    def update_velocity(self,pos_best_g):                                       #funcao para atualizar velocidades
        w=0.5                                                                   # constante do peso da inercia
        c1=2.05                                                                 # constante cognitiva
        c2=2.05                                                                 # constante social

        for i in range(0,num_dimensions):
            r1=random.random()                                                  #gera valores aleatorios entre 0 e 1 para r1 e r2
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social    #atualiza velocidade para cada individuo

    def update_position(self,bounds):                                           #funcao para atualizar posicao
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+int(self.velocity_i[i])       #atualiza posicao com base na velocidade
            if self.position_i[i]>bounds[i][1]:                                 #ajusta limite maximo
                self.position_i[i]=bounds[i][1]

            if self.position_i[i] < bounds[i][0]:                               #ajusta limite minimo
                self.position_i[i]=bounds[i][0]

#----------------------------------PSO---------------------------------------#
class PSO():
    def __init__(self,costFunc,dim,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=dim
        err_best_g=0                                                             #melhor erro por grupo
        self.pos_best_g=[]                                                       #melhor posicao por grupo
        self.best_history=[]
        swarm=[]                                                                 #cria a populacao
        for i in range(0,num_particles): 
            r = [round(random.randint(bounds[0][0],200)),random.randint(bounds[1][0],200)]      #gera valores aleatorios para populacao dentro dos limites
            swarm.append(Particle(r))                                                           #cria vetor

        i=0
        pop = []
        while i < maxiter:                                                      #loop de otimizacao
            print("Geracao: "+str(i))
            for j in range(0,num_particles):                                    #varre todos os individuos da populacao     
                swarm[j].evaluate(costFunc)                                     #avalia fitness
                if swarm[j].err_i < err_best_g or err_best_g == 0:              #determina se a particula atual e a melhor global
                    self.pos_best_g=list(swarm[j].position_i)                   #se for, atualiza posicao global
                    err_best_g=float(swarm[j].err_i)                            #se for, atualiza erro global
                    print("Ger: "+str(i)+" Part: "+str(j)+" Pos: "+str(self.pos_best_g)+" Erro: "+str(err_best_g)) #imprime a geracao e individuo com melhor res

            for j in range(0,num_particles):                                    #varre individuos atualizando posicao e velocidade
                swarm[j].update_velocity(self.pos_best_g)                       #atualiza velocidade
                swarm[j].update_position(bounds)                                #atualiza posicao
            self.best_history.append(err_best_g)
            i+=1

        print ('FINAL:')
        print (self.pos_best_g)                                                 #imprime melhor posicao
        print (err_best_g)                                                      #imprime melhor fitness

    def output(self):                                                           #funcao para retornar valores
        return list(self.pos_best_g),list(self.best_history)



#--------------------------FUNCAO PRINCIPAL------------------------------------#
if __name__ == "__PSO__":                                                       
    main()

img = cv2.imread('main.png')                                                   #le imagem geral
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                    #converte para tons de cinza
img_h, img_w = img.shape[0], img.shape[1]                                      #extrai shape

template = cv2.imread('template.png')                                          #le o template
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)                          #converte para tons de cinza
temp_h, temp_w = template.shape[0], template.shape[1]                          #extrai o shape

dim=4                                                                          #define numero de dimensoes do problema
bounds=[(0.5,1.5),(0,img_w),(0,img_h),(0,360)]                                                  #define restricoes do espaco de busca
p=PSO(funcao,dim,bounds,num_particles=200,maxiter=100)                         #roda o PSO

pos_best_g,best_history = p.output()                                           #recebe valores de saida
xi,yi = int(pos_best_g[0]),int(pos_best_g[1])                                  #recebe valores da melhor fitness
hi=temp_h+yi                                                                   #define tamanho do template
wi=temp_w+xi
img2=img[yi:hi,xi:wi]                                                          
cv2.rectangle(img, (xi, yi), (wi, hi), (0, 0, 255), 2)                         #desenha retangulo na posicao encontrada
cv2.imshow("Image", img)                                                       #mostra imagem
cv2.waitKey(0)

#--------------------------GRAFICOS--------------------------------------------#
epoch = np.arange(1, len(best_history)+1, 1)
fig, ax = plt.subplots()
ax.plot(epoch, best_history)
ax.set(xlabel='epoch', ylabel='fitness',
       title='Epoch x Fitness')
ax.grid()
fig.savefig("fitness.png")
#plt.show()


xit = []
yit = []
fits = []
for xi in range(img_w-temp_w):
    for yi in range(img_h-temp_h):
        xit.append(xi)
        yit.append(yi)
        hit=temp_h+yit
        wit=temp_w+xit
        imgt=img[yit:hit,xit:wit] 
        fits.append((np.square(imgt - template)).mean(axis=None))
print(xit,yit,fits)
