import geppy as gep
from deap import creator, base, tools
import numpy as np
import random

import operator 
import math
#from math import sin, cos, exp
import datetime

import pandas as pd
import os

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

#Importando Datasets
treino = pd.read_csv("treino.csv")
teste = pd.read_csv("teste.csv")

#Checar os dados
print(treino.describe())
print(teste.describe())

#Adicionar variaveis em vetores numpy
sma = treino.sma.values
wma  = treino.wma.values
macd = treino.macd.values
rsi = treino.rsi.values
mom = treino.mom.values

Y = treino.classe2.values

#Imprimir vetores
print(sma)
print(wma)
print(macd)
print(rsi)
print(mom)
print(Y)


#FUNCAO DE DIVISAO PROTEGIDA
def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def seno(x):
    return math.sin(x)

#FUNCAO PULSO
def pulse(x):
    if x < -1:
        return 0
    if -1<=x<=1:
        return 1
    if x > 1:
        return 0

#FUNCAO RECT
def rect(x):
    return ((x+0.5)-(x-0.5))



#INPUT DATA
pset = gep.PrimitiveSet('Main', input_names=['sma','wma','macd','rsi','mom'])


#DEFININDO OS OPERADORES
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
#pset.add_function(rect, 1)
#pset.add_function(math.sin, 1)
#pset.add_function(math.cos, 1)
#pset.add_function(math.sin, 1)


pset.add_constant_terminal(1)
pset.add_constant_terminal(-1)
#pset.add_rnc_terminal()

#Criando individuo e populacao
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)





#CORE SETTINGS ----------------------------------------------------------------------------------------------------------------
h = 7          # head length
n_genes = 2    # number of genes in a chromosome
#r = 10         # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique
#------------------------------------------------------------------------------------------------------------------------------




#Definindo estrutura do gene
toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-1, b=1)   # each RNC is random integer within [-5, 5]
#toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)



#DEFININDO FUNCAO DE FITNESS --------------------------------------------------------------------------------------------------------------
from numba import jit

@jit
def evaluate(individual):
    """Evaluate the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    
    # Yp = np.array(list(map(func, X)))
    Yp = np.array(list(map(func, sma, wma, macd, rsi, mom))) 
    
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...
    #print (Y)
    #print (Yp)
    return np.mean((Y - Yp) ** 2),


@jit
def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual 
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func, sma, wma, macd, rsi, mom)))
    
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)   
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),

if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)
#-----------------------------------------------------------------------------------------------------------------------------------------

#Definindo operadores geneticos

toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

toolbox.pbs['cx_1p'] = 0.4
toolbox.pbs['mut_uniform'] = 0.1

# 2. Dc-specific operators
#toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
#toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
#toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)

# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
#toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
#toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

#ESTATISTICAS
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


#EVOLUTIONS---------------------------------------------------------------------------------------------------------------------------------------
# size of population and number of generations
n_pop = 250
n_gen = 80
champs = 3
pop = toolbox.population(n=n_pop) # 
hof = tools.HallOfFame(champs)   # only record the best three individuals ever found in all generations
#-------------------------------------------------------------------------------------------------------------------------------------------------

#TEMPO INICIO
startDT = datetime.datetime.now()
print (str(startDT))

#INICIO EVOLUCAO
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)

#TEMPO TOTAL
print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))

#HALL OF FAME -- MELHOR INDIVIDUO
print(hof[0])


#SIMPLIFICAR ARVORES------------------------------------------------------------------------------------------
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)
#sbest = compile(str(best_ind),error.txt,'eval')
#print(sbest)

print (symplified_best)

if enable_ls:
    symplified_best = best_ind.a * symplified_best + best_ind.b

key= '''
Given training examples of

    sma, wma, macd, rsi, mom

we trained a computer using Genetic Algorithms to predict the 

    Classe2

Our symbolic regression process found the following equation offers our best prediction:

'''
print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

from sympy import *
init_printing()
symplified_best
#-----------------------------------------------------------------------------------------------------------

# Renomear labels e exportar grafico
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
gep.export_expression_tree(best_ind, rename_labels, 'data/numerical_expression_tree.png')


# Mostrar imagem
from IPython.display import Image
Image(filename='data/numerical_expression_tree.png')



def CalculateBestModelOutput(sma, wma, macd, rsi, mom, model):
    return eval(model)

print(teste.describe())

print(type(str(symplified_best)))
pred = CalculateBestModelOutput(teste.sma, teste.wma, teste.macd, teste.rsi, teste.mom, str(symplified_best))

def wrapper(pred):
    pred = pred.values

    for i in range(len(pred)):
        if pred[i] < -0.5:
            pred[i] = -1 
            #print ("Reescrito")
        if pred[i] >= -0.5 and pred[i] <= 0.5:
            pred[i] = 0
            #print ("Reescrito")
        if pred[i] > 0.5:
            pred[i] = 1
            #print ("Reescrito")

    pred = pd.Series(pred)
    return pred

#pred = CalculateBestModelOutput(teste.sma, teste.wma, teste.macd, teste.rsi, teste.mom, str(best_ind))

wrapper(pred)

print(pred.describe())
print(pred.head())

#Validar MSE
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f" % mean_squared_error(teste.classe2, pred))
print("R2 score : %.2f" % r2_score(teste.classe2, pred))

#Numero de acertos
acertos = pd.Series.eq(teste.classe2,pred)
acertos = acertos.values
print("Acertos: "+ str(np.sum(acertos))+ " --- "+ (str((np.sum(acertos)/len(acertos)))+"%"))

#Plotar grafico
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [20, 5]
plotlen=200
pyplot.plot(pred.head(plotlen))       # predictions are in blue
pyplot.plot(teste.classe2.head(plotlen-2)) # actual values are in orange
pyplot.show()

#Histograma
pyplot.rcParams['figure.figsize'] = [10, 5]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)

numBins = 100
#ax.hist(holdout.PE.head(plotlen)-predPE.head(plotlen),numBins,color='green',alpha=0.8)
ax.hist(teste.classe2-pred,numBins,color='green',alpha=0.8)
pyplot.show()




