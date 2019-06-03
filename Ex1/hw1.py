from time import time
import time
from random import Random
import numpy as np
import inspyred
from inspyred import ec
from inspyred.ec import terminators
import pylab
import math
import multiprocessing
import random


#Dados do problema
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Distancia entre sedes
distancia = [448,1021,589,445,2709,413,853,1001,3058,728,465,434,1162,554,746,962,2153,465,1027,1162,650,539,2745,1267,287,421,698]

#Custo de envio por caminhao
custo = [7667,22509,9902,7010,64803,5969,14539,16760,72317,12407,7739,6645,19279,7570,15189,15068,42101,10619,29910,25711,11235,7110,49876,16910,3756,4699,6711]

#Remuneracao por viagem
remuneracao = [10703,24398,14069,10626,110813,11248,22101,25729,127278,19716,12606,11759,31504,12188,21151,21164,53181,13854,27345,28820,22169,14300,86746,33548,7356,11754,18326]

#Demanda naquela rota
demanda = [1701,1453,1072,975,384,1908,929,431,153,218,1245,547,304,5246,1152,1118,407,1804,1452,703,307,325,109,156,509,100,111]

#Numero maximo de caminhoes
nr_max = 68

#Tempo de viagem por rota -- (d/55)+(d/75)+4
tempo_viagem = [19,37,23,19,91,18,32,37,101,28,20,18,42,23,28,35,73,20,37,42,25,22,91,45,14,18,27]

#Numero maximo de viagens no mes -- 720/t
max_viagens = [37,19,31,37,7,40,22,19,7,25,36,40,17,31,25,20,9,36,19,17,28,32,7,16,51,40,26]

#Numero de bits por rota  -- (demanda/11)/max_viagens = nr caminhoes por mes  = 2^bits
bits = [3,3,3,2,3,3,3,2,2,1,3,2,2,4,3,3,3,3,3,3,1,1,2,1,1,1,1]

#Total de bits -- sum(bits)
total_bits = 62
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#candidato = [4, 0, 3, 2, 5, 4, 4, 2, 2, 1, 3, 1, 2, 15, 4, 5, 0, 4, 0, 1, 1, 1, 1, 1, 1, 0, 1]

#Restricoes
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Restricao 1 - quanto ao numero maximo de caminhoes
def h1(candidato):
	total_caminhoes = np.sum(candidato)
	diferenca = total_caminhoes-nr_max
	x = diferenca/13.
	return x

#Restricao 2 - quanto a demanda maxima
def h2(candidato):
	veiculos_mes = []
  	for i in range(0,27):
		if(candidato[i]*11*max_viagens[i] > demanda[i]):
			veiculos_mes = np.append(veiculos_mes, demanda[i])
		else:
			veiculos_mes = np.append(veiculos_mes, (candidato[i]*11*max_viagens[i]))
	return veiculos_mes

#Restricao 3 - quanto a regra de duas rotas
def h3(candidato):
	vetor = []
	value = 0
	erro = 0

	candidato = list(candidato.astype('int'))

	for i in (candidato):
		if candidato[i] != 0:
			vetor = np.append(vetor, 1)
		else:
			vetor = np.append(vetor, 0)

	a = 0,1,2,3,4
	b = 5,6,7,8,9
	c = 10,11,12
	d = 13,14,15,16
	e = 17,18,19
	f = 20,21,22,23
	g = 24,25,26

	conj = [a,b,c,d,e,f,g]
	for k in (conj):
		for j in (k):
			value = value+vetor[j]
			if value == 0 or value >= 2:
				erro = 0
			else:
				erro = erro+value
			value = 0

	return erro

#---------------------------------------------------------------------------------------------------------------------------------------------------------

#Implementacao do AG
#---------------------------------------------------------------------------------------------------------------------------------------------------------

#Decodifica o gene transformando em um valor inteiro que significa numero de caminhos na rota
def decoder(chromossome):
	out = []
	j=0
	for i in range(0,len(bits)):
		out = np.append(out, int("".join([str(c) for c in chromossome[j:j+bits[i]]]),2))
		j += bits[i]
	return out

def generator(random, args):
	chromossome = np.random.randint(2, size=total_bits)
  	return chromossome

@inspyred.ec.evaluators.evaluator
def evaluator(candidates, args):
	candidato = np.array(decoder(candidates))

	veiculos_mes = h2(candidato)
	#Calcula o custo total e o custo total por veiculo
	custo_total = veiculos_mes * np.array(custo)
	custo_total_por_veiculo= custo_total/11.

	#Calcula a remuneracao total e a remurancao por veiculo
	remuneracao_total = np.array(veiculos_mes * remuneracao)
	remuneracao_total_por_veiculo = remuneracao_total/11.

	lucro_total = np.array(remuneracao_total_por_veiculo - custo_total_por_veiculo)
	fitness = 0
	fitness = np.sum(lucro_total)

	fitness = (fitness / 12480537 )*h1(candidato)-h3(candidato)

	return fitness


def create_island(rand_seed, island_number, mp_migrator):
    rand = random.Random()
    rand.seed(rand_seed)
    ga = ec.GA(rand)
    ga.selector = inspyred.ec.selectors.tournament_selection
    ga.terminator = terminators.evaluation_termination
    #ga.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation]
    #ga.terminator = terminators.no_improvement_termination
    ga.observer = [inspyred.ec.observers.stats_observer, 
                   inspyred.ec.observers.plot_observer,
                   inspyred.ec.observers.file_observer]
    ga.migrator = mp_migrator
    final_pop = ga.evolve(evaluator=evaluator,
                          generator=generator,
                          statistics_file=open("stats_%d.csv" % island_number, "w"),
                          individuals_file=open("inds_%d.csv" % island_number, "w"),
                          crossover_rate=0.9,
                          mutation_rate=0.01,
                          pop_size=500,
                          max_evaluations=50000,
                          #max_generations = 15,
                          tournament_size=3,
                          num_selected=500,
                          evaluate_migrant=False)
    final_pop.sort(reverse=True)
    print (final_pop[0])

    A = decoder(final_pop[0].candidate)
    s = 0
    for i in A:
        s += i
    print(A)
    print (s)

if __name__ == "__main__":  
    cpus = 4
    mp_migrator = inspyred.ec.migrators.MultiprocessingMigrator(1)
    rand_seed = int(time.time())
    jobs = []
    for i in range(cpus):
        p = multiprocessing.Process(target=create_island, args=(rand_seed + i, i, mp_migrator))
        p.start()
        jobs.append(p)
    for j in jobs:
        j.join()

