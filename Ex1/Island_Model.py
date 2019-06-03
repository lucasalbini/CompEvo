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

#Distancia entre sedes
distances = [448,1021,589,445,2709,413,853,1001,3058,728,465,434,1162,554,746,962,2153,465,1027,1162,650,539,2745,1267,287,421,698]

#Custo de envio
costs = [7667,22509,9902,7010,64803,5969,14539,16760,72317,12407,7739,6645,19279,7570,15189,15068,42101,10619,29910,25711,11235,7110,49876,16910,3756,4699,6711]

#Remuneracao
remuneration = [10703,24398,14069,10626,110813,11248,22101,25729,127278,19716,12606,11759,31504,12188,21151,21164,53181,13854,27345,28820,22169,14300,86746,33548,7356,11754,18326]

#Demanda naquela rota
demand = [1701,1453,1072,975,384,1908,929,431,153,218,1245,547,304,5246,1152,1118,407,1804,1452,703,307,325,109,156,509,100,111]

#Numero max de caminhoes
max_trucks = 68

#Tempo de Viagem
route_time =[]
for d in distances:
    route_time = np.append(route_time, math.ceil(d/55.)+math.ceil(d/75.)+4)

#Numero maximo de viagens por mes
trips_by_month = []
for t in route_time:
    trips_by_month = np.append(trips_by_month, math.floor(720/float(t)))

#Numero de bits por rota
bits = [3,3,3,2,3,3,3,2,2,1,3,2,2,4,3,3,3,3,3,3,1,1,2,1,1,1,1]
#Total de bits
total_of_bits = 62
#Numero de rotas por origem
nr_of_routes = [5,5,3,4,3,4,3]


#Decodifica o gene transformando em um valor inteiro que significa numero de caminhos na rota
def decoder(chromossome):
  out = []
  j=0
  for i in range(0,len(bits)):
    out = np.append(out, int("".join([str(c) for c in chromossome[j:j+bits[i]]]),2))
    j += bits[i]
  return out

def generator(random, args):
  chromossome = np.random.randint(2, size=total_of_bits)
  return chromossome

def h3(trucks):
    j = 0
    error = 0
    for i in (nr_of_routes):
        route = list(trucks[j:i+j])
        j += i
        zeros = route.count(0)
        if (i - zeros == 1):
            error += 1
    return error

@inspyred.ec.evaluators.evaluator
def evaluator(candidates, args):
  trucks = np.array(decoder(candidates))
  total_of_trucks = np.sum(trucks)

  #Calcula o numero de viagem que cada caminhao vai fazer
  trips_by_trucks = trips_by_month * trucks
  
  #Calcula o numero de veiculos transportado por mes
  vehicles_by_month = []
  for i in range(0,27):
    if(trucks[i]*11*trips_by_month[i] > demand[i]):
      num = 11*int(demand[i]/11.)
    else:
      num = trucks[i]*11*trips_by_month[i]
    vehicles_by_month = np.append(vehicles_by_month, num)

  #Calcula o custo total e o custo total por veiculo
  total_cost = vehicles_by_month * np.array(costs)
  total_cost_per_vehicle = total_cost/11.

  #Calcula a remuneracao total e a remurancao por veiculo
  total_remuneration = np.array(vehicles_by_month * remuneration)
  total_remuneration_per_vehicle = total_remuneration/11.

  total_profit = np.array(total_remuneration_per_vehicle - total_cost_per_vehicle)
  fitness = 0
  fitness = np.sum(total_profit)

  difference = total_of_trucks - max_trucks

  x = difference/total_of_trucks

  print x
  
  penalty = 1
  if difference>0:
    penalty = 0.5


  fitness = (fitness / 12480537 )*x-h3(trucks)

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

    