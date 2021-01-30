from os import supports_effective_ids
import pickle
import random
import statistics
import time
from multiprocessing import Pool, Value, cpu_count
from multiprocessing.context import ProcessError

import ctrnn
import line_location

simulation_seconds = 3
time_const = line_location.line_location.timestep
tasks = []

def fitness(genome,tasks):
    sender = ctrnn.CTRNN(genome)
    receiver = ctrnn.CTRNN(genome)
    fitnesses = []

    for sp,rp,goal in tasks:
        sim = line_location.line_location(senderPos=sp,receiverPos=rp, goal=goal)
        sender.reset()
        receiver.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderdx   = sim.motor(sender.eulerStep(sim.getState(True),time_const)[0])
            receiverdx = sim.motor(receiver.eulerStep(sim.getState(False),time_const)[0])

            sim.step(senderdx,receiverdx)

        fitness = sim.fitness()


        fitnesses.append(fitness)

            #print("{0} fitness {1}".format(net, fitness))

    return rank_reduce(fitnesses)


# Calculate a weighted sum of fitnesses, with stronger weighting to lower scores
def rank_reduce(fitnesses):
    s_fitnesses = sorted(fitnesses)
    return sum([elem/(i+1) for i,elem in enumerate(s_fitnesses)])

# Create pop_size random genomes
def initialise(pop_size):
    pop = []
    while len(pop)<pop_size:
        genome = ctrnn.Genome()
        pop.append({"fitness":None, "genome":genome})

    return pop

# Calculate fitness score for each member of the population
def assess(pop, pool):
    global tasks 
    goals = [0.55,0.65,0.75,0.85,0.95]
    tasks = [(random.uniform(0,0.3),random.uniform(0,0.3),goal) for goal in goals]
    pop = [(x,tasks) for x in pop]
    pop = pool.starmap(assess_item, pop)
    return sorted(pop, key = lambda i: i["fitness"], reverse=True) 

# Calculate the fitness of a single genome
def assess_item(item,tasks):
    item["fitness"] = fitness(item["genome"],tasks)
    return item

# Mutate each member of the population
def mutate(pop, pool):
    global tasks
    pop = [(x,tasks) for x in pop]
    pop = pool.starmap(mutate_item,pop)
            
    return pop

# Mutate a single genome, only keep the result if it's better than the parent
def mutate_item(item,tasks):
    child = item["genome"].copy()
    child.mutate()
    if fitness(child,tasks) > item["fitness"]:
        item["genome"] = child

    return item


# Replace 0 fitness members of the population with strong fitness members
def select(pop):
    size = len(pop)
    survivors = (9*size)//10
    pop = pop[:survivors]
    for i in range(size-survivors):
        pop.append(pop[i].copy())
    return sorted(pop[:size], key = lambda i: i["fitness"], reverse=True) 


def evolve(pop_size=100, max_gen=1, write_every=1, file=None):

    random.seed()

    with Pool(processes=cpu_count()) as pool:

        pop = initialise(pop_size)
        pop = assess(pop, pool)
        generation = 0
        best = pop[0]
        batch_start = time.time()
        while generation < max_gen:
            if file != None and generation % 20 == 0:
                print(f"Generation {generation}")
                print(f"Batch Time {time.time()-batch_start}")
                print(f"Mean fitness {statistics.mean([x['fitness'] for x in pop])}")
                batch_start = time.time()

            generation += 1
            pop = select(pop)
            pop = mutate(pop, pool)
            pop = assess(pop, pool)
            best = pop[0]
            # print(pop)
            if write_every and generation % write_every==0:
                write_fitness(pop, generation, file)

    return best

def write_fitness(pop, gen, file=None):
    
    fitness = []
    for p in pop:
        fitness.append(p["fitness"])

    line = "{:4d}: max:{:.3f}, min:{:.3f}, mean:{:.3f}".format(gen,max(fitness),min(fitness),statistics.mean(fitness))

    if file:
        file.write(line+"\n")
    else:
        print(line)

if __name__ == '__main__':
    start = time.time()
    path = f"logs/{int(start)}.txt"
    with open(path,'w') as f:
        print(f"Logs are in {path}")
        best = evolve(128,200,file=f)
        print(best["fitness"])
        with open("best_genome.pkl",'wb') as g:
            pickle.dump(best["genome"],g)
    print(f"Finished training in {time.time() - start} seconds")