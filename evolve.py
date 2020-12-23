import pickle
import random
import statistics
import time
from multiprocessing import Pool, Value, cpu_count
from multiprocessing.context import ProcessError

import ctrnn
import line_location

runs_per_net = 10
simulation_seconds = 2
time_const = line_location.line_location.timestep
tasks = ()

def fitness(genome,tasks):
    sender = ctrnn.CTRNN(genome)
    receiver = ctrnn.CTRNN(genome)
    fitnesses = []

    # for run in range(runs_per_net):
    for sp,rp,goal in tasks:
        sim = line_location.line_location(senderPos=sp,receiverPos=rp, goal=goal)
        sender.reset()
        receiver.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderdx   = sim.motor(sender.eulerStep(sim.getState(True),time_const)[0])
            receiverinputs = sim.getState(False)
            receiverinputs[0] = goal
            receiverdx = sim.motor(receiver.eulerStep(receiverinputs,time_const)[0])

            sim.step(senderdx,receiverdx)

        fitness = sim.fitness()


        fitnesses.append(fitness)

            #print("{0} fitness {1}".format(net, fitness))


    # The genome's fitness is its worst performance across all runs.
    # print(fitnesses,rank_reduce(fitnesses)) 
    return rank_reduce(fitnesses)

def rank_reduce(fitnesses):
    # print(fitnesses,sum([elem/(i+1) for i,elem in enumerate(sorted(fitnesses,reverse=True))]))
    s_fitnesses = sorted(fitnesses)
    return sum([elem/(i+1) for i,elem in enumerate(s_fitnesses)])

def initialise(pop_size):
    
    pop = []
    while len(pop)<pop_size:
        genome = ctrnn.Genome()
        pop.append({"fitness":None, "genome":genome})

    return pop

def assess(pop, pool):
    # You could pass in a list of trials here
    global tasks 
    tasks = [(random.uniform(0,0.3),random.uniform(0,0.3),goal) for goal in [0.55,0.65,0.75,0.85,0.95]]
    pop = [(x,tasks) for x in pop]
    pop = pool.starmap(assess_item, pop)
    return sorted(pop, key = lambda i: i["fitness"], reverse=True) 

def assess_item(item,tasks):
    item["fitness"] = fitness(item["genome"],tasks)
    return item

def mutate(pop, pool):
    global tasks
    pop = [(x,tasks) for x in pop]
    pop = pool.starmap(mutate_item,pop)
            
    return pop

def mutate_item(item,tasks):
    child = item["genome"].copy()
    child.mutate()
    if fitness(child,tasks) > item["fitness"]:
        item["genome"] = child

    return item


def expand(pop):
    size = len(pop)
    for idx, i in enumerate(pop):
        if i["fitness"] == 0:
            pop = pop[:idx]
            break
    while len(pop) < size:
        pop.extend(pop.copy())
    return sorted(pop[:size], key = lambda i: i["fitness"], reverse=True) 

def evolve(pop_size=100, max_gen=1, write_every=1, file=None):

    random.seed()

    mutation = 0.01

    with Pool(processes=cpu_count()) as pool:

        pop = initialise(pop_size)
        pop = assess(pop, pool)
        # This might be cursed???
        # Can be replaced with selection
        # pop = expand(pop)
        generation = 0
        best = pop[0]
        while generation < max_gen:
            if file != None and generation % 20 == 0:
                print(f"Generation {generation}")
            generation += 1
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

    # genome = pop[0]["genome"]
    # sender = ctrnn.CTRNN(genome)
    # receiver = ctrnn.CTRNN(genome)
    # sim = line_location.line_location(senderPos=0.3,receiverPos=0.1,goal=0.75)
    # sender.reset()
    # receiver.reset()
    # # Run the given simulation for up to num_steps time steps.
    # fitness = 0.0
    # while sim.t < simulation_seconds:
    #     senderdx = sim.motor(sender.eulerStep(sim.getState(True),time_const)[0])
    #     receiverdx = sim.motor(receiver.eulerStep(sim.getState(False),time_const)[0])
    #     sim.step(senderdx,receiverdx)
    # fitness = sim.fitness()
    # print("Final conditions:")
    # print("   Sender = {0:.4f}".format(sim.senderPos))
    # print(" Receiver = {0:.4f}".format(sim.receiverPos))
    # print("     Goal = {0:.4f}".format(sim.goal))
    # print("  Fitness = {0:.4f}".format(sim.fitness()))

    if file:
        file.write(line+"\n")
    else:
        print(line)

if __name__ == '__main__':
    start = time.time()
    with open("poglog.txt",'w') as f:
        best = evolve(128,1000,file=f)
        print(best["fitness"])
        with open("best_genome.pkl",'wb') as g:
            pickle.dump(best["genome"],g)
    print(f"Finished training in {time.time() - start} seconds")