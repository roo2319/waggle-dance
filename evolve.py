from multiprocessing.context import ProcessError
import random
import statistics
import time
from multiprocessing import Value, cpu_count, Pool
import ctrnn
import line_location
        
runs_per_net = 10
simulation_seconds = 2
time_const = 0.01

def fitness(genome):
    sender = ctrnn.CTRNN(genome)
    receiver = ctrnn.CTRNN(genome)

    fitnesses = []
    for run in range(runs_per_net):
        sim = line_location.line_location(goal=0.75)
        sender.reset()
        receiver.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderdx = sim.motor(sender.eulerStep(sim.getState(True),time_const)[0])
            receiverdx = sim.motor(sender.eulerStep(sim.getState(False),time_const)[0])

            sim.step(senderdx,receiverdx)

        fitness = sim.fitness()

        # these guys aren't moving are they
        # if sim.receiverPos < 0.5:
        #     fitness = 0 

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
        genotype = ctrnn.Genome()
        pop.append({"fitness":None, "genotype":genotype})

    return pop

def assess(pop, pool):
    
    pop = pool.map(assess_item, pop)
    return sorted(pop, key = lambda i: i["fitness"], reverse=True) 

def assess_item(item):
    item["fitness"] = fitness(item["genotype"])
    return item

def mutate(pop, pool):
    
    pop = pool.map(mutate_item,pop)
            
    return pop

def mutate_item(item):
    child = item["genotype"].copy()
    child.mutate()
    if fitness(child) > fitness(item["genotype"]):
        item["genotype"] = child

    return item


def expand(pop):
    size = len(pop)
    for idx, i in enumerate(pop):
        if i["fitness"] == 0:
            pop = pop[:idx]
            break
    while len(pop) < size:
        pop.extend(pop)
    return sorted(pop[:size], key = lambda i: i["fitness"], reverse=True) 

def evolve(pop_size=100, max_gen=1, write_every=1, file=None):

    random.seed()

    mutation = 0.01

    with Pool(processes=cpu_count()) as pool:

        pop = initialise(pop_size)
        pop = assess(pop, pool)
        # This might be cursed???
        pop = expand(pop)
        generation = 0
        best = pop[0]
        while generation < max_gen:
            print(f"Generation {generation}")
            generation += 1
            pop = mutate(pop, pool)
            pop = assess(pop, pool)
            best = pop[0]
            if write_every and generation % write_every==0:
                write_fitness(pop, generation, file)

    return(generation,best)

def write_fitness(pop, gen, file=None):
    
    fitness = []
    for p in pop:
        fitness.append(p["fitness"])

    line = "{:4d}: max:{:.3f}, min:{:.3f}, mean:{:.3f}".format(gen,max(fitness),min(fitness),statistics.mean(fitness))

    genome = pop[0]["genotype"]
    sender = ctrnn.CTRNN(genome)
    receiver = ctrnn.CTRNN(genome)
    fitnesses = []
    sim = line_location.line_location(goal=0.75)
    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    fitness = 0.0
    while sim.t < simulation_seconds:
        senderdx = sim.motor(sender.eulerStep(sim.getState(True),time_const)[0])
        receiverdx = sim.motor(sender.eulerStep(sim.getState(False),time_const)[0])
        sim.step(senderdx,receiverdx)
    fitness = sim.fitness()
    print()
    print("Final conditions:")
    print("   Sender = {0:.4f}".format(sim.senderPos))
    print(" Receiver = {0:.4f}".format(sim.receiverPos))
    print("     Goal = {0:.4f}".format(sim.goal))
    print("  Fitness = {0:.4f}".format(sim.fitness()))
    print()

    if file:
        file.write(line+"\n")
    else:
        print(line)

if __name__ == '__main__':
    with open("poglog.txt",'w') as f:
        evolve(100,100,file=f)
    