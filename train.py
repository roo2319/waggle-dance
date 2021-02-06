from os import supports_effective_ids
import pickle
import random
import statistics
import time
from multiprocessing import Pool, Value, cpu_count
from multiprocessing.context import ProcessError

import ctrnn
import evolve
import line_location

simulation_seconds = 3
ntrials = 20
aggregate_fitness = evolve.rank_reduce
maxfitness = aggregate_fitness([1]*ntrials)
time_const = line_location.line_location.timestep

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


    return aggregate_fitness(fitnesses)/maxfitness


def train(pop_size=100, max_gen=1, write_every=1, file=None):

    random.seed()

    with Pool(processes=cpu_count()) as pool:
        # goals = [0.55,0.65,0.75,0.85,0.95]
        # tasks = [(random.uniform(0,0.3),random.uniform(0,0.3),goal) for goal in goals]
        tasks = [(random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0.5,1.0)) for _ in range(ntrials)]


        batch_start = time.time()
        pop = evolve.initialise(pop_size)
        pop = evolve.assess(pop, pool, tasks, fitness)
        generation = 0
        while generation < max_gen:
            if file != None and generation % 20 == 0:
                evolve.log_fitness(pop, generation, None)
                print(f"Batch Time {time.time()-batch_start}")
                batch_start = time.time()

            generation += 1
            # goals = [0.55,0.65,0.75,0.85,0.95]
            # pos = (random.uniform(0,0.3),random.uniform(0,0.3))
            # tasks = [pos + (goal,) for goal in goals]
            tasks = [(random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0.5,1.0)) for _ in range(ntrials)]

            pop = evolve.assess(pop, pool, tasks, fitness)
            pop = evolve.rank_roulette_select(pop)
            pop = evolve.mutate(pop, pool, tasks, fitness)
            # print(pop)
            if write_every and generation % write_every==0:
                evolve.log_fitness(pop, generation, file)
        best = pop[0]

    return best

if __name__ == '__main__':
    start = time.time()
    path = f"logs/{int(start)}.txt"
    with open(path,'w') as f:
        print(f"Logs are in {path}")
        # initial pop could differ from final pop
        best = train(96,1000,file=f)
        print(f"Best fitness: {best['fitness']}")
        with open("best_genome.pkl",'wb') as g:
            pickle.dump(best["genome"],g)
    print(f"Finished training in {time.time() - start} seconds")