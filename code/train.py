import json
import pickle
import random
import statistics
import sys
import numpy as np
import time
from multiprocessing import Pool, Value, cpu_count
from multiprocessing.context import ProcessError
from os import supports_effective_ids

import ctrnn
import evolve
import line_location

if len(sys.argv) < 2:
    print("Usage: train.py config.json")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

elitism = settings.get("elitism",0)
generations = settings.get("generations", 1000)
ntrials = settings.get("ntrials",20)
population_size = settings.get("population_size",96)
simulation_seconds = settings.get("simulation_seconds",3)

evolve.mutationRate = settings.get("mutationRate",0.447)
evolve.centerCrossing = settings.get("centerCrossing", False)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]


aggregate_fitness = evolve.rank_reduce
maxfitness = aggregate_fitness([1]*ntrials)
time_const = line_location.line_location.timestep


def fitness(genome):
    sender = ctrnn.CTRNN(genome)
    receiver = ctrnn.CTRNN(genome)
    fitnesses = []

    for sp, rp, goal in [(random.uniform(0,0.3), random.uniform(0,0.3), random.uniform(0.5,1)) for _ in range(ntrials)]:
        sim = line_location.line_location(senderPos=sp,receiverPos=rp, goal=goal)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderOut   = sender.eulerStep(sim.getState(True),time_const)[0]
            receiverOut = receiver.eulerStep(sim.getState(False),time_const)[0]

            sim.step(senderOut,receiverOut)

        fitness = sim.fitness()


        fitnesses.append(fitness)


    return aggregate_fitness(fitnesses)/maxfitness


def train(pop_size=100, max_gen=1, write_every=1, file=None):

    with Pool(processes=cpu_count()) as pool:
        batch_start = time.time()
        pop = evolve.initialise(pop_size, pool,fitness)
        generation = 0
        while generation < max_gen:
            pop = evolve.assess(pop, pool, fitness)

            if file != None and generation % 20 == 0:
                evolve.log_fitness(pop, generation, None)
                print(f"Batch Time {time.time()-batch_start}")
                batch_start = time.time()
                with open("models/checkpoint.pkl",'wb') as g:
                    pickle.dump(pop[0].genome,g)
            if write_every and generation % write_every==0:
                evolve.log_fitness(pop, generation, file)
            generation += 1

            pop = evolve.sus(pop)
            pop = evolve.mutate(pop, pool, fitness)
        
        
        best = pop[0]

    return best



def main():
    print(f"Configuration is \n\tElitism : {elitism}\n\tNumber of generations : {generations}\n\tNumber of trials : {ntrials}\n\tPopulation size : {population_size}\n\tSimulation length {simulation_seconds} seconds\n\tMutation Rate : {evolve.mutationRate}\n\tCenter Crossing : {evolve.centerCrossing}\n\tMotor function : {settings.get('motor','clippedMotor1')}")

    start = time.time()
    path = f"logs/{int(start)}.txt"
    with open(path,'w') as f:
        print(f"Logs are in {path}")
        best = train(population_size,generations,file=f)
        print(f"Best fitness: {best.fitness}")
        with open("models/best_genome.pkl",'wb') as g:
            pickle.dump(best.genome,g)
    print(f"Finished training in {time.time() - start} seconds")

if __name__ == '__main__':
    main()
