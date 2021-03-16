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


def fitness(genome,rs):
    fitnesses = []

    random.seed(rs)
    for sp, rp, goal in [(random.uniform(0,0.3), random.uniform(0,0.3), random.uniform(0.5,1)) for _ in range(ntrials)]:
        sender = ctrnn.CTRNN(genome)
        receiver = ctrnn.CTRNN(genome)
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


def evaluate(genome, rs):
    fitnesses = []

    random.seed(rs)
    for sp, rp, goal in [(random.uniform(0,0.3), random.uniform(0,0.3), x) for x in np.arange(0.5,1,0.01)]:
        sender = ctrnn.CTRNN(genome)
        receiver = ctrnn.CTRNN(genome)
        sim = line_location.line_location(senderPos=sp,receiverPos=rp, goal=goal)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderOut   = sender.eulerStep(sim.getState(True),time_const)[0]
            receiverOut = receiver.eulerStep(sim.getState(False),time_const)[0]

            sim.step(senderOut,receiverOut)

        fitness = 1 if abs(sim.receiverPos - sim.goal) <= 0.05 else 0


        fitnesses.append(fitness)


    return statistics.mean(fitnesses)

def train(pop_size=100, max_gen=1, write_every=1, file=None):

    with Pool(processes=cpu_count()) as pool:
        rs = random.random()
        batch_start = time.time()
        pop = evolve.initialise(pop_size)
        generation = 0
        mcount = 0
        mcount2 = 0
        while generation < max_gen:
            rs = random.random()
            pop = evolve.assess(pop, pool, fitness,rs)
            if generation % 20 == 0 and file != None:
                evolve.log_fitness(pop, generation, mcount2, None)
                print(f"Batch Time {time.time()-batch_start}")
                batch_start = time.time()
                mcount2 = 0
                with open("models/checkpoint.pkl",'wb') as g:
                    pickle.dump(pop[0].genome,g)
            if write_every and generation % write_every==0:
                evolve.log_fitness(pop, generation, mcount, file)
                mcount = 0
            generation += 1
            pop = evolve.sus(pop)
            pop, c = evolve.mutate(pop, pool, fitness,rs)
            mcount += c
            mcount2 += c

            if generation % 100 == 0:
                rs = random.random()
                pop = evolve.assess(pop, pool, evaluate, rs)
                print([x.fitness for x in pop])
                
        rs = random.random()
        pop = evolve.assess(pop, pool, evaluate, rs)
    return pop[0]



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
