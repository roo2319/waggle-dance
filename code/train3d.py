import json
import pickle
import random
import statistics
import sys
import numpy as np
import time
import itertools
from multiprocessing import Pool, Value, cpu_count
from multiprocessing.context import ProcessError
from os import supports_effective_ids
from statistics import mean

import ctrnn
import evolve
import cube_location

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
cube_location.motorFunction = cube_location.motors[settings.get("motor","clippedMotor1")]


aggregate_fitness = evolve.rank_reduce
maxfitness = aggregate_fitness([1]*ntrials)
time_const = cube_location.cube_location.timestep

goals = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
goal = [cube_location.Point(*goals[0])]
goals.remove(goals[0])


def fitness(genome,rs,goal):
    fitnesses = []
    random.seed(rs)
    # for sp, rp in [((random.uniform(-0.3,0.3), random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)),(random.uniform(-0.3,0.3), random.uniform(-0.3,0.3),random.uniform(-0.3,0.3))) for _ in range(ntrials)]:
    for _ in range(ntrials):    
        sender = ctrnn.CTRNN(genome)
        receiver = ctrnn.CTRNN(genome)
        # sp = cube_location.Point(*sp)
        # rp = cube_location.Point(*rp)
        sp = cube_location.Point(*[0,0,0])
        rp = cube_location.Point(*[0,0,0])
        # goal = cube_location.Point(*goal)


        sim = cube_location.cube_location(senderPos=sp,receiverPos=rp, goal=random.choice(goal))

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            senderOut   = sender.eulerStep(sim.getState(True),time_const)
            receiverOut = receiver.eulerStep(sim.getState(False),time_const)

            sim.step(senderOut,receiverOut)

        fitness = sim.fitness()


        fitnesses.append(fitness)


    return aggregate_fitness(fitnesses)/maxfitness

def train(pop_size=100, max_gen=1, write_every=1, file=None):
    global goal
    with Pool(processes=cpu_count()) as pool:
        rs = random.random()
        batch_start = time.time()
        pop = evolve.initialise(pop_size,shape=(6,6,3))
        generation = 0
        mcount = 0
        mcount2 = 0
        nextfit = 0.1
        while generation < max_gen:
            rs = random.random()
            pop = evolve.assess(pop, pool, fitness,rs,goal)
            meanfit = mean([x.fitness for x in pop])
            if meanfit > nextfit:
                nextfit = min(meanfit * 1.2,0.9)
                if len(goals) != 0:
                    ng = random.choice(goals)
                    goals.remove(ng)
                    goal.append(cube_location.Point(*ng))
                print(f"Changed goal to {goal}")

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
            pop, c = evolve.mutate(pop, pool, fitness,rs,goal)
            mcount += c
            mcount2 += c

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
