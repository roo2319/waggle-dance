import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev

import matplotlib.pyplot as plt

import ctrnn
import cube_location

if len(sys.argv) < 4:
    print("Usage: py ./evaluate.py config.json genome_name.pkl ntrials")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
cube_location.motorFunction = cube_location.motors[settings.get("motor","clippedMotor1")]


def runtrial(c):
    sim = cube_location.cube_location()
    time_const = cube_location.cube_location.timestep

    sender = ctrnn.CTRNN(c)
    receiver = ctrnn.CTRNN(c)

    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    while sim.t < simulation_seconds:
        senderstate = sim.getState(True)
        receiverstate = sim.getState(False)
        act1 = sender.eulerStep(senderstate,time_const)
        act2 = receiver.eulerStep(receiverstate,time_const)

        senderOut = act1
        receiverOut = act2

        sim.step(senderOut,receiverOut)


        # print(sim.getAsciiState())
    distance = sim.goal.dist(sim.receiverPos)
    if sim.fitness() > 0.9:
        return (1,distance,0,0)
    else:
        return (0,distance,0,0)


def main():
    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)


    ntrials = int(sys.argv[3])
    success = 0
    absdist = 0
    
    with Pool(processes=cpu_count()) as pool:
        tasks = [c for _ in range(ntrials)]
        results = pool.map(runtrial,tasks)

    successes = [result[0] for result in results]
    distances = [result[1] for result in results]
    nudges    = [result[2] for result in results]
    ctime     = [result[3] for result in results]
    print(len(results))
    print(f"{sum(successes)} ({100*sum(successes)/ntrials}%) successes across {ntrials} trials")
    print(f"Mean absolute distance from goal: {mean(distances):.4f} (Standard deviation {stdev(distances):.4f})")
    print(f"Mean Nudges: {mean(nudges)} ")
    print(f"Mean ctime: {mean(ctime)}")

if __name__ == '__main__':
    main()
    