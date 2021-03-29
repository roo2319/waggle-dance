import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev

import matplotlib.pyplot as plt

import ctrnn
import line_location

if len(sys.argv) < 4:
    print("Usage: py ./evaluate.py config.json genome_name.pkl ntrials")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]


def runtrial(c, task):
    sp, rp = task
    hi = random.uniform(0.5,1)
    lo = random.uniform(0.5,hi)
    if random.random() < 0.5:
        goals = lo,-hi
    else:
        goals = hi,-lo
        
    goal, goal2 = goals

    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal, goal2=goal2)
    time_const = line_location.line_location.timestep

    sender = ctrnn.CTRNN(c,time_const)
    receiver = ctrnn.CTRNN(c,time_const)

    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    while sim.t < simulation_seconds:
        senderstate = sim.getState(True)
        receiverstate = sim.getState(False)
        act1 = sender.eulerStep(senderstate)
        act2 = receiver.eulerStep(receiverstate)

        senderOut = act1[0]
        receiverOut = act2[0]

        sim.step(senderOut,receiverOut)


        # print(sim.getAsciiState())
    rdistance = abs(sim.goal - sim.receiverPos)
    sdistance = abs(sim.goal2 - sim.senderPos)

    if sim.fitness() > 0.9:
        return (1,rdistance,sdistance,sim.touches,sim.ctime)
    else:
        return (0,rdistance,sdistance,sim.touches,sim.ctime)


def main():
    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)


    ntrials = int(sys.argv[3])
    success = 0
    absdist = 0
    
    with Pool(processes=cpu_count()) as pool:
        tasks = [(c,(0,0)) for _ in range(ntrials)]
        results = pool.starmap(runtrial,tasks)

    successes  = [result[0] for result in results]
    rdistances  = [result[1] for result in results]
    sdistances = [result[2] for result in results]
    nudges     = [result[3] for result in results]
    ctime      = [result[4] for result in results]

    print(f"{sum(successes)} ({100*sum(successes)/ntrials}%) successes across {ntrials} trials")
    print(f"Mean absolute rdistance from goal: {mean(rdistances):.4f} (Standard deviation {stdev(rdistances):.4f})")
    print(f"Mean absolute sdistance from goal: {mean(sdistances):.4f} (Standard deviation {stdev(sdistances):.4f})")
    print(f"Mean Nudges: {mean(nudges)} ")
    print(f"Mean ctime: {mean(ctime)}")

if __name__ == '__main__':
    main()
    