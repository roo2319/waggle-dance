import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

import ctrnn
import line_location

simulation_seconds = 3

if len(sys.argv) < 3:
    print("Usage: py ./meandistance.py config.json genome_name.pkl")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)
simulation_seconds = settings.get("simulation_seconds",3)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]

def runtrial(c, goal):
    dist = []
    endpos = []
    count = 0 
    for sp,rp in [(sp, rp) for sp in np.arange(0,0.3,0.03) for rp in np.arange(0,0.3,0.03)]:
        sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
        time_const = line_location.line_location.timestep

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

            senderOut = act1[0]
            receiverOut = act2[0]

            sim.step(senderOut,receiverOut)


            # print(sim.getAsciiState())
        dist.append(abs(sim.goal - sim.receiverPos))
        endpos.append(sim.receiverPos)
    return (mean(dist),stdev(dist),mean(endpos),stdev(endpos),goal)
       

def main():


    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)

    print(c)

    with Pool(processes=cpu_count()) as pool:
        tasks = [(c,goal) for goal in np.arange(0.5,1,0.01)]
        results = pool.starmap(runtrial,tasks)

    meandist = [result[0] for result in results]
    meanerr = [result[1] for result in results]
    endpos = [result[2] for result in results]
    enderr = [result[3] for result in results]

    goal = [result[4] for result in results]

    fig, ax = plt.subplots(1,2)
    ax[0].errorbar(goal,meandist,yerr=meanerr,fmt="bo")
    ax[0].set(xlabel="Goal Location",ylabel="Mean Absolute Distance")
    ax[1].errorbar(goal,endpos,yerr=enderr,fmt="bo")
    ax[1].set(xlabel="Goal Location",ylabel="Mean End Position")
    ax[1].plot(np.arange(0.5,1,0.01),np.arange(0.5,1,0.01))
    fig.suptitle(sys.argv[2])
    plt.show()

if __name__ == '__main__':
    main()
    