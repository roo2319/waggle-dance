from hashlib import new
import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev
from matplotlib.collections import LineCollection
import seaborn as sns

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
# ub = 126
ub = 301
lb = 50

def runtrial(c, goal):
    step = -1

    fit = []
    endpos = []
    count = 0 
    # print(goal,lb,ub)
    for goal2 in np.arange(-lb,-ub,step): #make me the whole range of other values
        # if abs(goal + goal2) < 15:
        #     fit.append(np.nan)
        #     continue
        if goal == -goal2:
            fit.append(np.nan)
            continue

        g = goal/100
        g2=goal2/100
        sim = line_location.line_location(senderPos=0,receiverPos=0,goal=g,goal2=g2)

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
        # rdistance = abs(sim.truegoal - sim.receiverPos)
        # sdistance = abs(sim.truegoal - sim.senderPos)
        # fit.append(1 if rdistance < 0.1 and sdistance < 0.1 else 0.5)
        # fit.append(1 if sim.fitness() > 0.9 else 0.5)
        fit.append(sim.fitness())

    return fit
       
def main():
    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)

    print(c)


    with Pool(processes=cpu_count()) as pool:
        tasks = [(c,goal) for goal in np.arange(lb,ub,1)]
        fit = pool.starmap(runtrial,tasks)

    print(np.nanmin(fit,))
    yt = [goal/100 for goal in np.arange(lb,ub,1)]
    xt=[-goal/100 for goal in np.arange(lb,ub,1)]
    # c = sns.color_palette("viridis", as_cmap=True)
    c = sns.color_palette("vlag", as_cmap=True)
    c.set_bad("black")

    ax = sns.heatmap(fit,xticklabels=xt,yticklabels=yt,cmap=c,vmin=0,vmax=1)
    ax.set_xticks(ax.get_xticks()[::10])
    ax.set_xticklabels(xt[::10])
    ax.set_xlabel("Negative Goal")
    ax.set_ylabel("Positive Goal")
    ax.set_yticks(ax.get_yticks()[::5])
    ax.set_yticklabels(yt[::5])
    ax.invert_yaxis()

    # high bound
    x = 0
    y = 15
    xs = [x]
    ys = [y]
    while y < 51:
        x += 1
        xs.append(x)
        ys.append(y)
        y += 1
        xs.append(x)
        ys.append(y)
    ax.plot(xs,ys,c='r')

    # low bound
    x = 15
    y = 0
    xs = [x]
    ys = [y]
    while x < 51:
        y += 1
        xs.append(x)
        ys.append(y)
        x += 1
        xs.append(x)
        ys.append(y)
    ax.plot(xs,ys,c='r')

    ax.plot(range(0,37),[51]*37,c='r')
    ax.plot([51]*37,range(0,37),c='r')
    plt.show()

    # fig, ax = plt.subplots(1,2)
    # ax[0].errorbar(goal,meandist,yerr=meanerr,fmt="bo")
    # ax[0].set(xlabel="Goal Location",ylabel="Mean Absolute Distance")
    # ax[1].errorbar(goal,endpos,yerr=enderr,fmt="bo")
    # ax[1].set(xlabel="Goal Location",ylabel="Mean End Position")
    # ax[1].plot(np.arange(0.5,1,0.01),np.arange(0.5,1,0.01))
    # fig.suptitle(sys.argv[2])
    # plt.show()

if __name__ == '__main__':
    main()
    