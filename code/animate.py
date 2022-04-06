import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev

import matplotlib.pyplot as plt

import ctrnn
import line_location

if len(sys.argv) < 5:
    print("Usage: py ./animate.py config.json genome_name.pkl goal goal2")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]


def runtrial(c, task):
    goal,goal2 = task
    sim = line_location.line_location(senderPos=0,receiverPos=0,goal=goal,goal2=goal2)
    time_const = line_location.line_location.timestep

    sender = ctrnn.CTRNN(c,time_const)
    receiver = ctrnn.CTRNN(c,time_const)

    sender.reset()
    receiver.reset()
    sender_positions = []
    receiver_positions = []
    # Run the given simulation for up to num_steps time steps.
    while sim.t < simulation_seconds:
        senderstate = sim.getState(True)
        receiverstate = sim.getState(False)
        act1 = sender.eulerStep(senderstate)
        act2 = receiver.eulerStep(receiverstate)

        senderOut = act1[0]
        receiverOut = act2[0]

        sim.step(senderOut,receiverOut)
        sender_positions.append(sim.senderPos)
        receiver_positions.append(sim.receiverPos)
        plt.figure()
        # plt.vlines(0.3,-0.5,sim.t,colors='black',linestyles='dotted')

        plt.xlim(left=-1.2,right=1.2)
        plt.ylim(bottom=0,top=-300)
        plt.xlabel("Position")
        plt.ylabel("Time")
        plt.vlines(sim.truegoal,0,-300,colors='green',linestyles='dotted')
        plt.vlines(sim.falsegoal,0,-300,colors='red',linestyles='dotted')

        history_c = len(sender_positions)
        history_ys = range(-history_c+1,1)
        alphalist = [i/history_c for i in range(history_c)] if history_c != 0 else 1
        
        plt.scatter(sender_positions,history_ys,c="blue",alpha=alphalist) 
        plt.scatter(receiver_positions,history_ys,c="orange",alpha=alphalist) 

        plt.savefig(f"./frames/{sim.t}")
        plt.close()
        # print(sim.getAsciiState())
        
    # 2 second pause at the end
    for i in range(1,49):
        plt.figure()
        plt.xlim(left=-1.2,right=1.2)
        plt.ylim(bottom=0,top=-300)
        plt.xlabel("Position")
        plt.ylabel("Time")
        plt.vlines(sim.truegoal,0,-300,colors='green',linestyles='dotted')
        plt.vlines(sim.falsegoal,0,-300,colors='red',linestyles='dotted')
        
        history_c = len(sender_positions)
        history_ys = range(-history_c+1,1)

        plt.scatter(sender_positions,history_ys,c="blue",alpha=alphalist) 
        plt.scatter(receiver_positions,history_ys,c="orange",alpha=alphalist) 

        plt.savefig(f"./frames/{sim.t+i}")
        plt.close()


def main():
    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)

    task = (float(sys.argv[3]),float(sys.argv[4]))
    runtrial(c,task)
    
    
if __name__ == '__main__':
    main()
    