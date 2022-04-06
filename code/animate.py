import json
import pickle
import random
import sys
from multiprocessing import Pool, Value, cpu_count
from statistics import mean, stdev

import matplotlib.pyplot as plt

import ctrnn
import line_location

if len(sys.argv) < 6:
    print("Usage: py ./animate.py config.json genome_name.pkl senderstart receiverstart goal")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]


def runtrial(c, task):
    sp, rp, goal = task
    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
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
        plt.xlim(left=0,right=1.2)
        plt.ylim(bottom=0,top=-300)
        plt.xlabel("Position")
        plt.ylabel("Time")
        
        plt.vlines(0.3,0,-300,colors='black',linestyles='dotted')
        plt.vlines(goal,0,-300,colors='green',linestyles='dotted')

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
        plt.xlim(left=0,right=1.2)
        plt.ylim(bottom=0,top=-300)
        plt.xlabel("Position")
        plt.ylabel("Time")

        plt.vlines(0.3,-0.5,sim.t,colors='black',linestyles='dotted')
        plt.vlines(goal,-0.5,sim.t,colors='green',linestyles='dotted')
        
        history_c = len(sender_positions)
        history_ys = range(-history_c+1,1)

        plt.plot(sender_positions,history_ys,c="blue") 
        plt.plot(receiver_positions,history_ys,c="orange") 


        plt.savefig(f"./frames/{sim.t+i}")
        plt.close()


def main():
    with open(sys.argv[2], 'rb') as f:
        c = pickle.load(f)

    task = (float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))
    runtrial(c,task)
    
    
if __name__ == '__main__':
    main()
    