import json
import pickle
import random
import sys
import numpy as np

import matplotlib.pyplot as plt

import ctrnn
import line_location

if len(sys.argv) < 3:
    print("Usage: py ./test.py config.json genome_name.pkl")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
line_location.motorFunction = line_location.motors[settings.get("motor","clippedMotor1")]

# load the winner
with open(sys.argv[2], 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

ntrials = 3

goal = 1
low = -0.5
high = -1
fig,ax = plt.subplots(2,1)
# for goal in np.linspace(0.5,1,50):
for goal in [0.5,1]:
    first = True
    if goal == 1:
        high = -0.85
    for goal2 in np.linspace(low,high,25):
            print(goal,goal2)
            if abs(goal + goal2) < 0.15:
                continue
        
            if first:
                first = False
                col = 'r'
                a = 1
            elif goal2 == high:
                col = 'g'
                a = 1
            elif goal == 0.5:
                col = 'tab:pink'
                a = 0.1
            else:
                col = 'tab:cyan'
                a = 0.1
        
            displacement = [[],[],[]]
            rsens = [[],[],[]]
            ssens = [[],[],[]]
        
            # sim = line_location.line_location(senderPos=0,receiverPos=0,goal=goal,goal2=goal2)
            sim = line_location.line_location(senderPos=0,receiverPos=0,goal=goal2,goal2=goal)

            time_const = line_location.line_location.timestep
        
        
            sender = ctrnn.CTRNN(c,time_const)
            receiver = ctrnn.CTRNN(c,time_const)
        
            sender.reset()
            receiver.reset()
            vals = []
            # Run the given simulation for up to num_steps time steps.
            while sim.t < simulation_seconds:
                senderstate = sim.getState(True)
                receiverstate = sim.getState(False)
                # senderstate[0] = 1 \ changing this has a massive effect on receiver only
                # if sim.t < 1.5:
                #     receiverstate[0] = 0
                #     senderstate[0] = 0
        
                act1 = sender.eulerStep(senderstate)
                act2 = receiver.eulerStep(receiverstate)
        
                vals.append(act1)
                vals.append(act2)
        
                senderOut = act1[0]
                receiverOut = act2[0]
                # We can model force here
                # print(f"network thinks {act1,act2}")
                # We can model force here
                sim.step(senderOut,receiverOut)
                r,s,t = sim.getLoggingData()
        
                displacement[0].append(t)
                displacement[1].append(r)
                displacement[2].append(s)
        
                for j in range(3):
                    rsens[j].append(receiverstate[j])
                    ssens[j].append(senderstate[j])
        
            ax[0].plot(displacement[0],displacement[2],col if col is not None else 'tab:blue',alpha=a,label="sender")
            ax[1].plot(displacement[0],displacement[1],col if col is not None else 'tab:orange',alpha=a,label="receiver")
            # ax[0].plot(displacement[0],displacement[2],'tab:blue',alpha=0.1,label="sender")
            # ax[1].plot(displacement[0],displacement[1],'tab:orange',alpha=0.1,label="receiver")
    
            # plt.plot(displacement[0],[goal]*len(displacement[0]),label="goal")
            # plt.plot(displacement[0],[sim.falsegoal]*len(displacement[0]),label="bad-goal")









# ax[0].set_title(f"A1 Position (Fixed Goal)")
# ax[1].set_title(f"A2 Position (Variable Goal)")
ax[0].set_title(f"A1 Position (Variable Goal)")
ax[1].set_title(f"A2 Position (Fixed Goal)")
# plt.legend()
# plt.set_ylim([-1,1])
plt.show()

