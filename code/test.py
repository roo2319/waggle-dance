import json
import pickle
import random
import sys

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

ntrials = 6


positions = (0,0)
# positions = (0.3,0.05)

sp,rp = positions
fig, ax = plt.subplots(3,ntrials)
# for i in range(ntrials):
for i,(hi,lo) in enumerate([(0.9,0.7),(0.8,0.6),(0.7,0.5),(0.9,0.7),(0.8,0.6),(0.7,0.5)]):
    displacement = [[],[],[]]
    rsens = [[],[],[]]
    ssens = [[],[],[]]
    # hi = random.uniform(0.65,1)
    # lo = random.uniform(0.5,hi-0.15)
    if i < ntrials//2:
        goals = hi,-lo
    else:
        goals = lo,-hi
        
    goal, goal2 = goals

    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal,goal2=goal2)
    goal = sim.truegoal
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

    ax[0][i].plot(displacement[0],displacement[1],color="tab:blue",label="A2")
    ax[0][i].plot(displacement[0],displacement[2],color="tab:orange",label="A1")
    ax[0][i].plot(displacement[0],[goal]*len(displacement[0]),color="tab:green",label="goal")
    ax[0][i].plot(displacement[0],[sim.falsegoal]*len(displacement[0]),color="tab:red",label="bad-goal")

    ax[0][i].set_title(f"A1 goal, A2 goal = {sim.goal:.2f},{sim.goal2:.2f}\nA1 End Pos = {sim.senderPos:.2f}\nA2 End Pos = {sim.receiverPos:.2f}\nFitness = {sim.fitness():.2f}")
    ax[0][i].legend()
    ax[0][i].set_ylim([-1,1])
    ax[0][0].set_ylabel("Position")


    ax[1][i].plot(displacement[0],ssens[0],color="tab:olive",label="Distance")
    ax[1][i].plot(displacement[0],ssens[1],color="tab:purple",label="Self Position")
    ax[1][i].plot(displacement[0],ssens[2],color="tab:cyan",label="Goal Distance")
    ax[1][i].set_title(f"A1 Sensors")
    ax[1][i].legend()
    ax[1][0].set_ylabel("Value")
    ax[1][i].set_ylim([-1,1])

    ax[2][i].plot(displacement[0],rsens[0],color="tab:olive",label="Distance")
    ax[2][i].plot(displacement[0],rsens[1],color="tab:purple",label="Self Position")
    ax[2][i].plot(displacement[0],rsens[2],color="tab:cyan",label="Goal Distance")
    ax[2][i].set_title(f"A2 Sensors")
    ax[2][i].legend()
    ax[2][0].set_ylabel("Value")
    ax[2][i].set_xlabel("Time")
    ax[2][i].set_ylim([-1,1])


        # print(sim.getAsciiState())



plt.show()
