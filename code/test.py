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



positions = (random.uniform(0,0.3),random.uniform(0,0.3))
# positions = (0.3,0.05)

sp,rp = positions
print("Initial conditions:")
print("   Sender = {0:.4f}".format(sp))
print(" Receiver = {0:.4f}".format(rp))
print()
fig, ax = plt.subplots(3,3)
i = 0
# for goal in [0.5,0.6,0.7,0.8,0.9]:
for goal in [0.5,0.7,0.9]:
    displacement = [[],[],[]]
    rsens = [[],[],[]]
    ssens = [[],[],[]]
    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
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

    ax[0][i].plot(displacement[0],displacement[1],color="tab:blue",label="receiver")
    ax[0][i].plot(displacement[0],displacement[2],color="tab:orange",label="sender")
    ax[0][i].plot(displacement[0],[goal]*len(displacement[0]),color="tab:green",label="goal")
    ax[0][i].set_title(f"Target Position = {goal}\nFinal Receiver Position = {sim.receiverPos:.2f}\nFitness = {sim.fitness():.2f}")
    ax[0][i].legend()
    ax[0][i].set_ylabel("Position")
    ax[1][i].plot(displacement[0],rsens[0],color="tab:red",label="Contact")
    ax[1][i].plot(displacement[0],rsens[1],color="tab:purple",label="Self Position")
    # ax[1][i].plot(displacement[0],rsens[2],label="Constant Value")
    ax[1][i].set_title(f"Receiver Sensors")
    ax[1][i].legend()
    ax[1][i].set_ylabel("Value")

    ax[2][i].plot(displacement[0],ssens[0],color="tab:red",label="Contact")
    ax[2][i].plot(displacement[0],ssens[1],color="tab:purple",label="Self Position")
    ax[2][i].plot(displacement[0],ssens[2],color="tab:cyan",label="Goal Distance")
    ax[2][i].set_title(f"Sender Sensors")
    ax[2][i].set_ylabel("Value")
    ax[2][i].set_xlabel("Time")
    ax[2][i].legend()
    i+=1
        # print(sim.getAsciiState())




    print("Final conditions:")
    print("   Sender = {0:.4f}".format(sim.senderPos))
    print(" Receiver = {0:.4f}".format(sim.receiverPos))
    print("     Goal = {0:.4f}".format(sim.goal))
    print(f"  Nudges = {sim.touches}")
    print(f"  C time = {sim.ctime}")
    print(f" fitness = {sim.fitness()}")
    print()
    print(f"Max {max(vals)},Min {min(vals)}, Average {sum(vals)/len(vals)}")

plt.show()
