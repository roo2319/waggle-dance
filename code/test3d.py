import json
import pickle
import random
import itertools
import sys

import matplotlib.pyplot as plt

import ctrnn
import cube_location

if len(sys.argv) < 3:
    print("Usage: py ./test3d.py config.json genome_name.pkl")
    exit()

with open(sys.argv[1],'r') as config:
    settings = json.load(config)

simulation_seconds = settings.get("simulation_seconds",3)
cube_location.motorFunction = cube_location.motors[settings.get("motor","clippedMotor1")]

# load the winner
with open(sys.argv[2], 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)



positions = (random.uniform(0,0.3), random.uniform(0,0.3),random.uniform(0,0.3)),(random.uniform(0,0.3), random.uniform(0,0.3))
# positions = (0.3,0.05)

sp,rp = map(lambda x: cube_location.Point(*x), positions)
print("Initial conditions:")
print(f"   Sender = {sp}")
print(f" Receiver = {rp}")
print()
fig, ax = plt.subplots(3,5)
i = 0
for g in itertools.product([0,1],repeat=3):
    goal = cube_location.Point(*g)
    displacement = [[],[],[]]
    rsens = [[],[],[]]
    ssens = [[],[],[]]
    sim = cube_location.cube_location(senderPos=sp,receiverPos=rp,goal=goal)
    time_const = cube_location.cube_location.timestep


    sender = ctrnn.CTRNN(c)
    receiver = ctrnn.CTRNN(c)

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

        act1 = sender.eulerStep(senderstate,time_const)
        act2 = receiver.eulerStep(receiverstate,time_const)

        vals.append(act1)
        vals.append(act2)

        senderOut = act1
        receiverOut = act2
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

    # ax[0][i].plot(displacement[0],displacement[1],label="receiver")
    # ax[0][i].plot(displacement[0],displacement[2],label="sender")
    # ax[0][i].plot(displacement[0],[goal]*len(displacement[0]),label="goal")
    # ax[0][i].set_title(f"Goal = {goal}\nReceiver = {sim.receiverPos:.2f}\nFitness = {sim.fitness():.2f}")
    # ax[0][i].legend()
    # ax[1][i].plot(displacement[0],rsens[0],label="Contact")
    # ax[1][i].plot(displacement[0],rsens[1],label="Self Position")
    # # ax[1][i].plot(displacement[0],rsens[2],label="Constant Value")
    # ax[1][i].set_title(f"Receiver Sensors")
    # ax[1][i].legend()
    # ax[2][i].plot(displacement[0],ssens[0],label="Contact")
    # ax[2][i].plot(displacement[0],ssens[1],label="Self Position")
    # ax[2][i].plot(displacement[0],ssens[2],label="Goal Distance")
    # ax[2][i].set_title(f"Sender Sensors")
    # ax[2][i].legend()
    i+=1
        # print(sim.getAsciiState())




    print("Final conditions:")
    print(f"   Sender = {sim.senderPos}")
    print(f" Receiver = {sim.receiverPos}")
    print(f"     Goal = {sim.goal}")
    print(f" fitness = {sim.fitness()}")
    print()
    # print(f"Max {max(vals)},Min {min(vals)}, Average {sum(vals)/len(vals)}")

# plt.show()
