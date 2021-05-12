import json
import pickle
import random
import itertools
import sys

import matplotlib.pyplot as plt

import ctrnn
import line_location
import numpy as np

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
# for i in range(ntrials):
inputs = [0,0,-0.73]
# This is decoupled, what would coupled look like?
for _ in range(1):
    # inputs = [np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)]
    print(inputs)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for state in itertools.product([-10,10],repeat=3):
        # state = [np.random.uniform(-10,10),np.random.uniform(-10,10),np.random.uniform(-10,10)]
        
        states = state
        t = 0
            
        time_const = line_location.line_location.timestep

        brain = ctrnn.CTRNN(c,time_const)
        brain.setStates(np.array(state,dtype=np.float64))
        # Run the given simulation for up to num_steps time steps.
        while t < 10000:
            # if sim.t < 1.5:
            #     receiverstate[0] = 0
            #     senderstate[0] = 0

            brain.eulerStep(inputs)

            states = np.vstack([states,brain.states])
            t += time_const
        
        ax.plot3D(states[:,0],states[:,1],states[:,2])
        print(states[-1])
        ax.scatter3D(states[::10,0],states[::10,1],states[::10,2])
        ax.set_xlabel("Neuron 1")
        ax.set_ylabel("Neuron 2")
        ax.set_zlabel("Neuron 3")


            # print(sim.getAsciiState())

    plt.show()
