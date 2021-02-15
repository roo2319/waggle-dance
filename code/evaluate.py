import pickle
import random
import sys
import matplotlib.pyplot as plt
import line_location
import ctrnn
from multiprocessing import Pool, Value, cpu_count
from statistics import stdev, mean


def runtrial(c, task):
    sp, rp, goal = task
    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
    time_const = line_location.line_location.timestep

    sender = ctrnn.CTRNN(c)
    receiver = ctrnn.CTRNN(c)

    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    while sim.t < 3:
        senderstate = sim.getState(True)
        receiverstate = sim.getState(False)
        act1 = sender.eulerStep(senderstate,time_const)
        act2 = receiver.eulerStep(receiverstate,time_const)

        senderOut = act1[0]
        receiverOut = act2[0]

        sim.step(senderOut,receiverOut)


        # print(sim.getAsciiState())
    distance = abs(sim.goal - sim.receiverPos)
    if sim.fitness() > 0.95:
        return (1,distance)
    else:
        return (0,distance)


def main():
    if len(sys.argv) < 3:
        print("Usage: py ./evaluate.py genome_name.pkl ntrials")
        exit()

    with open(sys.argv[1], 'rb') as f:
        c = pickle.load(f)


    ntrials = int(sys.argv[2])
    success = 0
    absdist = 0
    
    with Pool(processes=cpu_count()) as pool:
        tasks = [(c,(random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0.5,1.0))) for _ in range(ntrials)]
        results = pool.starmap(runtrial,tasks)

    successes = [result[0] for result in results]
    distances = [result[1] for result in results]

    print(f"{sum(successes)} ({100*sum(successes)/ntrials}%) successes across {ntrials} trials")
    print(f"Mean absolute distance from goal: {mean(distances):.4f} (Standard deviation {stdev(distances):.4f})")

if __name__ == '__main__':
    main()
    