import ctrnn
import statistics
import random

mutationRate = 0.447
centerCrossing = False
class Citizen():
    def __init__(self,genome=None,fitness=None):
        if genome == None:
            genome = ctrnn.Genome(centerCrossing=centerCrossing)
        self.genome = genome
        self.fitness = fitness

def assess(pop, pool, tasks, fitness):
    """
    Calculate fitness score for each member across the given population

    Parameters:
        pop (list) : A list of members of the population
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        tasks (list) : A list of tasks used to assess the population
        fitness (func) : A function with type Genome, Task -> Float

    Return:
        The population, sorted by their newly assessed fitness scores 
    """

    pop = [(x,tasks,fitness) for x in pop]
    pop = pool.starmap(assess_item, pop)
    return sorted(pop, key = lambda i: i.fitness, reverse=True) 

def assess_item(item,tasks,fitness):
    """
    Calculate the fitness of a single genome

    Parameters:
        item : A single member of the population
        tasks (list) : A list of tasks used to assess the population
        fitness (func) : A function with type Genome, Task -> Float
    
    Return:
        item, with an updated fitness value

    """
    item.fitness = fitness(item.genome,tasks)
    return item

def mutate(pop, pool, tasks,fitness):
    """
    Mutate each member of the population

    Parameters:
        pop (list) : A list of members of the population
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        tasks (list) : A list of tasks used to assess the population
        fitness (func) : A function with type Genome, Task -> Float

    Return:
        pop, with a mutation applied to each member

    """
    pop = [(x,tasks,fitness) for x in pop]
    pop = pool.starmap(mutate_item,pop)
            
    return sorted(pop, key = lambda i: i.fitness, reverse=True) 

def mutate_item(item,tasks,fitness):
    """
    Mutate a single genome, only keep the result if it's better than the parent

    Parameters:
        item : A single member of the population
        tasks (list) : A list of tasks used to assess the population
        fitness (func) : A function with type Genome, Task -> Float
    
    Return:
        item, with a mutated genome (if it has better fitness than the original)
    """

    child = item.genome.copy()
    child.mutate(mutationRate)
    cfitness = fitness(child,tasks)
    if cfitness >= item.fitness:
        return Citizen(child,cfitness)
    return item

def rank_reduce(fitnesses):
    """
    Calculate a weighted sum of fitnesses, with stronger weighting to lower scores

    Parameters:
        fitnesses (list(float))
    
    Return:
        A single float, representing overall fitness
    """

    s_fitnesses = sorted(fitnesses)
    return sum([elem/(i+1) for i,elem in enumerate(s_fitnesses)])

def min_fitness(fitnesses):
    return min(fitnesses)

def initialise(initial_pop_size, pop_size, pool, tasks, fitness):
    """
    Initialise a population of random genomes

    Parameters:
        pop_size (int) : The number of members of the seed population
        pop_size (int) : The number of members population members to be returned
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        tasks (list) : A list of tasks used to assess the population
        fitness (func) : A function with type Genome, Task -> Float
        
    Return:
        A list containing pop_size members
    """

    pop = []
    while len(pop)<initial_pop_size:
        pop.append(Citizen())

    pop = assess(pop,pool,tasks,fitness)
    pop = [x for x in pop if x.fitness != 0]
    print(f"{len(pop)} non-zero members from a seed population of {initial_pop_size}")
    pop = selection(pop,size=pop_size)

    return pop

def selection(pop,elitism=0,size=None):
    elites = pop[:elitism]
    sel = rank_roulette_select(pop[elitism:],size)
    return sorted(elites+sel, key = lambda i: i.fitness, reverse=True) 


def rank_roulette_select(pop,size=None):
    """
    Perform roulette selection, weighted by rank rather than fitness

    Parameters:
        pop : A sorted list of population members

    Return:
        A new population chosen by rank based roulette selection
    """
    if size == None:
        size = len(pop)
    pop = random.choices(pop,reversed(range(1,len(pop)+1)),k=size)
    return pop


def truncation_select(pop):
    """
    Replace 0 fitness members of the population with strong fitness members

    Parameters:
        pop (list) : The initial population

    Return:
        The updated population
    """

    size = len(pop)
    survivors = (9*size)//10
    pop = pop[:survivors]
    for i in range(size-survivors):
        pop.append(pop[i].copy())
    return pop 



def log_fitness(pop, gen, file=None):
    """
    Write aggregate statistics of the population, either to a file or to stdout

    Parameters:
        pop (list) : The population
        gen (int) : The generation number
        file (file) : The file that will be written to, None -> Stdout
    """
    
    fitness = []
    for p in pop:
        fitness.append(p.fitness)

    line = "{:4d}: max:{:.3f}, min:{:.3f}, mean:{:.3f}".format(gen,max(fitness),min(fitness),statistics.mean(fitness))

    if file:
        file.write(line+"\n")
    else:
        print(line)

def main():
    pass

if __name__ == '__main__':
    main()
    