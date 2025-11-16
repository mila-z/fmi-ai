import time
import random

POPULATION_SIZE = 200
GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
ELITE_COUNT = 5

# define fitness function
def fitness(chrom, items, limit):
    """
    fitness function - how good a chromosome is
    """
    # total weight and value of selected items
    weight = value = 0
    # loop over the bits
    for i, gene in enumerate(chrom):
        if gene:
            weight += items[i][0]
            value += items[i][1]
            # if we have gone over the allowed weigh, then the chromosome is not "good"
            if weight > limit:
                return 0
    return value

def gen_chromosome(length, items, limit):
    """
    creates a single random chromosome
    """
    # generates a random list of '0' and '1'
    chrom = [random.randint(0, 1) for _ in range(length)]
    # if the weight limit is exceeded we "fix" the chromosome
    repair(chrom, items, limit)
    return chrom

def gen_population(size, length, items, limit):
    """
    generates a population - a list of chromosomes
    """
    return [gen_chromosome(length, items, limit) for _ in range(size)]

def repair(chrom, items, limit):
    """
    fix a chromosome if it is not "good"
    """
    # get the total weight 
    curr = sum(items[i][0] for i, bit in enumerate(chrom) if bit)

    # if it does not exceed the limit, then the chrom is good
    if curr <= limit:
        return
    
    # efficiency = [(index, value to weight ratio) for items in the knapsack]
    efficiency = [(i, items[i][1]/items[i][0]) for i, bit in enumerate(chrom) if bit]
    # sort is by the worst being first
    efficiency.sort(key=lambda p: p[1])

    # removes until the weight is <= the limit
    for i, _ in efficiency:
        chrom[i] = 0
        curr -= items[i][0]
        if curr <= limit:
            break

def select_parent(pop, scores):
    """
    tournament selection for the parent
    """
    # pick TOURNAMENT_SIZE chromosomes randomly from the population
    contenders = random.sample(list(zip(pop, scores)), TOURNAMENT_SIZE)
    # get the ones with the best fitness
    contenders.sort(key=lambda pair: pair[1], reverse=True)
    # return the first chromosome
    return contenders[0][0]

def crossover(p1, p2):
    """
    one point crossover - first + second part of the parents
    """
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(chrom):
    """
    mutation - randomly flip a bit
    """
    for i in range(len(chrom)):
        if random.random() < MUTATION_RATE:
            chrom[i] ^= 1

def genetic_alg(M, N, items):
    """
    main genetic algorithm:
    M - max weight
    N - max num of items
    items = [(weight, value)]
    we want least weight and most value
    """
    # start_time = time.time()

    # generate a population
    population = gen_population(POPULATION_SIZE, N, items, M)

    # store the best fitness values from certain generations
    best_values = []

    for gen in range(GENERATIONS):
        # fitness values for each chromosome in the population
        scores = [fitness(c, items, M) for c in population]

        # sort the population by best fitness values first
        sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        # get the top ELITE_COUNT chroms 
        new_pop = [p for p, _ in sorted_pop[:ELITE_COUNT]]

        while len(new_pop) < POPULATION_SIZE:
            # selection
            p1 = select_parent(population, scores)
            p2 = select_parent(population, scores)

            # crossover
            c1, c2 = crossover(p1, p2)

            # mutation
            mutate(c1)
            mutate(c2)

            # fix
            repair(c1, items, M)
            repair(c2, items, M)
            
            # add to new population
            new_pop.append(c1)
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(c2)

        # replace the population
        population = new_pop

        # best value is the best fitness in this generation
        best_value = max(scores)
        # if it is the first, last or every 9th gen, print the value and store it in best values
        if gen == 0 or gen == GENERATIONS - 1 or gen % (GENERATIONS // 9) == 0:
            print(best_value)
            best_values.append(best_value)

    print()
    print(best_values[-1])
    # print("Time:", round(time.time() - start_time, 3), "seconds")

if __name__ == "__main__":
    M, N = map(int, input().split())

    items =[tuple(map(int, input().split())) for _ in range(N)]

    genetic_alg(M, N, items)