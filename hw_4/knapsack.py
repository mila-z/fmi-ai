import time
import random
import math

POPULATION_SIZE = 200
GENERATIONS = 100
# TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.015
ELITE_COUNT = 5

# define fitness function
def fitness(chrom, items, max_weight):
    weight = value = 0
    for bit, (w, v) in zip(chrom, items):
        if bit:
            weight += w
            value += v
    if weight > max_weight:
        return 0
    return value

# generate initial population
def random_chromosome(n):
    return [random.randint(0, 1) for _ in range(n)]

def generate_population(size, n):
    return [random_chromosome(n) for _ in range(size)]

# repair -> remove random
def repair(chrom, items, max_weight):
    while True:
        total_w = sum(items[i][0] for i in range(len(chrom)) if chrom[i])
        if total_w <= max_weight:
            return
        idxs = [i for i in range(len(chrom)) if chrom[i]]
        remove_i = random.choice(idxs)
        chrom[remove_i] = random.choice(idxs)
        chrom[remove_i] = 0

# roulette selection
def roulette_selection(population, fitnesses):
    total_f = sum(fitnesses)
    if total_f == 0:
        return random.choice(population)
    
    pick = random.uniform(0, total_f)
    current = 0
    for chrom, f in zip(population, fitnesses):
        current += f
        if current >= pick:
            return chrom
    return population[-1]

# uniform crossover
def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for a, b in zip(p1, p2):
        if random.random() < 0.05:
            c1.append(a)
            c2.append(b)
        else:
            c1.append(b)
            c2.append(a)
    return c1, c2

# mutation
def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUTATION_RATE:
            chrom[i] ^= 1

# genetic alg
def genetic_algorithm(M, N, items):
    # start_time = time.time()

    population = generate_population(POPULATION_SIZE, N)

    best_values = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(c, items, M) for c in population]

        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_COUNT]
        new_population = [population[i][:] for i in elite_indices]

        while len(new_population) < POPULATION_SIZE:
            p1 = roulette_selection(population, fitnesses)
            p2 = roulette_selection(population, fitnesses)

            c1, c2 = uniform_crossover(p1, p2)

            mutate(c1)
            mutate(c2)

            repair(c1, items, M)
            repair(c2, items, M)

            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)

        population = new_population

        best_value = max(fitness(c, items, M) for c in population)
        if gen == 0 or gen == GENERATIONS - 1 or gen % (GENERATIONS // 9) == 0:
            print(best_value)
            best_values.append(best_value)

    print()
    print(best_values[-1])
    # print("Time:", round(time.time() - start_time, 3), "seconds")

if __name__ == "__main__":
    M, N = map(int, input().split())

    items = []
    for _ in range(N):
        w, v, = map(int, input().split())
        items.append((w, v))

    genetic_algorithm(M, N, items)