import time
import random
import math

POPULATION_SIZE = 200
GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
ELITE_COUNT = 5

# define fitness function
def fitness(chrom, items, limit):
    weight = value = 0
    for i, gene in enumerate(chrom):
        if gene:
            weight += items[i][0]
            value += items[i][1]
            if weight > limit:
                return 0
    return value

# generate initial population
def gen_chromosome(length, items, limit):
    chrom = [random.randint(0, 1) for _ in range(length)]
    repair(chrom, items, limit)
    return chrom

def gen_population(size, length, items, limit):
    return [gen_chromosome(length, items, limit) for _ in range(size)]

# repair -> remove random
def repair(chrom, items, limit):
    curr = sum(items[i][0] for i, bit in enumerate(chrom) if bit)

    if curr <= limit:
        return
    
    efficiency = [(i, items[i][1]/items[i][0]) for i, bit in enumerate(chrom) if bit]
    efficiency.sort(key=lambda p: p[1])

    for i, _ in efficiency:
        chrom[i] = 0
        curr -= items[i][0]
        if curr <= limit:
            break



# roulette selection
def select_parent(pop, scores):
    contenders = random.sample(list(zip(pop, scores)), TOURNAMENT_SIZE)
    contenders.sort(key=lambda pair: pair[1], reverse=True)
    return contenders[0][0]

# crossover
def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

# mutation
def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUTATION_RATE:
            chrom[i] ^= 1

# genetic alg
def genetic_alg(M, N, items):
    # start_time = time.time()

    population = gen_population(POPULATION_SIZE, N, items, M)

    best_values = []

    for gen in range(GENERATIONS):
        scores = [fitness(c, items, M) for c in population]

        sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        new_pop = [p for p, _ in sorted_pop[:ELITE_COUNT]]

        while len(new_pop) < POPULATION_SIZE:
            p1 = select_parent(population, scores)
            p2 = select_parent(population, scores)

            c1, c2 = crossover(p1, p2)

            mutate(c1)
            mutate(c2)

            repair(c1, items, M)
            repair(c2, items, M)

            new_pop.append(c1)
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(c2)

        population = new_pop

        best_value = max(scores)
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