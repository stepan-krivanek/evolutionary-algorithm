
import numpy as np

from LocalSearch.local_search import *
from EvolutionAlgorithm import EvolutionAlgorithm as EA


def init_real_chromosome(size, variation=2):
    return np.random.normal(0, variation, size)


def init_bin_chromosome(size):
    return np.random.binomial(1, 0.5, size)


def initialization(population_size, init_f):
    return np.array([init_f() for x in range(population_size)])


def tournament_selection(parents_num, fitness):
    result = np.empty(parents_num, dtype='int32')

    k = int(fitness.size / 4) + 1
    p = 0.5

    indices = np.arange(0, fitness.size)
    probabilities = np.array([p * (1 - p) ** x for x in range(k)])
    probabilities = probabilities / np.sum(probabilities)
    for i in range(parents_num):
        candidates = np.random.choice(indices, k, False)
        sorted_candidates = candidates[np.argsort(fitness[candidates])]
        result[i] = np.random.choice(sorted_candidates, p=probabilities)

    return result


def k_point_crossover(parents, k=2):
    np.random.shuffle(parents)
    a, b = np.array_split(parents, 2)

    offsprings = []
    for parent_a, parent_b in zip(a, b):
        cross_points = np.random.randint(1, parent_a.size, k)

        for cross_point in cross_points:
            parent_a = np.append(parent_a[:cross_point], parent_b[cross_point:])
            parent_b = np.append(parent_b[:cross_point], parent_a[cross_point:])

        offsprings.append(parent_a)
        offsprings.append(parent_b)
    return np.array(offsprings)


def truncation_replacement_strategy(old_candidates, old_fitness, new_candidates, new_fitness):
    candidates = np.concatenate((old_candidates, new_candidates))
    fitness = np.concatenate((old_fitness, new_fitness))
    indices = np.argsort(fitness)[:old_candidates.shape[0]]
    return candidates[indices], fitness[indices]


def evolutionary_algorithm(obj_f, population_size, generations):
    ea = EA()

    ea.set_initialization(lambda x: initialization(x, lambda: init_real_chromosome(10)))
    ea.set_selection(tournament_selection)
    ea.set_mutation(normal_distribution_addition)
    ea.set_crossover(k_point_crossover, 0.5)
    ea.set_replacement_strategy(truncation_replacement_strategy)

    ea.initialize(obj_f, population_size)
    ea.run(generations)

    return ea.get_best_result(), ea.get_best_fitness()


if __name__ == "__main__":
    print(evolutionary_algorithm(sphere, 100, 1000))
