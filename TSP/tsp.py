
import sys
import numpy as np
import tsplib95
import matplotlib.pyplot as plt
from EvolutionAlgorithm import EvolutionAlgorithm as EA


def candidate_init(number_of_cities):
    candidate = np.array([x+1 for x in range(number_of_cities)])
    np.random.shuffle(candidate)
    return candidate


def population_init(number_of_cities, population_size):
    return np.array([candidate_init(number_of_cities) for x in range(population_size)])


def memetic_population_init(number_of_cities, population_size, local_search_f):
    population = population_init(number_of_cities, population_size)
    return np.apply_along_axis(local_search_f, 1, population)


def calculate_fitness(candidate, weights):
    return np.sum(weights[candidate[:-1] - 1, candidate[1:] - 1])


def subseq_reverse_mutation(candidate, probability=0.5):
    if np.random.rand() < probability:
        idx_from, idx_to = np.sort(np.random.randint(0, candidate.size, 2))
        candidate = np.concatenate((candidate[:idx_from], np.flip(candidate[idx_from:idx_to]), candidate[idx_to:]))
    return candidate


def shift_mutation(candidate, probability=0.5):
    if np.random.rand() < probability:
        idx_from, idx_to = np.sort(np.random.randint(0, candidate.size, 2))
        candidate = np.insert(
            np.concatenate((candidate[:idx_from], candidate[idx_from+1:])), idx_to, candidate[idx_from]
        )
    return candidate


def swap_mutation(candidate, probability=0.5):
    candidate = np.copy(candidate)
    if np.random.rand() < probability:
        i = np.random.randint(0, candidate.size)
        j = np.random.randint(0, candidate.size)
        candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def random_selection(parents_num, fitness):
    indices = np.arange(0, fitness.size)
    return np.random.choice(indices, parents_num)


def tournament_selection(parents_num, fitness, k, p=1.0):
    result = np.empty(parents_num, dtype='int32')

    indices = np.arange(0, fitness.size)
    probabilities = np.array([p * (1 - p) ** x for x in range(k)])
    probabilities = probabilities / np.sum(probabilities)
    for i in range(parents_num):
        candidates = np.random.choice(indices, k, False)
        sorted_candidates = candidates[np.argsort(fitness[candidates])]
        result[i] = np.random.choice(sorted_candidates, p=probabilities)

    return result


def cycle_crossover(parents):
    def get_children(parent_a, parent_b):
        visited = [False] * parent_a.size
        b_to_a = [0] * parent_a.size
        for i, x in enumerate(parent_a):
            b_to_a[x - 1] = i

        cycles = []
        for i in range(parent_a.size):
            if visited[i]:
                continue
            visited[i] = True

            idx = b_to_a[parent_b[i] - 1]
            cycle = [i]
            while idx != i:
                cycle.append(idx)
                visited[idx] = True
                idx = b_to_a[parent_b[idx] - 1]

            cycles.append(cycle)

        for i, cycle in enumerate(cycles):
            if i % 2 == 1:
                continue

            for idx in cycle:
                parent_a[idx], parent_b[idx] = parent_b[idx], parent_a[idx]

        return parent_a, parent_b

    parents = np.copy(parents)
    np.random.shuffle(parents)
    a, b = np.array_split(parents, 2)

    result = []
    for parent_a, parent_b in zip(a, b):
        child_a, child_b = get_children(parent_a, parent_b)
        result.append(child_a)
        result.append(child_b)
    return np.array(result)


def order_crossover(parents):
    def get_child(parent_a, parent_b):
        x, y = np.sort(np.random.randint(0, parent_a.size, 2))
        child = parent_b[~np.in1d(parent_b, parent_a[x:y], assume_unique=True)]
        return np.concatenate((child[:x], parent_a[x:y], child[x:]))

    np.random.shuffle(parents)
    a, b = np.array_split(parents, 2)
    res1 = np.array(list(map(lambda x, y: get_child(x, y), a, b)))
    res2 = np.array(list(map(lambda x, y: get_child(y, x), a, b)))
    return np.concatenate((res1, res2))


def truncation_replacement_strategy(old_candidates, old_fitness, new_candidates, new_fitness):
    candidates = np.concatenate((old_candidates, new_candidates))
    fitness = np.concatenate((old_fitness, new_fitness))
    indices = np.argsort(fitness)[:old_candidates.shape[0]]
    return candidates[indices], fitness[indices]


def local_search_first_improving(init_chromosome, objective_f, perturbation_f, iterations):
    best_chromosome = init_chromosome
    best_fitness = objective_f(best_chromosome)

    while iterations > 0:
        chromosome = perturbation_f(best_chromosome)
        fitness = objective_f(chromosome)

        if fitness < best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome

        if iterations % 1000 == 0:
            print(f'best solution so far: {best_chromosome}')
            print(f'its fitness: {best_fitness}')

        iterations -= 1

    return best_chromosome, best_fitness


def create_weight_matrix(tsp):
    nodes = list(tsp.get_nodes())
    matrix = []

    for y in nodes:
        row = []
        for x in nodes:
            row.append(tsp.get_weight(y, x))
        matrix.append(row)

    return np.array(matrix)


def visualize_solution(tsp, solution, name=None):
    points = np.array(list(tsp.node_coords.values()))

    plt.rcParams['toolbar'] = 'None'
    plt.title("TSP solution graph")
    plt.scatter(points[:, 0], points[:, 1])

    sorted_points = points[np.argsort(solution)]
    for i in range(points.shape[0] - 1):
        x, y = sorted_points[i]
        x2, y2 = sorted_points[i + 1]
        plt.arrow(x, y, x2 - x, y2 - y, head_width=2, length_includes_head=True)

    if name is not None:
        plt.savefig(name)

    plt.show()


if __name__ == "__main__":
    TSP_FILE_PATH = "./a280.tsp"

    if len(sys.argv) > 1:
        TSP_FILE_PATH = sys.argv[1]

    tsp = tsplib95.load(TSP_FILE_PATH)
    weights = create_weight_matrix(tsp)
    num_of_cities = weights.shape[0]
    fitness_function = lambda x: calculate_fitness(x, weights)

    # Evolutionary algorithm
    POPULATION_SIZE = 100
    GENERATIONS = 10000
    VISUALIZE = True
    SAVE = False
    SAVE_DIR = TSP_FILE_PATH.removesuffix(".tsp") + "_result.txt"
    STEP = 1000  # number of iterations to run before next save or visualization

    # Change the settings of EA here
    ea = EA()
    ea.set_selection(random_selection)
    ea.set_mutation(subseq_reverse_mutation, 0.5)
    ea.set_crossover(order_crossover, 1)
    ea.set_replacement_strategy(truncation_replacement_strategy)
    ea.set_initialization(lambda x: population_init(num_of_cities, x))

    # Memetic initialization
    # local_search_f = lambda x: local_search_first_improving(
    #     x, fitness_function, lambda y: subseq_reverse_mutation(y, 1), 1000
    # )[0]
    # ea.set_initialization(lambda x: memetic_population_init(num_of_cities, x, local_search_f))

    ea.initialize(fitness_function, POPULATION_SIZE)

    solutions_history = []
    if VISUALIZE or SAVE:
        for i in range(int(GENERATIONS / STEP)):
            ea.run(STEP)
            solution, fitness = ea.get_best_result(), ea.get_best_fitness()

            solutions_history.append((solution, fitness, (i + 1) * STEP))
            print(f'best solution so far: {solution}')
            print(f'its fitness: {fitness}')

            if VISUALIZE:
                visualize_solution(tsp, solution)
    else:
        ea.run(GENERATIONS)

    if SAVE:
        print(len(solutions_history))
        f = open(SAVE_DIR, 'w')
        for solution, fitness, iterations in solutions_history:
            f.write(f'\n# {iterations} iterations\n')
            f.write(str(solution) + '\n')
            f.write(str(fitness) + '\n')
        f.close()

    solution, fitness = ea.get_best_result(), ea.get_best_fitness()
    print(f'best solution found: {solution}')
    print(f'its fitness: {fitness}')
    visualize_solution(tsp, solution)

    # Local search
    # ITERATIONS = 1000000
    #
    # solution, fitness = local_search_first_improving(
    #     solution, fitness_function, lambda x: subseq_reverse_mutation(x, 1), ITERATIONS
    # )
    # print(f'best solution found: {solution}')
    # print(f'its fitness: {fitness}')
    # visualize_solution(tsp, solution)
