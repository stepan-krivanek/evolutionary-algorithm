
import numpy as np

################################################
#              fitness functions               #
################################################


def one_max(chromosome):
    return np.sum(chromosome)


def labs(chromosome):
    new_chrom = [-1 if x == 0 else 1 for x in chromosome]
    N = len(new_chrom)
    correlations = [sum([new_chrom[i] * new_chrom[i+k] for i in range(0, N - k)]) for k in range(N)]
    return sum([x**2 for x in correlations[1:]])


def sphere(chromosome, offset=None):
    if offset is None:
        offset = np.ones(chromosome.size)
    return np.sum((chromosome - offset) ** 2)


def rosenbrock(chromosome):
    return np.sum(100 * (chromosome[1:] - chromosome[:-1] ** 2) ** 2 + (1 - chromosome[:-1]) ** 2)


def linear(chromosome, a0=1, ai=None):
    if ai is None:
        ai = np.ones(chromosome.size)

    return a0 + np.sum(ai * chromosome)


def step(chromosome, a0=1, ai=None):
    if ai is None:
        ai = np.ones(chromosome.size)

    return a0 + np.sum(np.floor(ai * chromosome))


def rastrigin(chromosome):
    return 10 * chromosome.size + np.sum(chromosome ** 2 - 10 * np.cos(2 * np.pi * chromosome))


def griewank(chromosome):
    return 1 + np.sum(chromosome ** 2) / 4000 - np.prod(np.cos(chromosome / np.sqrt(np.arange(1, chromosome.size + 1))))


def schwefel(chromosome):
    return - np.sum(chromosome * np.sin(np.sqrt(np.abs(chromosome))))


################################################
#         perturbation functions               #
################################################


def bits_inversion(chromosome, probability=0.1):
    return chromosome ^ np.random.binomial(1, probability, chromosome.size)


def normal_distribution_addition(chromosome, variation=2):
    return chromosome + np.random.normal(0, variation, chromosome.size)


def cauchy_distribution_addition(chromosome):
    return chromosome + np.random.standard_cauchy(chromosome.size)

################################################
#                  others                      #
################################################


def bin2realXD(chromosome, lu_bound=([0], [1])):
    lower_bounds = lu_bound[0]
    upper_bounds = lu_bound[1]

    if not float.is_integer(chromosome.size / len(lower_bounds)):
        print("The binary vector length is not divisible by the dimensionality of the target vector space.")
        return None

    chunks = np.array_split(chromosome, len(lower_bounds))
    max_num = 2 ** chunks[0].size - 1
    result = []
    for i, chunk in enumerate(chunks):
        lb = lower_bounds[i]
        ub = upper_bounds[i]
        num = int("".join(str(i) for i in chunk), 2)
        real_num = num / max_num
        result.append(real_num * (ub - lb) + lb)

    return result


# (1+1)-ES with 1/5 rule
def ES(init_chromosome, objective_f, iterations):
    best_chromosome = init_chromosome
    best_fitness = objective_f(best_chromosome)

    i = 0
    sigma = 1
    while i < iterations:
        chromosome = best_chromosome + sigma * np.random.normal(0, 2, init_chromosome.size)
        fitness = objective_f(chromosome)
        b = fitness < best_fitness
        sigma *= np.exp(int(b) - 1/5) ** (1 / init_chromosome.size)

        if b:
            best_fitness = fitness
            best_chromosome = chromosome
        i += 1

    return best_chromosome, best_fitness


def local_search_first_improving(init_chromosome, objective_f, perturbation_f, iterations):
    best_chromosome = init_chromosome
    best_fitness = objective_f(best_chromosome)
    i = 0

    while i < iterations:
        chromosome = perturbation_f(best_chromosome)
        fitness = objective_f(chromosome)
        if fitness < best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome
        i += 1

    return best_chromosome, best_fitness


def fitness_function_test(test_file, test_function):
    f = open(test_file, "r")
    lines = f.readlines()
    f.close()

    for line in lines:
        if line.startswith('#'):
            continue
        chromosome = np.array(list(map(float, line.split(':')[0].split())))
        print(f'{chromosome} : {test_function(chromosome)}')


if __name__ == "__main__":
    # fitness_function_test("./rastrigin.txt", rastrigin)

    init_chromosome = np.random.normal(0, 2, 10)
    print(ES(init_chromosome, sphere, 1000))
