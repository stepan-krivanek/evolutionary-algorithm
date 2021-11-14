
import numpy as np
import time


class EvolutionAlgorithm:
    def __init__(self):
        self.obj_f = None
        self.initialization_f = None
        self.selection_f = None
        self.mutation_f = None
        self.mutation_prob = None
        self.crossover_f = None
        self.crossover_prob = 1
        self.replacement_strategy_f = None
        self.best_result = None
        self.best_fitness = float("inf")
        self.current_population = None

    def set_initialization(self, initialization_function):
        self.initialization_f = initialization_function

    def set_selection(self, selection_function):
        self.selection_f = selection_function

    def set_mutation(self, mutation_function, mutation_prob=None):
        self.mutation_f = mutation_function
        self.mutation_prob = mutation_prob

    def set_crossover(self, crossover_function, crossover_prob=0.5):
        self.crossover_f = crossover_function
        self.crossover_prob = crossover_prob

    def set_replacement_strategy(self, replacement_strategy_function):
        self.replacement_strategy_f = replacement_strategy_function

    def initialize(self, obj_function, population_size):
        self.obj_f = obj_function
        self.current_population = self.initialization_f(population_size)
        self.update_best_result(self.current_population, np.apply_along_axis(obj_function, 1, self.current_population))

    def update_best_result(self, population, fitness):
        index = np.argmin(fitness)
        if fitness[index] < self.best_fitness:
            self.best_result = population[index]
            self.best_fitness = fitness[index]

    def run(self, generations):
        old_fitness = np.apply_along_axis(self.obj_f, 1, self.current_population)

        while generations > 0:
            # Selection
            new_candidates = self.current_population[self.selection_f(self.current_population.shape[0], old_fitness)]

            # Crossover
            if np.random.rand() < self.crossover_prob:
                new_candidates = self.crossover_f(new_candidates)

            # Mutation
            if self.mutation_prob is None:
                new_candidates = np.apply_along_axis(self.mutation_f, 1, new_candidates)
            else:
                new_candidates = np.apply_along_axis(lambda x: self.mutation_f(x, self.mutation_prob), 1, new_candidates)

            # Evaluation
            new_fitness = np.apply_along_axis(self.obj_f, 1, new_candidates)
            self.update_best_result(new_candidates, new_fitness)

            # Replacement
            self.current_population, old_fitness = self.replacement_strategy_f(
                self.current_population, old_fitness, new_candidates, new_fitness
            )

            if generations % 100 == 0:
                print(generations)
            generations -= 1

    def get_best_result(self):
        return self.best_result

    def get_best_fitness(self):
        return self.best_fitness
