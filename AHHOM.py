import random
import numpy as np

class AEHOM:
    def __init__(self, num_clans, num_dimensions, lower_limit, upper_limit):
        self.num_clans = num_clans
        self.num_dimensions = num_dimensions
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def initialize_population(self, population_size):
        population = []
        for _ in range(population_size):
            clan_positions = []
            for _ in range(self.num_clans):
                clan_position = np.random.uniform(self.lower_limit, self.upper_limit, self.num_dimensions)
                clan_positions.append(clan_position)
            population.append(clan_positions)
        return population

    def evaluate_fitness(self, position):
        # Placeholder for fitness evaluation function
        return np.sum(position)

    def update_position(self, current_position, optimal_solution, matriarch_impact):
        updated_position = []
        for clan_position, optimal_clan_solution in zip(current_position, optimal_solution):
            updated_clan_position = optimal_clan_solution + matriarch_impact * (clan_position - optimal_clan_solution) + np.random.uniform() * (self.upper_limit - self.lower_limit)
            updated_position.append(updated_clan_position)
        return updated_position

    def revise_position(self, current_position, matriarch_influence):
        revised_position = []
        for clan_position in current_position:
            revised_clan_position = matriarch_influence * clan_position
            revised_position.append(revised_clan_position)
        return revised_position

    def conduct_exploratory_activities(self, current_position, min_search_limit, max_search_limit, k):
        new_position = []
        for clan_position in current_position:
            new_clan_position = np.maximum(np.minimum(clan_position + k * (max_search_limit - min_search_limit) * np.random.uniform() * (2 * np.random.randint(0, 2, size=clan_position.shape) - 1), self.upper_limit), self.lower_limit)
            new_position.append(new_clan_position)
        return new_position

    def crossover(self, parent1, parent2):
        x1 = parent1[2]  # Assuming this is Px+1,En3 from Equation (25)
        x2 = x1 + parent2[1]  # Assuming this is Px+1,En2 from Equation (26)
        return x1, x2

    def mutate(self, position, mutation_rate):
        mutated_position = []
        for clan_position in position:
            if np.random.uniform() < mutation_rate:
                mutation_index = np.random.randint(0, len(clan_position))
                clan_position[mutation_index] = np.random.uniform(self.lower_limit, self.upper_limit)
            mutated_position.append(clan_position)
        return mutated_position

    def select_parents(self, population):
        fitness_scores = [self.evaluate_fitness(solution) for solution in population]
        sorted_indices = np.argsort(fitness_scores)
        # Select the top solutions as parents for crossover
        parent_indices = sorted_indices[:len(population) // 2]
        return [population[i] for i in parent_indices]

    def optimize(self, population_size, max_iterations, mutation_rate, k, matriarch_impact, matriarch_influence, min_search_limit, max_search_limit, termination_condition):
        population = self.initialize_population(population_size)
        iteration = 0
        while iteration < max_iterations:
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(solution) for solution in population]

            # Log best solution and fitness
            best_solution_index = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_solution_index]
            best_solution = population[best_solution_index]
            print(f"Iteration {iteration}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")

            # Check termination condition
            if termination_condition(best_fitness):
                break

            # Select parents for crossover
            parents = self.select_parents(population)

            # Perform crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(offspring1, mutation_rate))
                offspring.append(self.mutate(offspring2, mutation_rate))

            # Replace old solutions with offspring
            population = offspring
            iteration += 1

        return best_solution, best_fitness

# Example termination condition: stop when fitness reaches a certain threshold
def termination_condition(fitness):
    return fitness >= 1000

# Example usage:
num_clans = 5
num_dimensions = 10
lower_limit = -10
upper_limit = 10
population_size = 50
max_iterations = 100
mutation_rate = 0.1
k = 0.1
matriarch_impact = 0.5
matriarch_influence = 0.5
min_search_limit = -10
max_search_limit = 10

aehom = AEHOM(num_clans, num_dimensions, lower_limit, upper_limit)
best_solution, best_fitness = aehom.optimize(population_size, max_iterations, mutation_rate, k, matriarch_impact, matriarch_influence, min_search_limit, max_search_limit, termination_condition)
print(f"Best Fitness = {best_fitness}, Best Solution = {best_solution}")


