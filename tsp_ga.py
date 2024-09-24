import random
import numpy as np
from itertools import permutations
from typing import List
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, cities_list: List[str], population_size: int, generations_size: int, 
                 crossover_rate: float, mutation_rate: float):
        self.cities_list: List[str] = cities_list
        self.population_size: int = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations_size = generations_size
        self.best_solution: List[List[str]] = []

    def initial_population(self) -> List[List[str]]:
        """
        Generating the initial population of cities <randomly> selected from all 
        possible <permutations> of the given cities.
        
        Input (collected from the class attribute):
        1- Cities list 
        2- Number of population 
        
        Output:
        Populated list of cities as the initial population.
        """
        population_perms = []
        # Use the cities_list from the class attribute
        possible_perms = list(permutations(self.cities_list))

        # Sample the population size
        random_ids = random.sample(range(0, len(possible_perms)), population_size)

        for i in random_ids:
            population_perms.append(list(possible_perms[i]))

        return population_perms
    
    def dist_two_cities(self, city_1: str, city_2: str) -> float:
        """
        Calculate the Euclidean distance between two cities, which is the length of the edge connecting the two cities together.

        Input:
        1- city_1 (str): The name of the first city.
        2- city_2 (str): The name of the second city.

        Output:
        float: The Euclidean distance between the two cities.

        Procedure:
        - The function retrieves the coordinates (latitude, longitude, or other) of each city from a dictionary `city_coords`.
        - Converts the coordinates of both cities into NumPy arrays to perform element-wise operations.
        - Calculates the squared difference between the corresponding coordinates (e.g., latitude difference squared and longitude difference squared).
        - Sums these squared differences.
        - Takes the square root of the sum to get the Euclidean distance, which is the straight-line distance between the two points in the coordinate space.
        """

        # Retrieve the coordinates of the first city from the city_coords dictionary
        city_1_coords = city_coords[city_1]

        # Retrieve the coordinates of the second city from the city_coords dictionary
        city_2_coords = city_coords[city_2]

        # Convert both coordinate lists to NumPy arrays and calculate the Euclidean distance
        # The formula used here is: sqrt((x1 - x2)^2 + (y1 - y2)^2) in 2D or its generalization in higher dimensions
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))
    
    def total_dist_individual(self, individual: list) -> float:
        """
        Calculate the total distance traveled by a individual, which represents a possible solution (one permutation of cities).

        Input:
        1- individual (list of str): A list of city names representing the order of travel (one permutation of cities).

        Output:
        float: The total distance traveled by visiting all cities in the order specified by the individual and returning to the starting city.

        Procedure:
        - The function iterates through each city in the individual.
        - It calculates the distance between consecutive cities in the individual.
        - If the current city is the last city in the individual, it adds the distance from this city back to the first city (to complete the cycle).
        - The total distance is accumulated and returned as the final result.
        """

        total_dist = 0  # Initialize total distance to 0

        # Loop through each city in the individual to calculate the distance between consecutive cities
        for i in range(0, len(individual)):
            if i == len(individual) - 1:
                # If it's the last city, add the distance from the last city back to the first city
                total_dist += self.dist_two_cities(individual[i], individual[0])
            else:
                # Otherwise, add the distance between the current city and the next city
                total_dist += self.dist_two_cities(individual[i], individual[i+1])
        
        return total_dist  # Return the total distance traveled
    
    def fitness_prob(self, population: list) -> np.ndarray:
        """
        Calculate the fitness probabilities for a population of individuals.

        Input:
        1- population (list of lists): A list of individuals, where each individual is a list of city names (one permutation).

        Output:
        np.ndarray: An array of fitness probabilities corresponding to each individual in the population.

        This function calculates the fitness of each individual based on the total distance traveled. The fitness is inversely related to the total distance (shorter distances are better).
        The fitness values are normalized to create probabilities, which can be used for selection in the next step of the genetic algorithm.
        """

        # Initialize an empty list to store the total distance for each individual in the population
        total_dist_all_individuals = []

        # Loop through each individual in the population and calculate its total distance
        for i in range(len(population)):
            total_dist_all_individuals.append(self.total_dist_individual(population[i]))

        # Convert the list of distances to a NumPy array for vectorized operations
        total_dist_all_individuals = np.array(total_dist_all_individuals)

        # Find the maximum total distance in the population to calculate relative fitness
        max_population_cost = max(total_dist_all_individuals)

        # Calculate the fitness for each individual
        # Fitness is defined as the difference between the max distance and the individual's total distance
        # A higher total distance results in a lower fitness (shorter distance is better)
        population_fitness = max_population_cost - total_dist_all_individuals

        # Calculate the sum of fitness values to normalize them into probabilities
        population_fitness_sum = np.sum(population_fitness)

        # Normalize the fitness values to probabilities
        # Each fitness probability represents the likelihood of selecting that individual in the next generation
        population_fitness_probs = population_fitness / population_fitness_sum

        # Return the fitness probabilities for the population
        return population_fitness_probs
    
    def roulette_wheel(self, population: list, fitness_probs: np.ndarray):
        """
        Implement a selection strategy based on proportionate roulette wheel selection.

        Input:
        1- population (list): The list of individuals (possible solutions).
        2- fitness_probs (np.ndarray): The fitness probabilities associated with each individual in the population.

        Output:
        The selected individual from the population.
        """
        
        # Step 1: Calculate the cumulative sum of fitness probabilities 
        # that represent the "segments" of the roulette wheel.
        population_fitness_probs_cumsum = fitness_probs.cumsum()

        # Step 2: Generate a random float between 0 and 1
        # Compare each cumulative sum with a random number uniformly drawn from [0,1)
        bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)

        # Step 3: Find the last "True" index in the boolean array
        # This index corresponds to the selected individual
        selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1

        # Step 4: Return the individual from the population corresponding to the selected index
        return population[selected_individual_index]
    
    def crossover(self, parent_1: list, parent_2: list):
        """
        Perform a single-point crossover between two parent individuals.

        Input:
        1- parent_1 (list): First parent individual (a list of city names).
        2- parent_2 (list): Second parent individual (a list of city names).

        Output:
        Tuple: Two offspring individuals, each a combination of genetic material from both parents.

        Procedure:
        - A random crossover point is chosen.
        - The first part of each offspring is taken from one parent up to the crossover point.
        - The remaining cities are taken from the other parent, preserving the order and avoiding duplicates.
        """
        # Define the cut point, where crossover will happen
        # The cut point is randomly chosen between the second and the second-last city
        cut = random.randint(1, len(parent_1) - 1)  # Get an integer cut point between 1 and n-1

        # Initialize offspring as empty lists
        child_1 = parent_1[:cut]  # First part of offspring 1 is copied from parent 1 up to the cut point
        child_2 = parent_2[:cut]  # First part of offspring 2 is copied from parent 2 up to the cut point

        # Append remaining cities from the other parent, while avoiding duplicates
        child_1 += [city for city in parent_2 if city not in child_1]
        child_2 += [city for city in parent_1 if city not in child_2]

        # Return the two offspring generated by crossover
        return child_1, child_2
    
    def mutation(self, child: list) -> list:
        """
        Perform mutation on a single child by randomly swapping two cities (genes).

        Input:
        1- child (list): A list of city names representing the individual.

        Output:
        1- mutated_child (list): The child after mutation (two cities swapped).

        Procedure:
        - Two random indices are selected in the individual (child).
        - The cities at these indices are swapped to introduce variation.
        """
        # Randomly select two distinct indices to swap cities in the child
        index_1 = random.randint(0, len(child) - 1)  # Random index between 0 and len(child)-1
        index_2 = random.randint(0, len(child) - 1)

        # Ensure that the two indices are not the same
        while index_1 == index_2:
            index_2 = random.randint(0, len(child) - 1)

        # Swap the cities at index_1 and index_2
        child[index_1], child[index_2] = child[index_2], child[index_1]

        # Return the mutated child
        return child
    
    def visualize_progress(self, best_path, distance, generation):
        x_shortest, y_shortest = [], []
        for city in best_path:
            x_value, y_value = city_coords[city]
            x_shortest.append(x_value)
            y_shortest.append(y_value)

        x_shortest.append(x_shortest[0])
        y_shortest.append(y_shortest[0])

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label=f'Best Route (Gen: {generation})', linewidth=2.5)
        plt.legend()

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

        plt.title(f"TSP Best Route Using GA (Generation {generation})")
        plt.suptitle(f"Total Distance: {round(distance, 3)}")
        
        for i, txt in enumerate(best_path):
            ax.annotate(str(i+1) + "- " + txt, (x_shortest[i], y_shortest[i]), fontsize= 12)

        fig.set_size_inches(10, 8)
        # plt.grid(color='k', linestyle='dotted')
        plt.savefig('solution.png')
        plt.show()

    def run_ga(self):
        # Step 1: Initialize population
        # Generate initial population of possible solutions (city permutations).
        population = self.initial_population()  
        
        # Step 2: Calculate fitness probabilities
        fitness_probs = self.fitness_prob(population)
        
        # Step 3: Select parents for crossover
        parents_list = []

        # Loop to select a number of parents proportional to the crossover rate
        for i in range(0, int(self.crossover_rate * self.population_size)):
            # Select a best parent using roulette wheel selection
            parents_list.append(self.roulette_wheel(population, fitness_probs))
        
        # Step 4: Generate new children (next generation) through crossover
        child_list = []
        for i in range(0, len(parents_list), 2):
            # Perform crossover between pairs of parents to generate two children
            child_1, child_2 = self.crossover(parents_list[i], parents_list[i + 1])

            mutate_threashold = random.random()
            if(mutate_threashold > (1 - self.mutation_rate)):
                child_1 = self.mutation(child_1)

            mutate_threashold = random.random()
            if(mutate_threashold > (1 - self.mutation_rate)):
                child_2 = self.mutation(child_2)

            child_list.append(child_1)
            child_list.append(child_2)

        # Combine parents and children into the next generation population
        next_generation = parents_list + child_list

        # Step 5: Recalculate fitness probabilities for the new generation
        # Evaluate the fitness of each individual in the new generation.
        fitness_probs = self.fitness_prob(next_generation)
        
        # Sort the individuals based on their fitness, from most fit to least fit.
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        
        # Select the top 'population_size' individuals with the best fitness to form the next population.
        best_fitness_indices = sorted_fitness_indices[0:self.population_size]

        # These best individuals will be used to start the next generation of the genetic algorithm.
        for i in best_fitness_indices:
            self.best_solution.append(next_generation[i])


        for i in range(0, self.generations_size):
            print(i)
            fitness_probs = self.fitness_prob(self.best_solution)

            parents_list = []
            for i in range(0, int(self.crossover_rate * self.population_size)):
                parents_list.append(self.roulette_wheel(self.best_solution, fitness_probs))
            
            child_list = []
            for i in range(0, len(parents_list), 2):
                child_1, child_2 = self.crossover(parents_list[i], parents_list[i + 1])

                mutate_threashold = random.random()
                if(mutate_threashold > (1 - self.mutation_rate)):
                    child_1 = self.mutation(child_1)

                mutate_threashold = random.random()
                if(mutate_threashold > (1 - self.mutation_rate)):
                    child_2 = self.mutation(child_2)

                child_list.append(child_1)
                child_list.append(child_2)

            next_generation = parents_list + child_list
            fitness_probs = self.fitness_prob(next_generation)
            sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
            best_fitness_indices = sorted_fitness_indices[0:int(0.8 * population_size)]

            for i in best_fitness_indices:
                self.best_solution.append(next_generation[i])

            old_population_indices = [random.randint(0, (population_size - 1)) for j in range(int(0.2 * population_size))]
            for i in old_population_indices:
                self.best_solution.append(population[i])
            
            random.shuffle(self.best_solution)
            # Calculate the current best path and distance
            total_distances = [self.total_dist_individual(ind) for ind in self.best_solution]
            best_index = np.argmin(total_distances)
            best_distance = total_distances[best_index]
            best_path = self.best_solution[best_index]

        # Visualize the progress
        self.visualize_progress(best_path, best_distance, i)
        
        return self.best_solution



population_size = 250
generations_size = 200
crossover_rate = 0.2
mutation_rate = 0.2

x = [15,10,8,5,3,0]
y = [5,3.5,2.5,1.5,2.2,2.75]

sweden_cities = ['Stockholm', 'Norrköping', 'Linköping', 'Jönköping', 'Borås', 'Gothenburg']
city_coords = dict(zip(sweden_cities, zip(x, y)))

ga = GeneticAlgorithm(sweden_cities, population_size, generations_size, crossover_rate, mutation_rate)
best_solution = ga.run_ga()