import random
import numpy as np


 # -----------------------------------------------
 # Step 1: Define the cities and distance matrix
 # -----------------------------------------------

cities = [
    "UH College Lane campus", "UH de Havilland Campus", "The Galleria",
    "ASDA superstore", "Hatfield House", "St Albans Cathedral",
    "Mill Green Museum", "Hatfield swim center", "Heartwood Forest", "Verulamium Park"]

num_cities = len(cities)

# Distance matrix (symmetric, in km - These are real distances so if you ever visit UH in England you can go for a bike ride using this)
distances = np.array([
    [0, 1.6, 1, 1.6, 3.4, 9.3, 4.4, 1.5, 8.9, 10.1],
    [1.6, 0, 0.55, 1.8, 4, 9.2, 4.1, 1.7, 7.9, 9.4],
    [1, 0.55, 0, 1.2, 3.7, 9, 3.5, 1.1, 8.4, 9.2],
    [1.6, 1.8, 1.2, 0, 2.2, 10, 3.1, 0.16, 9.5, 10.2],
    [3.4, 4, 3.7, 2.2, 0, 12.2, 2.4, 2, 11.7, 12.4],
    [9.3, 9.2, 9, 10, 12.2, 0, 11.8, 9.3, 5.7, 1.1],
    [4.4, 4.1, 3.5, 3.1, 2.4, 11.8, 0, 3, 11.8, 12.5],
    [1.5, 1.7, 1.1, 0.16, 2, 9.3, 3, 0, 9.4, 10.1],
    [8.9, 7.9, 8.4, 9.5, 11.7, 5.7, 11.8, 9.4, 0, 6.5],
    [10.1, 9.4, 9.2, 10.2, 12.4, 1.1, 12.5, 10.1, 6.5, 0]
])

# ----------------------------------------------
# Step 2: Define distance and fitness functions
# ----------------------------------------------

# Compute the total distance of a given route


def calculate_total_distance(route, distances):
    total = 0
    for i in range(len(route)):
        start = route[i]
        end = route[(i + 1) % len(
            route)]  # much like a circular queue data structure you can use this wrap around feature using modulo. When i is at the last index, (i + 1) % length becomes 0, so it wraps around to the start, creating a complete loop.
        total += distances[start][end]
    return total


# Fitness is the inverse of distance (higher = better)
def calculate_fitness(route, distances):
    return 1 / (calculate_total_distance(route,
                                         distances) + 1e-8)  # 1e-8 is to ensure we dont divide by zero (0.00000001)


# -----------------------------------------
# Step 3: Generate initial population
# -----------------------------------------

def generate_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population


# -----------------------------------------
# Step 4: Tournament Selection
# -----------------------------------------

def tournament_selection(population, distances, tournament_size=5):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda route: calculate_total_distance(route, distances))
    return selected[0]  # Best route from the tournament


# -----------------------------------------
# Step 5: Order Crossover (OX)
# -----------------------------------------

def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    # Copy a slice from parent1
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    # Fill the rest with cities from parent2 (in order, skipping duplicates)
    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]

    return child


# -----------------------------------------
# Step 6: Swap Mutation
# -----------------------------------------

def mutate(route, mutation_rate=0.1):
    new_route = route.copy()
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


# -----------------------------------------
# Step 7: Run the Genetic Algorithm
# -----------------------------------------

def run_genetic_algorithm(
        distances, pop_size=100, generations=500, mutation_rate=0.1, tournament_size=5
):
    population = generate_population(pop_size, len(distances))

    best_route = min(population, key=lambda r: calculate_total_distance(r, distances))
    best_distance = calculate_total_distance(best_route, distances)

    for gen in range(generations):
        new_population = []

        for _ in range(pop_size):
            parent1 = tournament_selection(population, distances, tournament_size)
            parent2 = tournament_selection(population, distances, tournament_size)

            child = order_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population

        current_best = min(population, key=lambda r: calculate_total_distance(r, distances))
        current_distance = calculate_total_distance(current_best, distances)

        if current_distance < best_distance:
            best_route = current_best
            best_distance = current_distance

        if gen % 50 == 0 or gen == generations - 1:
            print(f"Generation {gen}: Best Distance = {best_distance:.2f}")

    return best_route, best_distance


# ---------------------------------------------
# Step 8: Run the algorithm and display result
# ---------------------------------------------

best_route, best_distance = run_genetic_algorithm(distances, generations=500, mutation_rate=0.1)

# Show results
print("\n Best Route (indices):", best_route)
print(f" Best Distance: {best_distance:.2f} km")

print("\n Route with city names:")
for i in best_route:
    print("-", cities[i])
print("-", cities[best_route[0]], "(return to start)")
