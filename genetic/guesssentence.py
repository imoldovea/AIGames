# guesssentence.py

import os
import random
import string

import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from tqdm import tqdm

import utils

OUTPUT_FOLDER = "output"
HTML_FILE_PATH = os.path.join(OUTPUT_FOLDER, "guesssentence.html")

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize the HTML file
with open(HTML_FILE_PATH, "w", encoding="utf-8") as f:
    f.write("<html><head><title>GA Progress</title></head><body>")
    f.write("<h1>Genetic Algorithm Progress</h1>")

# 🎯 Target sentence we want the GA to evolve
TARGET = "The quick brown fox jumps over the lazy dog "
# Allowed characters for mutation and initial generation
CHARS = string.ascii_letters + string.punctuation + " "

# 🧬 Genetic Algorithm hyperparameters
POP_SIZE = 300  # Number of individuals in the population
CX_PROB = 2  # Crossover probability
MUT_PROB = 0.2  # Mutation probability
N_GEN = 1000  # Maximum number of generations
IND_SIZE = len(TARGET)  # Length of each individual (same as target sentence)

# 🧱 Define the fitness function and individual type in DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # We want to maximize fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

# 🛠️ Toolbox setup: defines how to create individuals and population
toolbox = base.Toolbox()
toolbox.register("attr_char", lambda: random.choice(CHARS))  # Random character
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_char, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 📈 Fitness function: counts how many characters match the target
def eval_fitness(individual):
    return sum(individual[i] == TARGET[i] for i in range(IND_SIZE)),


# 🧬 Register genetic operators
toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover


# Custom mutation function that replaces one character in the individual
def mutate_char(individual, indpb=0.05):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(CHARS)
    return individual,


# Register the custom mutation operator
toolbox.register("mutate", mutate_char)
toolbox.register("select", tools.selTournament, tournsize=3)


# 🔁 Decode an individual (list of characters) into a string
def decode(ind):
    return ''.join(ind)


# Function to color correct letters in HTML
def color_correct_char(individual_str, target_str):
    html_result = ""
    for ind_char, tgt_char in zip(individual_str, target_str):
        if ind_char == tgt_char:
            html_result += f'<span style="color:green;">{ind_char}</span>'
        else:
            html_result += ind_char
    return html_result


def append_to_html(content):
    with open(HTML_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(content)


# 🎥 Main GA loop
def run_ga():
    # Step 1: Initialize a population of random individuals
    pop = toolbox.population(n=POP_SIZE)
    # Step 2: Create a Hall of Fame to keep the best individual ever seen
    hof = tools.HallOfFame(1)
    # Step 3: Set up statistics tracking (we'll record max and average fitness)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", max)
    stats.register("avg", lambda x: sum(x) / len(x))

    # This list will store (generation, max_fitness, avg_fitness) tuples
    fitnesses = []

    # Step 4: Main GA loop over generations
    for gen in tqdm(range(N_GEN), desc="GA Generations"):
        # 4a: Apply crossover and mutation to create new offspring
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB)

        # 4b: Evaluate fitness of each new individual
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # 4c: Select the next generation from the offspring
        pop = toolbox.select(offspring, k=len(offspring))
        # 4d: Update Hall of Fame with the best individuals in the new population
        hof.update(pop)

        # 4e: Identify the top two individuals by fitness
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        top2 = sorted_pop[:2]

        # 4f: Append the top two to the HTML log with correct letters colored
        append_to_html(f"<h2>Generation {gen + 1}</h2>")
        for i, individual in enumerate(top2, start=1):
            decoded_str = decode(individual)
            colored_str = color_correct_char(decoded_str, TARGET)
            append_to_html(
                f"<p>{i}: {colored_str} (Fitness: {individual.fitness.values[0]})</p>"
            )

        # 4g: Record statistics for plotting later
        record = stats.compile(pop)
        fitnesses.append((gen, record["max"], record["avg"]))

        # 4h: Early exit if we've matched the target exactly
        if hof[0].fitness.values[0] == IND_SIZE:
            break

    # Step 5: Output the best result and close the HTML file
    print(f"✅ Best sentence found in {gen + 1} generations: {decode(hof[0])}")
    with open(HTML_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("</body></html>")

    # Step 6: Return the history of fitness values for plotting
    return fitnesses


# 📊 Plot fitness over generations
def plot_fitness(fitnesses):
    gens, max_vals, avg_vals = zip(*fitnesses)
    plt.plot(gens, max_vals, label="Max Fitness")
    plt.plot(gens, avg_vals, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.title("GA Progress Toward Target Sentence")
    plt.show()


# 🚀 Execute the algorithm and show results
if __name__ == "__main__":
    utils.clean_output_folder()
    utils.setup_logging()
    # Make sure to create the output folder if it doesn’t exist at the start:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Optionally, initialize the HTML file
    with open(os.path.join(HTML_FILE_PATH), "w", encoding="utf-8") as f:
        f.write("<html><body><h1>GA Progress</h1>")
    fitness_records = run_ga()

    # Save the fitness data to an HTML file
    with open(HTML_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Fitness History</title></head><body>")
        f.write("<h1>Fitness over Generations</h1>")
        f.write("<table border='1'><tr><th>Generation</th><th>Max Fitness</th><th>Average Fitness</th></tr>")
        for gen, max_fit, avg_fit in fitness_records:
            f.write(f"<tr><td>{gen}</td><td>{max_fit}</td><td>{avg_fit}</td></tr>")
        f.write("</table></body></html>")
    plot_fitness(fitness_records)


def color_correct_char_terminal(individual_str, target_str):
    result = ""
    for ind_char, tgt_char in zip(individual_str, target_str):
        if ind_char == tgt_char:
            result += f"\033[92m{ind_char}\033[0m"  # Green color
        else:
            result += ind_char
    return result
