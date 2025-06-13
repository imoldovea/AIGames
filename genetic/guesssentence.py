# guesssentence.py

import os
import random
import string

import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from tqdm import tqdm

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
CX_PROB = 0.5  # Crossover probability
MUT_PROB = 0.1  # Mutation probability
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
    pop = toolbox.population(n=POP_SIZE)  # Initial population
    hof = tools.HallOfFame(1)  # Store best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Track fitness over time
    stats.register("max", max)
    stats.register("avg", lambda x: sum(x) / len(x))

    fitnesses = []

    # During your generation loop, when printing the top two individuals:
    for gen in tqdm(range(N_GEN)):
        # Apply crossover and mutation
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB)
        # Evaluate new offspring
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # Select the next generation
        pop = toolbox.select(offspring, k=len(offspring))
        hof.update(pop)  # Track best

        # Get top 2 individuals
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        top2 = sorted_pop[:2]

        # Append the top 2 individuals to HTML with colored correct letters
        append_to_html(f"<h2>Generation {gen + 1}</h2>")
        for i, individual in enumerate(top2):
            decoded_str = decode(individual)
            colored_str = color_correct_char(decoded_str, TARGET)
            append_to_html(f"<p>{i + 1}: {colored_str} (Fitness: {individual.fitness.values[0]})</p>")

        # Record statistics for plotting
        record = stats.compile(pop)
        fitnesses.append((gen, record["max"], record["avg"]))

        # Stop if we perfectly match the target
        if hof[0].fitness.values[0] == IND_SIZE:
            break

    print(f"✅ Best sentence found:in {gen + 1} steps :, {decode(hof[0])}")
    # Finally, close the HTML after processing:
    # (add this at the end of run_ga after the loop)
    with open(HTML_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("</body></html>")
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
    # Make sure to create the output folder if it doesn’t exist at the start:
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Optionally, initialize the HTML file
    with open(os.path.join(OUTPUT_FOLDER, "generation_progress.html"), "w", encoding="utf-8") as f:
        f.write("<html><body><h1>GA Progress</h1>")
    fitnesses = run_ga()
    plot_fitness(fitnesses)


def color_correct_char_terminal(individual_str, target_str):
    result = ""
    for ind_char, tgt_char in zip(individual_str, target_str):
        if ind_char == tgt_char:
            result += f"\033[92m{ind_char}\033[0m"  # Green color
        else:
            result += ind_char
    return result
