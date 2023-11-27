import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from skrebate import ReliefF
from multiprocessing import Pool
import time


# Fitness function using ReliefF for feature selection
# Modify the evaluate function to accept a tuple (individual, X_data, X_scaled, y_data)
def evaluate_parallel(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_data.columns, individual) if include
    ]

    # Instantiate and fit ReliefF feature selector
    relief_selector = ReliefF(n_features_to_select=len(selected_features))
    relief_selector.fit(X_scaled[selected_features].values, y_data.values)

    # Calculate ReliefF score (sum of feature importance scores)
    relief_score = (np.sum(relief_selector.feature_importances_)) / len(
        selected_features
    )

    return (relief_score,)


def customRand(type, number):
    # Check if there's a 'last_position.txt' file to resume from
    try:
        with open("last_position" + number + ".txt", "r") as last_position_file:
            last_position = int(last_position_file.read())
    except FileNotFoundError:
        last_position = 0  # Start from the beginning if no last position is saved

    # Read numbers from the text file starting from the last position
    with open("new_random_numbers" + number + ".txt", "r") as file:
        numbers = [int(line) for line in file.readlines()]

    # Normalize the numbers to the range [0, 1]
    normalized_numbers = [number / 4294967295 for number in numbers]

    # Determine the number of elements remaining to be read
    num_elements = len(normalized_numbers)
    remaining_elements = num_elements - last_position

    if remaining_elements == 0:
        raise ValueError("No remaining elements in the file.")

    # Get the corresponding normalized number
    random_number = normalized_numbers[last_position]

    # Update the last position
    last_position = last_position + 1

    # Save the new last position to 'last_position.txt'
    with open("last_position" + number + ".txt", "w") as last_position_file:
        last_position_file.write(str(last_position))

    if type == "individual":
        # Return 0 or 1 based on the generated number
        return 0 if random_number < 0.5 else 1
    else:
        return random_number


if __name__ == "__main__":
    # Check if the output file name is provided as a command-line argument
    if len(sys.argv) < 4:
        print("Please provide the output file name as a command-line argument.")
        sys.exit(1)

    # Extract the output file name from command-line arguments
    output_file_name = sys.argv[1]
    type = sys.argv[2]
    number = sys.argv[3]

    # Load data
    with open("SeisBenchV1_v1_1.json", "r") as json_file:
        data = pd.read_json(json_file)

    output_file = open("results/" + output_file_name, "w")

    # Split data into X and y
    y_data = data.iloc[:, 0]  # Select the first column as 'y'
    X_data = data.iloc[:, 1:]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    # Convert the normalized features into a dataframe
    X_scaled = pd.DataFrame(X_scaled, columns=X_data.columns)

    # Create a fitness class
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Create an individual class
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create a toolbox
    toolbox = base.Toolbox()

    # Create a random bit generator
    if type == "quantum":
        toolbox.register("attr_bool", customRand, "individual", number)
    else:
        toolbox.register("attr_bool", random.randint, 0, 1)

    # Initialize a population
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=len(X_scaled.columns),
    )

    # Define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation operator
    toolbox.register("evaluate", evaluate_parallel)

    # Register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # Register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)

    # Register the selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)

    # Define probabilities of crossing and mutating
    probab_crossing, probab_mutating = 0.3, 0.4

    num_processes = 100  # Adjust this based on your system's capabilities
    pool = Pool(processes=num_processes)

    num_generations = 100

    # Evaluate the entire population in parallel
    fitnesses = list(
        pool.map(
            toolbox.evaluate, [(ind, X_data, X_scaled, y_data) for ind in population]
        )
    )

    # Set the fitness values of the population
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Close the pool of workers
    # pool.close()
    # pool.join()

    # Record the start time
    start_time = time.time()

    # Iterate through generations
    for g in range(num_generations):
        output_file.write(f"-- Generation {g} --\n")

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cross two individuals
            if type == "quantum":
                random_cross = customRand("normal", number)
            else:
                random_cross = random.random()

            if random_cross < probab_crossing:
                toolbox.mate(child1, child2)

                # Delete the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if type == "quantum":
                random_mut = customRand("normal", number)
            else:
                random_mut = random.random()

            if random_mut < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness in parallel
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(
            pool.map(
                toolbox.evaluate,
                [(ind, X_data, X_scaled, y_data) for ind in invalid_individuals],
            )
        )

        # Set the fitness values of the evaluated individuals
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        output_file.write(f"Evaluated {len(invalid_individuals)} individuals")

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fitnesses = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fitnesses) / length
        sum2 = sum(x * x for x in fitnesses)
        std = abs(sum2 / length - mean**2) ** 0.5

        output_file.write(f"Min = {min(fitnesses)}, Max = {max(fitnesses)}\n")
        output_file.write(f"Mean = {mean}, Std = {std}\n")

        best_individual = tools.selBest(population, k=1)[0]
        output_file.write(
            f"Best individual is {best_individual}, {best_individual.fitness.values}\n"
        )

    pool.close()
    pool.join()

    # Record the end time
    end_time = time.time()

    # Calculate the total time taken
    total_time = end_time - start_time

    # Write the total time taken to the file
    output_file.write(f"Total time taken: {total_time} seconds\n")

    best_individual = tools.selBest(population, 1)[0]
    output_file.write(f"Best individual: {best_individual}\n")
    selected_features = [
        feature
        for feature, include in zip(X_scaled.columns, best_individual)
        if include
    ]
    X_selected = X_scaled[selected_features]

    # Instantiate a random forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform stratified k-fold cross-validation
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize lists to store evaluation metrics for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    auc_list = []
    f1_list = []

    for train_index, test_index in stratified_kfold.split(X_selected, y_data):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        # Train the classifier
        rf_clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_clf.predict(X_test)
        y_probs = rf_clf.predict_proba(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
        auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

        # Append metrics to the lists
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc)
        f1_list.append(f1)

    # Calculate the mean and standard deviation of the metrics across folds
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)

    mean_precision = np.mean(precision_list)
    std_precision = np.std(precision_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    # Print the mean and standard deviation of the metrics
    output_file.write("\n Random Forest Classifier \n")
    output_file.write(f"Mean Accuracy: {mean_accuracy:.2f} (Std: {std_accuracy:.2f})\n")
    output_file.write(
        f"Mean Precision: {mean_precision:.2f} (Std: {std_precision:.2f})\n"
    )
    output_file.write(f"Mean Recall: {mean_recall:.2f} (Std: {std_recall:.2f})\n")
    output_file.write(f"Mean AUC: {mean_auc:.2f} (Std: {std_auc:.2f})\n")
    output_file.write(f"Mean F1 score: {mean_f1:.2f} (Std: {std_f1:.2f})\n")
    output_file.close()
