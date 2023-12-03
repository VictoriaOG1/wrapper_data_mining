import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from skrebate import ReliefF
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import time


# Fitness function using ReliefF for feature selection
# Modify the evaluate function to accept a tuple (individual, X_data, X_scaled, y_data)
def evaluate_svm(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Create and train an SVM classifier
    svm = SVC(probability=True)
    svm.fit(x_train, y_train)

    # Predict probabilities for the test set
    y_probs = svm.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_rf(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Create and train an SVM classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)

    # Predict probabilities for the test set
    y_probs = rf.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_ann(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Create and train an SVM classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(84, 84),
        activation="relu",
        max_iter=1000,
        random_state=42,
        solver="adam",
    )
    mlp.out_activation_ = "sigmoid"

    mlp.fit(x_train, y_train)

    # Predict probabilities for the test set
    y_probs = mlp.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_nv(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Create and train an SVM classifier
    nv = GaussianNB()
    nv.fit(x_train, y_train)

    # Predict probabilities for the test set
    y_probs = nv.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_knn(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Create and train an SVM classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    # Predict probabilities for the test set
    y_probs = knn.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_ada(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    base_classifier = DecisionTreeClassifier(max_depth=1)

    # Create AdaBoost classifier
    adaboost_classifier = AdaBoostClassifier(
        base_classifier, n_estimators=50, random_state=42
    )

    # Train the AdaBoost classifier
    adaboost_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_probs = adaboost_classifier.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


def evaluate_ensemble(args):
    individual, X_data, X_scaled, y_data = args
    selected_features = [
        feature for feature, include in zip(X_scaled.columns, individual) if include
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled[selected_features], y_data, test_size=0.2, random_state=42
    )

    # Define base classifiers
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = GradientBoostingClassifier(random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    clf4 = LogisticRegression(random_state=42)
    clf5 = KNeighborsClassifier()

    # Create an ensemble classifier
    ensemble_clf = VotingClassifier(
        estimators=[
            ("rf", clf1),
            ("gb", clf2),
            ("svc", clf3),
            ("lr", clf4),
            ("knn", clf5),
        ],
        voting="soft",
    )

    # Fit the ensemble classifier on the training data
    ensemble_clf.fit(x_train, y_train)

    # Make probability predictions on the test data
    y_probs = ensemble_clf.predict_proba(x_test)[
        :, 1
    ]  # Use the probability for the positive class

    # Evaluate the performance using AUC
    auc_score = roc_auc_score(y_test, y_probs)

    return (auc_score,)


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

    data = data[~data["Type"].isin(["REGIONAL", "HB", "ICEQUAKE"])]

    output_file = open(output_file_name, "w")

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
    if type == "svm":
        toolbox.register("evaluate", evaluate_svm)
    elif type == "rf":
        toolbox.register("evaluate", evaluate_rf)
    elif type == "ann":
        toolbox.register("evaluate", evaluate_ann)
    elif type == "nv":
        toolbox.register("evaluate", evaluate_nv)
    elif type == "ada":
        toolbox.register("evaluate", evaluate_ada)
    elif type == "ensemble":
        toolbox.register("evaluate", evaluate_ensemble)
    else:
        toolbox.register("evaluate", evaluate_knn)

    # Register the crossover operator
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxTwoPoint)

    # Register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

    # Register the selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selRoulette)

    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)

    # Define probabilities of crossing and mutating
    probab_crossing, probab_mutating = 0.4, 0.3

    num_processes = 8  # Adjust this based on your system's capabilities
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
            random_cross = random.random()

            if random_cross < probab_crossing:
                toolbox.mate(child1, child2)

                # Delete the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
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
    output_file.write(f"Selected features:  {selected_features}")

    output_file.close()
