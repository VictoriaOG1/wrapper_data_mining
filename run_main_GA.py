import subprocess
import multiprocessing


def run_script_with_argument(argument1, argument2, argument3):
    subprocess.run(
        [
            "python3",
            "genetic_algorithm_parallel.py",
            argument1,
            argument2,
            argument3,
        ]
    )


if __name__ == "__main__":
    type = ["svm", "knn", "ann", "rf", "nv", "ada", "ensemble"]
    # Convert integers to strings
    arguments = list()
    for i in range(1, 8):
        argument2 = type[i - 1]

        argument1 = "output" + type[i - 1] + ".txt"

        argument3 = str(i)

        arguments.append((argument1, argument2, argument3))

    # Create a pool of worker processes (adjust the number of processes as needed)
    pool = multiprocessing.Pool(processes=7)

    # Use the pool to execute the script with each argument in parallel
    pool.starmap(run_script_with_argument, arguments)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
