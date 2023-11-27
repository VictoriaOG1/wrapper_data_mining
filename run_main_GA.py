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
    number_of_runs = 20
    # Convert integers to strings
    arguments = list()
    for i in range(1, number_of_runs + 1):
        if i <= 10:
            argument2 = "quantum"
        else:
            argument2 = "classical"

        argument1 = "output" + str(i) + ".txt"

        argument3 = str(i)

        arguments.append((argument1, argument2, argument3))

    # Create a pool of worker processes (adjust the number of processes as needed)
    pool = multiprocessing.Pool(processes=number_of_runs)

    # Use the pool to execute the script with each argument in parallel
    pool.starmap(run_script_with_argument, arguments)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
