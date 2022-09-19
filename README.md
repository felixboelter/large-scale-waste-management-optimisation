# **Large-scale multi-objective optimisation for sustainable waste management using Evolutionary Algorithms**
A meta-heuristic approach to sustainable waste management.

# **Installing the prerequisites**
Make sure to append the **ibmdecisionoptimization** and **conda-forge** packages to the current channels. Write this into a terminal:
```
$ conda config --append channels ibmdecisionoptimization`
$ conda config --append channels conda-forge
```

# **Installation**
Clone the repository and install the requirements file via conda.
```
$ git clone https://github.com/felixboelter/large-scale-waste-management-optimisation
$ cd large-scale-waste-management-optimisation
$ `conda create --name <env> --file requirements.txt`
```

# **Usage**
The [main.py](main.py) file includes a class which generates and solves all of the graphs using the algorithms found in the [source](src) folder.

You can change the crossover and mutation probabilities by changing the `crossover_probs` and `mutation_probs` parameters with a list of selected probabilities between 0 and 1.

# **Results**
Solutions to every solved graph should be found in a created `Results` folder. Which will include:
+ A csv file with the numeric solution of the best hypervolume solutions for each algorithm.
+ The solved graph of the best hypervolume solution for each algorithm.
+ A numpy file (.npy) containing all the solutions of the algorithms and decision values.
