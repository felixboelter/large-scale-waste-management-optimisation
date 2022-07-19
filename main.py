from parameters import Parameters
from generate_graph import Graph
from solve_baseline import Model_Baseline, Multiobjective_model
from parameters import Parameters
from heuristic import Vectorized_heuristic, Minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.factory import get_reference_directions
from pymoo.indicators.hv import Hypervolume
import numpy as np
from IPython.display import display
# The code below is solving the multiobjective problem using the heuristic and the model.
if __name__ == '__main__':
    # Used to suppress scientific notation.
    np.set_printoptions(suppress=True)
    set_seed = 0
    plot_graph = False
    verbose = False
    # Algorithms include: nsga2, nsga3, unsga3, agemoea, moead, ctaea
    algorithms = ["ctae", "nsga3", "nsga2","unsga3", "agemoea", "moead"]
    # Increasing num_of_collection_centers takes alot more time.
    if isinstance(algorithms, str): algorithm = [algorithms]
    num_of_collection_centers = 5
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=plot_graph, seed=set_seed, baseline_scaler=5)
    parameters = Parameters(RandomGraph, set_seed)
    termination = MultiObjectiveSpaceToleranceTermination(tol=0.1,n_last=30,nth_gen=5,n_max_gen=2000,n_max_evals=None)
    three_objective_problem = Vectorized_heuristic(parameters)
    ref_dirs = get_reference_directions("das-dennis", n_dim = 3, n_partitions = 20)
    result_lengths = [0]
    results_F = np.array([])
    results_X = np.array([])
    for algo in algorithms:
        minimization = Minimize(problem = three_objective_problem, population_size = len(ref_dirs) + 50, reference_directions = ref_dirs, termination=termination, verbose = verbose, algorithm=algo)
        result = minimization.minimize_heuristic()
        print(f'Time for {algo.upper()} heuristic: {result.exec_time}')
        print(result.F[:10])
        if len(results_F) == 0: 
            results_F = result.F
            results_X = result.X
        else: 
            results_F = np.vstack([results_F, result.F])
            results_X = np.vstack([results_X, result.X])
        result_lengths.append(result.F.shape[0])
    print(results_X.shape)
    parameters = Parameters(G = RandomGraph, seed=set_seed)
    model_baseline = Model_Baseline(parameters=parameters, plot_graph = plot_graph, verbose=verbose)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_land_usage,model_baseline.minimize_health_impact]
    _, data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(parameters, df = data) 
    df = mo.solve_multi_objective(plot_graph=plot_graph, verbose=verbose)
    # df.to_csv(f"./minimization_objectives.csv")
    display(df)
    
    dataframe_matrix = df.to_numpy().astype(np.float64)
    normalization_matrix = np.vstack([results_F, dataframe_matrix[:4]])
    approx_ideal = normalization_matrix.min(axis=0)
    approx_nadir = normalization_matrix.max(axis=0)
    print(approx_ideal, approx_nadir)
    metric = Hypervolume(ref_point= np.array([1, 1, 1]),
                        norm_ref_point=False,
                        zero_to_one=True,
                        ideal=approx_ideal,
                        nadir=approx_nadir)
    metrics_for_heuristic = [metric.do(results_F[i]) for i in range(results_F.shape[0])]
    print(result_lengths)
    metrics_for_milp = [metric.do(dataframe_matrix[3])]
    heuristics = {}
    for i, algo in enumerate(algorithms):
        sum_so_far = np.sum(result_lengths[:i],dtype=int)
        print(f"Mean Hypervolume for the {algo.upper()} heuristic {np.mean(metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]])}")
        heuristics.update({np.mean(metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]]) : [metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]]]})
    print(f"Mean Hypervolume for the MILP {(metrics_for_milp)}")
    best_heuristic_length = np.sum(result_lengths[:np.argmax(list(heuristics.keys())) + 1])
    print("Best Heuristic result: ", results_F[np.argmax(heuristics[max(heuristics.keys())]) + best_heuristic_length])
    
