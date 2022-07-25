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
import networkx as nx
from plotter import Plotter

def create_node_sizes(one_result):
    all_binary_slice = slice(0, three_objective_problem.binary_end_facility_slice.stop, 1)
    res_reshaped = one_result[all_binary_slice].reshape(-1, 3)
    indices_node, indices_size = np.where(res_reshaped == 1)
    node_size_dictionary = {indices_node[i] + len(parameters.G.collection_locations) + 1: indices_size[i] for i in range(len(indices_node))}
    return node_size_dictionary

def get_key(val, dictionary):
    for key, value in dictionary.items():
         if val == value:
             return key

def create_solved_graph(solved_graph, link_for_j, row_indent, column_indent):
    mask_link_rows, mask_link_cols = np.where(link_for_j != 0)
    nodes_rows = np.add(mask_link_rows, np.ones(len(mask_link_rows), dtype = int))
    nodes_cols = np.add(mask_link_cols, np.ones(len(mask_link_cols), dtype = int))
    translated_nodes_rows = [get_key(node + row_indent, parameters.G.node_translator) for node in nodes_rows]
    translated_nodes_cols = [get_key(node + column_indent, parameters.G.node_translator) for node in nodes_cols]
    distances = [np.round(np.linalg.norm(np.array(translated_nodes_rows)[i] - np.array(translated_nodes_cols)[i])) for i in range(len(translated_nodes_rows))]
    for i in range(len(distances)):
        solved_graph.add_edge(translated_nodes_rows[i], translated_nodes_cols[i], weight=distances[i])
    return solved_graph


# The code below is solving the multiobjective problem using the heuristic and the model.
if __name__ == '__main__':
    # Used to suppress scientific notation.
    np.set_printoptions(suppress=True)
    set_seed = 1
    plot_graph = True
    verbose = False
    # Algorithms include: nsga2, nsga3, unsga3, agemoea, moead, ctaea
    algorithms = ["ctae", "nsga3", "nsga2","unsga3", "agemoea", "moead"]
    # algorithms = ["nsga3"]
    # Increasing num_of_collection_centers takes alot more time.
    if isinstance(algorithms, str): algorithms = [algorithms]
    num_of_collection_centers = 5
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=plot_graph, seed=set_seed, baseline_scaler=6)
    parameters = Parameters(RandomGraph, set_seed)
    termination = MultiObjectiveSpaceToleranceTermination(tol=0.1,n_last=30,nth_gen=5,n_max_gen=2000,n_max_evals=None)
    three_objective_problem = Vectorized_heuristic(parameters)
    ref_dirs = get_reference_directions("das-dennis", n_dim = 3, n_partitions = 20)
    result_lengths = [0]
    results_F = np.array([])
    results_X = np.array([])
    for algo in algorithms:
        minimization = Minimize(problem = three_objective_problem, population_size = len(ref_dirs) + 50, reference_directions = ref_dirs, termination=termination, verbose = verbose, algorithm=algo, seed = set_seed)
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
    model_baseline = Model_Baseline(parameters=parameters, plot_graph = plot_graph, verbose=verbose)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_land_usage, model_baseline.minimize_health_impact]
    _, data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(parameters, df = data) 
    df = mo.solve_multi_objective(plot_graph=plot_graph, verbose=verbose)
    # df.to_csv(f"./minimization_objectives.csv")
    display(df)
    
    dataframe_matrix = df.to_numpy().astype(np.float64)
    normalization_matrix = np.vstack([results_F, dataframe_matrix[:4]])
    approx_ideal = dataframe_matrix[:4].min(axis=0)
    approx_nadir = normalization_matrix.max(axis=0)
    print(approx_ideal, approx_nadir)
    metric = Hypervolume(ref_point= np.array([1, 1, 1]),
                        norm_ref_point=False,
                        zero_to_one=True,
                        ideal=approx_ideal,
                        nadir=approx_nadir)
    metrics_for_heuristic = [metric.do(results_F[i]) for i in range(results_F.shape[0])]
    metrics_for_milp = [metric.do(dataframe_matrix[3])]
    heuristics = {}
    for i, algo in enumerate(algorithms):
        sum_so_far = np.sum(result_lengths[:i],dtype=int)
        print(f"Mean Hypervolume for the {algo.upper()} heuristic {np.mean(metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]])}")
        heuristics.update({np.mean(metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]]) : [metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]]]})
    print(f"Mean Hypervolume for the MILP {(metrics_for_milp)}")
    best_heuristic_length = np.sum(result_lengths[:np.argmax(list(heuristics.keys())) + 1])
    best_index = np.argmax(heuristics[max(heuristics.keys())]) + best_heuristic_length
    best_algo = algorithms[np.argmax(list(heuristics.keys()))]
    best_result_F = results_F[best_index]
    best_result_X = results_X[best_index]
    best_hypervolume = metrics_for_heuristic[best_index]
    print("Best Heuristic result: ", best_result_F)
    if plot_graph:
        ij_for_j = best_result_X[three_objective_problem.continuous_ij_slice].reshape(-1, len(parameters.sorting_facilities))
        jk_for_j = best_result_X[three_objective_problem.continuous_jk_slice].reshape(len(parameters.sorting_facilities), -1)
        jkp_for_j = best_result_X[three_objective_problem.continuous_jkp_slice].reshape(len(parameters.sorting_facilities), -1)
        node_sizes = create_node_sizes(best_result_X)
        solved_graph = nx.DiGraph()
        solved_graph = create_solved_graph(solved_graph, ij_for_j, 0, ij_for_j.shape[0])
        solved_graph = create_solved_graph(solved_graph, jk_for_j, ij_for_j.shape[0], ij_for_j.shape[0] + ij_for_j.shape[1])
        solved_graph = create_solved_graph(solved_graph, jkp_for_j, ij_for_j.shape[0], ij_for_j.shape[0] + ij_for_j.shape[1] + jk_for_j.shape[1])
        plot = Plotter(parameters, solved_graph, node_sizes)
        plot.plot_graph(figure_name = f"{best_algo.upper()} with f1: {best_result_F[0]}, f2: {best_result_F[1]}, f3: {best_result_F[2]}, and Hypervolume {best_hypervolume}").show()
