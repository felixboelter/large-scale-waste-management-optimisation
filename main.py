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
from plotly.subplots import make_subplots
import sys
import pandas as pd
from itertools import combinations
import os
import pickle
import numpy as np
import pickle
from typing import Tuple
class Run():
    def __init__(self,algorithms : list, plot_graph : bool = False, verbose : bool = False):
        """
        This function takes in a list of algorithms and sets the plot_graph and verbose variables to
        False if they are not specified
        
        :param algorithms: list of algorithms to be used for the experiment. Algorithms include: nsga2, nsga3, unsga3, agemoea, moead, ctae
        :type algorithms: list
        :param plot_graph: If True, the algorithm will plot the best solutions found, defaults to
        False
        :type plot_graph: bool (optional)
        :param verbose: If True, prints out the progress of the algorithm, defaults to False
        :type verbose: bool (optional)
        """
        self._plot_graph = plot_graph
        self._verbose = verbose
        self.algorithms = algorithms
    def _create_node_sizes(self, one_result : np.ndarray):
        """
        It takes a solution vector and returns a dictionary that maps nodes to their sizes
        
        :param one_result: a single result from the optimization
        :return: A dictionary with the node number as key and the size as value.
        """
        all_binary_slice = slice(0, self.problem.binary_end_facility_slice.stop, 1)
        res_reshaped = one_result[all_binary_slice].reshape(-1, 3)
        indices_node, indices_size = np.where(res_reshaped == 1)
        node_size_dictionary = {indices_node[i] + len(self.parameters.G.collection_locations) + 1: indices_size[i] for i in range(len(indices_node))}
        return node_size_dictionary

    def _get_key(self, val, dictionary) -> tuple:
        """
        It takes a value and a dictionary as input, and returns the key that corresponds to that value
        
        :param val: the value you want to find the key for
        :param dictionary: the dictionary you want to search
        :return: The key of the value that is passed in.
        """
        for key, value in dictionary.items():
            if val == value:
                return key

    def _create_solved_graph(self, solved_graph : nx.DiGraph, link_for_j : np.ndarray, row_indent : int, column_indent : int):
        """
        It takes a graph and a matrix of links, and adds edges to the graph based on the links
        
        :param solved_graph: the graph that we're adding the edges to
        :type solved_graph: nx.DiGraph
        :param link_for_j: the link matrix for the jth subgraph
        :type link_for_j: np.ndarray
        :param row_indent: the number of rows in the previous subgraph
        :type row_indent: int
        :param column_indent: the number of columns in the previous subgraph
        :type column_indent: int
        :return: a solved graph.
        """
        mask_link_rows, mask_link_cols = np.where(link_for_j != 0)
        nodes_rows = np.add(mask_link_rows, np.ones(len(mask_link_rows), dtype = int))
        nodes_cols = np.add(mask_link_cols, np.ones(len(mask_link_cols), dtype = int))
        translated_nodes_rows = [self._get_key(node + row_indent, self.parameters.G.node_translator) for node in nodes_rows]
        translated_nodes_cols = [self._get_key(node + column_indent, self.parameters.G.node_translator) for node in nodes_cols]
        distances = [np.round(np.linalg.norm(np.array(translated_nodes_rows)[i] - np.array(translated_nodes_cols)[i])) for i in range(len(translated_nodes_rows))]
        for i in range(len(distances)):
            solved_graph.add_edge(translated_nodes_rows[i], translated_nodes_cols[i], weight=distances[i])
        return solved_graph

    def generate_instances(self, folder_name : str, instances_name : str) -> None:
        """
        It creates a folder in the current working directory, and then creates a dictionary of instances
        with keys of the form (centers, instance) and values of the form [locations, seed]
        
        :param folder_name: the name of the folder where the instances will be saved
        :type folder_name: str
        :param instances_name: the name of the file that will be created
        :type instances_name: str
        """
        path = os.path.join(os.getcwd(), folder_name)
        try: 
            os.mkdir(path) 
        except OSError as error: 
            print(error)  
        random_instances = dict()
        for centers in range(2,13,1):
            for instance in range(1,101):
                set_seed = np.random.randint(0,2**32)
                np.random.seed(set_seed)
                locations = tuple(map(tuple,100*np.random.random((centers*4,2))))
                random_instances.update({(centers,instance): [locations, set_seed]})
        for centers in range(15,105,5):
            for instance in range(1,101):
                set_seed = np.random.randint(0,2**32)
                np.random.seed(set_seed)
                locations = tuple(map(tuple,100*np.random.random((centers*4,2))))
                random_instances.update({(centers,instance): [locations, set_seed]})
        with open(os.path.join(path, f"{instances_name}.pickle"), 'wb') as handle:
            pickle.dump(random_instances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return random_instances
    def load_instances(self, folder_name : str, instances_name) -> dict:
        """
        Loads the instances from the specified folder and returns them as a dictionary
        
        :param folder_name: the name of the folder where the instances are stored
        :type folder_name: str
        :param instances_name: the name of the file that contains the instances
        :return: A dictionary of instances.
        """
        random_instances_path = os.path.join(os.getcwd(), folder_name)
        with open(os.path.join(random_instances_path, f"{instances_name}.pickle"), "rb") as handle:
            instances = pickle.load(handle)
        return instances

    def run_cplex(self):
        """
        It takes in the parameters, the plot_graph and the verbose parameters, runs the single-objective and multi-objective CPLEX models, and returns the total time,
        the dataframe and the multi_df_X.
        
        :param parameters: Parameters
        :type parameters: Parameters
        :param plot_graph: If True, the model will plot the solutions
        :type plot_graph: bool
        :param verbose: If True, prints the output of the CPLEX solver
        :type verbose: bool
        :return: The total time, the dataframe and the multi_df_X
        """
        model_baseline = Model_Baseline(parameters=self.parameters, plot_graph = self._plot_graph, verbose=self._verbose)
        list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_land_usage, model_baseline.minimize_health_impact]
        total_time_single, data, single_df_X, single_figs = model_baseline.solve_model(list_of_functions)
        mo = Multiobjective_model(self.parameters, df = data) 
        total_time_multi, df, multi_df_X, multi_figs = mo.solve_multi_objective(plot_graph=self._plot_graph, verbose=self._verbose)
        total_time = total_time_single + total_time_multi
        return total_time, df, multi_df_X, single_df_X, single_figs, multi_figs

    def run_heuristic(self, termination : MultiObjectiveSpaceToleranceTermination, population_size : int, seed : int):
        """
        This function runs a list of heuristic algorithms on a given problem, and returns the results in a
        format that is easily plottable / saveable
        
        :param algorithms: list of strings, the heuristic algorithms to run
        :type algorithms: list
        :param problem: the problem we want to solve
        :type problem: Vectorized_heuristic
        :param termination: the termination criteria for the algorithm
        :type termination: MultiObjectiveSpaceToleranceTermination
        :param population_size: the number of points in the population
        :type population_size: int
        :param verbose: whether to print the progress of the algorithm
        :type verbose: bool
        :param seed: random seed for the algorithm
        :type seed: int
        """
        objectives = 3
        ref_dirs = get_reference_directions("energy", objectives, population_size, seed = seed)
        result_lengths = [0]
        results_F = np.array([])
        results_X = np.array([])
        total_times = dict()
        for i, algo in enumerate(self.algorithms):
            minimization = Minimize(problem = self.problem, population_size = population_size, reference_directions = ref_dirs, termination=termination, verbose = self._verbose, algorithm=algo, seed = seed)
            result = minimization.minimize_heuristic()
            print(f'Time for {algo.upper()} heuristic: {result.exec_time}')
            total_times.update({i: result.exec_time})
            print(result.F[:2])
            print(self.problem.evaluate(result.X[0]))
            if len(results_F) == 0: 
                results_F = result.F
                results_X = result.X
            else: 
                results_F = np.vstack([results_F, result.F])
                results_X = np.vstack([results_X, result.X])
            result_lengths.append(result.F.shape[0])
        return results_F, results_X, result_lengths, total_times

    def _calulate_hypervolume(self, normalization_matrix : np.ndarray):
        approx_ideal = normalization_matrix.min(axis=0)
        approx_nadir = normalization_matrix.max(axis=0)
        metric = Hypervolume(ref_point= np.array([1, 1, 1]),
                            norm_ref_point=False,
                            zero_to_one=True,
                            ideal=approx_ideal,
                            nadir=approx_nadir)
        return metric
    def create_milp_results(self, df : pd.DataFrame, results_df : pd.DataFrame, metric_for_milp : int, total_cplex_time : float):
        single_objectives = df.to_numpy().astype(np.float64).min(axis=0)
        df_fc_fu = df.loc["Cost Objective, and Land Usage Objective"]
        df_fc_fh = df.loc["Cost Objective, and Health Impact Objective"]
        df_fu_fh = df.loc["Land Usage Objective, and Health Impact Objective"]
        df_tri = df.loc["Cost Objective, Land Usage Objective, and Health Impact Objective"]
        results_df.loc[len(results_df.index)] = ["MILP", single_objectives[0], single_objectives[1], single_objectives[2], (df_fc_fu["Cost Objective"], df_fc_fu["Land Usage Objective"]), (df_fc_fh["Cost Objective"], df_fc_fh["Health Impact Objective"]),(df_fu_fh["Land Usage Objective"], df_fu_fh["Health Impact Objective"]), (df_tri["Cost Objective"], df_tri["Land Usage Objective"], df_tri["Health Impact Objective"]), metric_for_milp * 100, 7, total_cplex_time]
        return results_df

    def create_heuristic_results(self, results_df : pd.DataFrame, normalization_matrix : np.ndarray, total_times : list) -> pd.DataFrame:
        approx_ideal = normalization_matrix.min(axis = 0)
        approx_nadir = normalization_matrix.max(axis = 0)
        for i, algo in enumerate(self.algorithms):
            comparison_array = []
            thetas = (self.heuristic_result[i] - approx_ideal) / (approx_nadir - approx_ideal)
            list_combinations = list()
            for n in range(1,4):
                list_combinations += list(combinations([0,1,2], n))
            for comb in list_combinations:
                theta_comparisons = thetas[:, np.r_[comb]]
                heuristic_comparisons = self.heuristic_result[i][:, np.r_[comb]]
                minimum_ub_ix = np.argmin(np.max(theta_comparisons, axis = 1))
                best_comparison = heuristic_comparisons[minimum_ub_ix]
                if len(best_comparison) == 1: best_comparison = best_comparison[0]
                elif len(best_comparison) > 1: best_comparison = tuple(best_comparison)
                comparison_array.append(best_comparison)
            best_hypervolume = self.heuristic_metric[i][minimum_ub_ix]
            results_df.loc[len(results_df.index)] = [algo.upper(), *comparison_array, best_hypervolume * 100, self.heuristic_result[i].shape[0], total_times[i]]
        return results_df

    def create_figures(self, result_lengths : list, results_X : np.ndarray, metrics_for_heuristic : list):
        _all_figs = []
        for i in range(len(self.algorithms)):
            _heuristic_length = np.sum(result_lengths[:i+1])
            _heuristic_array = list(self.heuristic_metric[i])
            _index = np.argmax(_heuristic_array) + _heuristic_length
            result_X = results_X[_index]
            hypervolume = metrics_for_heuristic[_index]
            ij_for_j = result_X[self.problem.continuous_ij_slice].reshape(-1, len(self.parameters.sorting_facilities))
            jk_for_j = result_X[self.problem.continuous_jk_slice].reshape(len(self.parameters.sorting_facilities), -1)
            jkp_for_j = result_X[self.problem.continuous_jkp_slice].reshape(len(self.parameters.sorting_facilities), -1)
            node_sizes = self._create_node_sizes(result_X)
            solved_graph = nx.DiGraph()
            solved_graph = self._create_solved_graph(solved_graph, ij_for_j, 0, ij_for_j.shape[0])
            solved_graph = self._create_solved_graph(solved_graph, jk_for_j, ij_for_j.shape[0], ij_for_j.shape[0] + ij_for_j.shape[1])
            solved_graph = self._create_solved_graph(solved_graph, jkp_for_j, ij_for_j.shape[0], ij_for_j.shape[0] + ij_for_j.shape[1] + jk_for_j.shape[1])
            plot = Plotter(self.parameters, solved_graph, node_sizes)
            fig = plot.plot_graph(figure_name = f"{self.algorithms[i].upper()} with Hypervolume {(hypervolume * 100):.3f}")
            if len(self.algorithms) == 1: 
                fig.show()
                sys.exit()
            else: _all_figs.append(fig)
        return _all_figs

    def plot_heuristic(self, title : str, figures : list):
        _num_rows = int(np.ceil(len(self.algorithms)/2))
        _num_cols = 2
        _fig_names = [fig['layout']['title']['text'] for fig in figures]
        _total_fig = make_subplots(rows = _num_rows, cols = _num_cols, shared_xaxes= False,start_cell="top-left",subplot_titles = _fig_names,vertical_spacing=0.1, horizontal_spacing=0.1)
        col_num = 1
        row_num = 1
        for idx, fig in enumerate(figures):
            if col_num > 2:
                row_num += 1
                col_num = 1
            for item in fig['data']:
                if idx > 0: item['showlegend'] = False
                _total_fig.add_trace(item, row = row_num, col = col_num)
            col_num +=1
        _total_fig.update_layout(width=1280, height=720,title_text = title, title_x = 0.5)
        return _total_fig

    def main(self,range_of_cities : range, range_of_instances : range, seeds : list, cplex : bool = True):
        randominstances_path = os.path.join(os.getcwd(), "RandomInstances")
        if os.path.exists(randominstances_path): instances = self.load_instances("RandomInstances", "instances")
        else: instances = self.generate_instances("RandomInstances", "instances")
        results_path = os.path.join(os.getcwd(), "Results")
        try: os.mkdir(results_path) 
        except OSError as error: print(f"{error}, Continuing without making directory at {results_path}")  
        for avg_num_of_cities in range_of_cities:
            for instance in range_of_instances:
                dict_entries = instances[(avg_num_of_cities,instance)]
                locations = dict_entries[0]
                set_seed = dict_entries[1]
                city_path = os.path.join(results_path, f"{avg_num_of_cities}_cities")
                if not os.path.exists(city_path): os.mkdir(city_path)
                city_results_path = os.path.join(city_path, f"{instance}_{set_seed}")
                os.mkdir(city_results_path)
                # Used to suppress scientific notation.
                np.set_printoptions(suppress=True)
                if isinstance(self.algorithms, str): self.algorithms = [self.algorithms]
                # Increasing num_of_collection_centers takes alot more time.
                RandomGraph = Graph(avg_num_of_cities,baseline=True,plot_graph=self._plot_graph, seed=set_seed, baseline_scaler=4, locations=locations)
                if self._plot_graph: 
                    RandomGraph.random_graph_figure.update_layout(width=1280, height=720, title_x= 0.5)
                    RandomGraph.random_graph_figure.write_image(os.path.join(city_results_path, f"original_graph_static.png"))
                    RandomGraph.random_graph_figure.write_html(os.path.join(city_results_path, f"original_graph_interactive.html"))
                self.parameters = Parameters(RandomGraph, set_seed)
                self.problem = Vectorized_heuristic(self.parameters)
                termination = MultiObjectiveSpaceToleranceTermination(tol=0.01,n_last=30,nth_gen=5,n_max_gen=1000,n_max_evals=200000)
                pop_size = 200
                if cplex:
                    total_cplex_time, df, multi_df_X, single_df_X, single_figs, multi_figs = self.run_cplex()
                    if single_figs is not None and multi_figs is not None:
                        single_figs.update_layout(width=1280, height=720,title_text = f"Single-objective solutions for {avg_num_of_cities} average number of cities.", title_x = 0.5)
                        multi_figs.update_layout(width=1280, height=720,title_text = f"Multi-objective solutions for {avg_num_of_cities} average number of cities.", title_x = 0.5)
                        single_figs.write_html(os.path.join(city_results_path, f"single_objective_cplex_interactive.html"))
                        multi_figs.write_html(os.path.join(city_results_path, f"multi_objective_cplex_interactive.html"))
                        single_figs.write_image(os.path.join(city_results_path, f"single_objective_cplex_static.png"))
                        multi_figs.write_image(os.path.join(city_results_path, f"multi_objective_cplex_static.png"))
                    multi_df_X.to_csv(os.path.join(city_results_path, "CPLEX_multi_X.csv"))
                    single_df_X.to_csv(os.path.join(city_results_path, "CPLEX_single_X.csv"))
                    df.to_csv(os.path.join(city_results_path, "CPLEX_single_multi_F.csv"))
                    dataframe_matrix = df.to_numpy().astype(np.float64)
                    normalization_matrix = dataframe_matrix
                for seed in seeds:
                    seed_path = os.path.join(city_results_path, f"data_{seed}")
                    results_F, results_X, result_lengths, total_times = self.run_heuristic(termination, pop_size, seed)
                    if cplex : normalization_matrix= np.vstack([normalization_matrix, results_F])
                    else: normalization_matrix = results_F
                    results_df = pd.DataFrame(columns=["Method", "Fc", "Fu", "Fh", "Fc + Fu", "Fc + Fh", "Fu + Fh", "Fc + Fu + Fh", "Hypervolume %", "Number of Solutions", "Total Time"])
                    metric = self._calulate_hypervolume(normalization_matrix)
                    if cplex: results_df = self.create_milp_results(df, results_df, metric.do(dataframe_matrix[3]), total_cplex_time)
                    metrics_for_heuristic = [metric.do(results_F[i]) for i in range(results_F.shape[0])]
                    self.heuristic_metric = {}
                    self.heuristic_result = {}
                    for i in range(len(self.algorithms)):
                        sum_so_far = np.sum(result_lengths[:i+1],dtype=int)
                        self.heuristic_metric.update({i : metrics_for_heuristic[sum_so_far:sum_so_far+result_lengths[i+1]]})
                        self.heuristic_result.update({i : results_F[sum_so_far:sum_so_far+result_lengths[i+1]]})
                    results_df = self.create_heuristic_results(results_df, normalization_matrix, total_times)
                    os.mkdir(seed_path)
                    results_df.to_csv(os.path.join(seed_path, "results_df.csv"))
                    with open(os.path.join(seed_path, "heuristic_results_F.npy"), 'wb') as f:
                        np.save(f, results_F)
                    with open(os.path.join(seed_path, "heuristic_results_X.npy"), 'wb') as f:
                        np.save(f, results_X)
                    if self._plot_graph:
                        figures = self.create_figures(result_lengths, results_X, metrics_for_heuristic)
                        subplot_figure = self.plot_heuristic(f"Heuristic plot for seed: {seed}", figures)
                        subplot_figure.write_image(os.path.join(seed_path, f"heuristic_static_{seed}.png"))
                        subplot_figure.write_html(os.path.join(seed_path, f"heuristic_interactive_{seed}.html"))
# The code below is solving the multiobjective problem using the heuristic and the model.
if __name__ == '__main__':
    list_of_algorithms = ["nsga2","nsga3", "unsga3", "ctae","agemoea", "moead"]
    list_of_random_seeds =  [37072584, 863683177, 272556303]
    runner = Run(list_of_algorithms, plot_graph= True, verbose= False)
    runner.main(range(2,13,1),range(1,11),seeds = list_of_random_seeds, cplex=True)
    runner.main(range(15,105,5),range(1,11), seeds = list_of_random_seeds, cplex=False)

