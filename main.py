from parameters import Parameters
from generate_graph import Graph
from solve_baseline import Model_Baseline, Multiobjective_model
from pymoo.core.problem import starmap_parallelized_eval
from parameters import Parameters
from heuristic import Elementwise_heuristic, Vectorized_heuristic, Minimize
from pymoo.factory import get_performance_indicator
import multiprocessing as mp
    
# The code below is solving the multiobjective problem using the heuristic and the model.
if __name__ == '__main__':
    pool = mp.pool.ThreadPool()
    set_seed = 0
    plot_graph = True
    verbose = True
    nsga3 = False
   # Increasing num_of_collection_centers takes alot more time.
    num_of_collection_centers = 5
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=plot_graph, seed=set_seed, baseline_scaler=3)
    parameters = Parameters(RandomGraph, set_seed)
    three_objective_problem = Vectorized_heuristic(parameters)
    minimization = Minimize(problem = three_objective_problem, population_size = 2000, number_of_generations = 800, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print('Time for 3 objective execution Vectorized:', result.exec_time)
    print(result.F, result.X)
    three_objective_problem = Elementwise_heuristic(parameters, runner=pool.starmap, func_eval = starmap_parallelized_eval)
    minimization = Minimize(problem = three_objective_problem, population_size = 1000, number_of_generations = 800, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print('Time for 3 objective execution Elementwise:', result.exec_time)
    print(result.F, result.X)
    pool.close()
    parameters = Parameters(G = RandomGraph, seed=set_seed)
    model_baseline = Model_Baseline(parameters=parameters, verbose=verbose)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_land_usage,model_baseline.minimize_health_impact]
    data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(parameters, df = data) 
    df = mo.solve_multi_objective(plot_graph=plot_graph, verbose=verbose)
    # df.to_csv(f"./minimization_objectives.csv")
    print(df)