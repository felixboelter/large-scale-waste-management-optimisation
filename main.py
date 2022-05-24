from generate_graph import Graph
from solve_baseline import Model_Baseline, Multiobjective_model

if __name__ == '__main__':
    set_seed = 1234567890
    plot_graph = True
    verbose = True
    # Increasing num_of_collection_centers takes alot more time.
    num_of_collection_centers = 4
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=plot_graph, seed=set_seed, baseline_scaler=10)
    model_baseline = Model_Baseline(RandomGraph, plot_graph=plot_graph, seed=set_seed, verbose=verbose)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_land_usage,model_baseline.minimize_health_impact]
    data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(RandomGraph, df = data, seed=set_seed) 
    df = mo.solve_multi_objective(plot_graph=plot_graph, verbose=verbose)
    df.to_csv(f"./minimization_objectives.csv")
    print(df)