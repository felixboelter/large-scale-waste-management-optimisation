from docplex.mp.model import Model
import docplex.mp, docplex.mp.solution, docplex.mp.linear
from generate_graph import Graph
import numpy as np
import networkx as nx
from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
from plotly import graph_objs as go
import pandas as pd
from itertools import combinations
from plotly.subplots import make_subplots
import ast
import time
from parameters import Parameters

FND = TypeVar("FND", float, np.ndarray)

class Model_Baseline():
    """
    Class Model_Baseline, inherits Parameters.
    Creates a baseline model object, which is based on Olapiriyakul, Sun & Pannakkong, Warut & Kachapanya, Warith & Starita, Stefano. (2019).
    """
    def __init__(self, parameters : Parameters, plot_graph : Optional[bool] = False,verbose : Optional[bool] = False) -> None:
        """
        The function takes in a graph object, a seed, and a boolean value for whether or not to plot the
        graph. It then creates a model object, and creates a list of edges for each type of facility. It
        then creates the decision variables, and creates a tuple for each type of facility. It then
        creates the constraints
        
        :param parameters: Parameters
        :type parameters: Parameters
        :param plot_graph: If True, the graph will be plotted after the model is solved, defaults to
        False
        :type plot_graph: Optional[bool] (optional)
        :param verbose: If True, prints out the model's constraints and objective function, defaults to
        False
        :type verbose: Optional[bool] (optional)
        """
        self._parameters = parameters
        self.model = Model(name="Baseline")
        self._verbose = verbose
        self._plot_graph = plot_graph
        self._ij_list = [(i,j,w['weight']) for i, j, w in self._parameters._G.G.edges(data=True) if (i in self._parameters._G.collection_locations and j in self._parameters._sorting_facilities)]
        self._jk_list = [(j,k,w['weight']) for j, k, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and k in self._parameters._incinerator_facilities)]
        self._jkp_list = [(j,kp,w['weight']) for j, kp, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and kp in self._parameters._landfill_facilities)]
        self._S = 0
        self._I = 1
        self._L = 2
        self._create_decision_variables()
        self._sorting_tuple = (self._S, self._y_sorting, self._parameters._sorting_facilities)
        self._incinerator_tuple = (self._I, self._y_incinerator, self._parameters._incinerator_facilities)
        self._landfill_tuple = (self._L, self._y_landfill, self._parameters._landfill_facilities)
        self.solved_model : Union[docplex.mp.solution.SolveSolution, None] = None
        self.minimization = str()
        self.solved_graph : nx.DiGraph = None
        self._create_constraints()
        
    def _create_decision_variables(self) -> None:
        """
        Private function. Creates the decision variables for the model.

        Decision variables
        ------------------
        y - Binary location variables, equal to 1 when sorting facilities,
        incinerators, and landfills of size l are open at their respective
        locations j, k, and k'.
        x - The number of trips on edges (i,j), (j,k), and (j,k').
        f - The amount of solid waste transported on edges (i,j), (j,k), and (j,k').
        """
        self._y_sorting = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._parameters._sorting_facilities for j in self._parameters._range_of_facility_sizes}
        self._y_incinerator = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._parameters._incinerator_facilities for j in self._parameters._range_of_facility_sizes}
        self._y_landfill = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._parameters._landfill_facilities for j in self._parameters._range_of_facility_sizes}
        self._x_ij = {(i,j) : self.model.integer_var(name=f'x_{i}_{j}') for i,j,_ in self._ij_list}
        self._x_jk = {(j,k) : self.model.integer_var(name=f'x_{j}_{k}') for j,k,_ in self._jk_list}
        self._x_jkp = {(j,kp) : self.model.integer_var(name=f'x_{j}_{kp}') for j,kp,_ in self._jkp_list}
        self._f_ij = {(i,j) : self.model.continuous_var(name=f'f_{i}_{j}') for i,j,_ in self._ij_list}
        self._f_jk_kp = {(j,k_kp) : self.model.continuous_var(name=f'f_{j}_{k_kp}') for j,k_kp,_ in self._jk_list+self._jkp_list}

    def _create_constraints(self) -> None:
        """
        Private function. Creates the constraints used by the model.

        Constraints
        -----------
        1 - Outflow of waste from any collection center i must be equal to the amount of available waste at i.
        2 - Flow balance, inflow of waste at sorting facility j is ENTIRELY forwarded to incinerator k or landfill k'.
        3, 4, 5 - Size dependent capacity constraint for sorting, incinerator, and landfill facilities.
        6, 7, 8 - Transportation capacity limitations for (i, j), (j, k), and (j, k').
        9, 10, 11 - Only one size is selected at a given location, for sorting, incinerator, and landfill facilities.
        """
        def _dictionary_array(array : List[tuple], inverse : bool = False) -> dict:
            """
            It takes a list of tuples and returns a dictionary
            
            :param array: The array to be made a dictionary out of
            :type array: List[tuple]
            :param inverse: If True, the dictionary will be inverted, defaults to False
            :type inverse: bool (optional)
            :return: A dictionary with the keys being the first element of the tuple and the values
            being the second element of the tuple.
            """
            dictionary = dict()
            for i,j,_ in array:
                if inverse == True:
                    try: dictionary[j].append(i)
                    except KeyError: dictionary[j] = [i]
                else:
                    try: dictionary[i].append(j)
                    except KeyError: dictionary[i] = [j]
            return dictionary
        
        def _size_dependent_capacity_constraint(type_of_constraint : int, y_variable : Model.binary_var, size_dependent_f : list, facilities : list) -> None:
            """
            The function takes in a type of constraint, a decision variable y, a list of sums of the
            decision variable f, and a list of facilities. It then creates a list of the sum of the
            decision variable multiplied by the facility storage capacity, and then adds a constraint to
            the model that the sum of the decision variables f is less than or equal to the sum of the
            decision variable y multiplied by the facility storage capacity.
            
            :param type_of_constraint: 0 for Sorting, 1 for Incinerator, and 2 for Landfill
            :type type_of_constraint: int
            :param y_variable: The y decision variable to be used
            :type y_variable: Model.binary_var
            :param size_dependent_f: List of sums of the decision variable f:
            :type size_dependent_f: list
            :param facilities: The list of facilities to loop over
            :type facilities: list
            """
           
            size_capacity= [self.model.sum(self._parameters.facility_storage_capacities[type_of_constraint][l] * y_variable[(j,l)] \
                 for l in self._parameters._range_of_facility_sizes) for j in facilities]
            for ix in range(0, len(size_dependent_f)):
                self.model.add_constraint(size_dependent_f[ix] <= size_capacity[ix])
       
        def _transportation_capacity_limitations(edge_list : list, x_variable : Model.integer_var, f_variable : Model.continuous_var) -> None:
            """
            For each edge in the edge list, add a constraint that the flow on that edge is less than
            the maximum amount of transport times the decision variable for that edge. 
            
            :param edge_list: Edges to loop over (i,j), (j,k), or (j,k')
            :type edge_list: list
            :param x_variable: The x decision variable to be used
            :type x_variable: Model.integer_var
            :param f_variable: The f decision variable to be used
            :type f_variable: Model.continuous_var
            """
            for i,j,_ in edge_list:
                for l in range(0,len(self._parameters.maximum_amount_transport)):
                    self.model.add_constraint(f_variable[(i,j)] <= self._parameters.maximum_amount_transport[l] * x_variable[(i,j)])
        
        def _one_size_selection_constraint(facilities : list, y_variable : Model.binary_var) -> None:
            """
            For each facility location, the sum of the y decision variables for each facility size
            must be less than or equal to 1. 
            
            This is a constraint that is used for all three of the facility size selection constraints.
            
            :param facilities: Facility locations to loop over
            :type facilities: list
            :param y_variable: The y decision variable to be used
            :type y_variable: Model.binary_var
            """
            for i in facilities:
                self.model.add_constraint(self.model.sum(y_variable[(i,l)] for l in self._parameters._range_of_facility_sizes) <= 1)
        
        _ij_for_i_dict = _dictionary_array(self._ij_list)
        _jk_for_j_dict = _dictionary_array(self._jk_list)
        _jkp_for_j_dict = _dictionary_array(self._jkp_list)
        _jk_kp_for_j_dict = _dictionary_array(self._jk_list + self._jkp_list)
        # Inverse dictionaries
        _ij_for_j_dict = _dictionary_array(self._ij_list,inverse=True)
        _jk_for_k_dict = _dictionary_array(self._jk_list,inverse=True)
        _jkp_for_kp_dict = _dictionary_array(self._jkp_list,inverse=True)

        # Constraints 
        # Constraint 1: Outflow of waste from any collection center i must be equal to the amount of available waste at i
        for key, val in _ij_for_i_dict.items():
            self.model.add_constraint(self.model.sum(self._f_ij[(key,j)] for j in val) == self._parameters._G.supplies[key][0])
        # Constraint 2: Flow balance, inflow of waste at sorting facility j is ENTIRELY forwarded to incinerator k or landfill k'
        _sum_f_ij_for_j= [self.model.sum(self._f_ij[(i,j)] for i in i_values) for j, i_values in _ij_for_j_dict.items()]
        _sum_f_jk_for_j = [self.model.sum(self._f_jk_kp[(j,k)] for k in k_values) for j, k_values in _jk_kp_for_j_dict.items()]
        for ix in range(0, len(_sum_f_jk_for_j)):
            self.model.add_constraint(_sum_f_ij_for_j[ix] == _sum_f_jk_for_j[ix])
        # Constraint 3: Size dependent capacity constraint for sorting facilities
        _size_dependent_capacity_constraint(self._S, self._y_sorting, _sum_f_ij_for_j, self._parameters._sorting_facilities)
        # Constraint 4: Size dependent capcity constraint for incinerator facilities
        _sum_f_jk_for_k = [self.model.sum(self._f_jk_kp[(j,k)] for j in j_value) for k, j_value in _jk_for_k_dict.items()] 
        _size_dependent_capacity_constraint(self._I, self._y_incinerator, _sum_f_jk_for_k, self._parameters._incinerator_facilities)
        # Constraint 5: Size dependent capcity constraint for landfill facilities
        _sum_f_jk_for_kp = [self.model.sum(self._f_jk_kp[(j,kp)]for j in j_value) for kp, j_value in _jkp_for_kp_dict.items()] 
        _size_dependent_capacity_constraint(self._L, self._y_landfill, _sum_f_jk_for_kp, self._parameters._landfill_facilities)
        # Constraint (6, 7, 8) : Transportation capacity limitations for ((i, j), (j, k), (j, kp))
        _ij_tuple = (self._ij_list, self._x_ij, self._f_ij)
        _jk_tuple = (self._jk_list, self._x_jk, self._f_jk_kp)
        _jkp_tuple = (self._jkp_list, self._x_jkp, self._f_jk_kp)
        for edge_list, x_decision_value, f_decision_value in [_ij_tuple, _jk_tuple, _jkp_tuple]:
            _transportation_capacity_limitations(edge_list, x_decision_value, f_decision_value)
        # Constraint (9, 10, 11): One size selection for (sorting, incinerator, landfill) facilities
        _sorting_tuple = (self._parameters._sorting_facilities, self._y_sorting)
        _incinerator_tuple = (self._parameters._incinerator_facilities, self._y_incinerator)
        _landfill_tuple = (self._parameters._landfill_facilities, self._y_landfill)
        for facilities, decision_values in [_sorting_tuple, _incinerator_tuple, _landfill_tuple]:
            _one_size_selection_constraint(facilities, decision_values)
    
    def minimize_cost(self, minimize : Optional[bool] = True) -> None:
        """
        The function `minimize_cost` is used to create the objective function of the model. The
        objective function is the sum of the opening costs of the facilities and the operational costs
        of the facilities
        
        :param minimize: If True, the model will be minimized. If False, then the total opening cost is returned, defaults to True
        :type minimize: Optional[bool] (optional)
        :return: The total opening cost sum.
        """
        
        def _create_opening_costs_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            The function takes in a type of facility, a decision variable, and a list of facilities
            and returns a linear expression of the sum of the opening costs of the facilities
            
            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill
            :type type_of_facility: int
            :param y_variable: The y decision variable to be used
            :type y_variable: Model.binary_var
            :param facilities: The list of facilities to loop over
            :type facilities: list
            :return: The opening costs linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum(y_variable.get((i,j)) * self._parameters.opening_costs[type_of_facility][j] for i in facilities for j in self._parameters._range_of_facility_sizes)
        
        def _create_operational_costs_sum(type_of_facility : int, x_variable : Model.integer_var, edge_list : list) -> docplex.mp.linear.LinearExpr:
            """
            The function takes in a type of facility, a decision variable, and a list of edges and
            returns a linear expression of the sum of the costs to transport and manage the solid waste
            flow
            
            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill
            :type type_of_facility: int
            :param x_variable: The x decision variable to be used
            :type x_variable: Model.integer_var
            :param edge_list: A list of tuples of the form (i,j) where i and j are the nodes of the
            graph
            :type edge_list: list
            :return: The sum of the operational costs for the facility type.
            """
            return self.model.sum((t_ij + self._parameters.operational_costs[type_of_facility]) * x_variable[(i,j)] for i, j, t_ij in edge_list)

        _sorting_operation_tuple = (self._S, self._x_ij, self._ij_list)
        _incinerator_operation_tuple = (self._I, self._x_jk, self._jk_list)
        _landfill_operation_tuple = (self._L, self._x_jkp, self._jkp_list)
        _cost_operational_objectives = [_create_operational_costs_sum(tuple_[0], tuple_[1], tuple_[2]) for tuple_ in [_sorting_operation_tuple, _incinerator_operation_tuple, _landfill_operation_tuple]]
        _cost_opening_objectives = [_create_opening_costs_sum(tuple_[0], tuple_[1], tuple_[2]) for tuple_ in [self._sorting_tuple, self._incinerator_tuple, self._landfill_tuple]]
        self.minimization = "Cost Objective"
        _total_opening_cost_sum = _cost_opening_objectives[self._S]+_cost_opening_objectives[self._I]+_cost_opening_objectives[self._L]+_cost_operational_objectives[self._S]+_cost_operational_objectives[self._I]+_cost_operational_objectives[self._L]
        if minimize == False: return _total_opening_cost_sum
        self.model.minimize(_total_opening_cost_sum)
    
    def minimize_land_usage(self, minimize : Optional[bool] = True) -> None:
        """
        The function `minimize_land_usage` is used to minimize the land usage objective
        
        :param minimize: Whether or not to minimize the land usage. If False, then the land usage linear
        expression is returned, defaults to True
        :type minimize: Optional[bool] (optional)
        :return: The total land usage sum.
        """
        
        def _create_land_usage_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            The function takes in a type of facility, a decision variable, and a list of facilities
            and returns a linear expression that sums the land stress ratios for each facility and size
            
            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill
            :type type_of_facility: int
            :param y_variable: The y decision variable to be used
            :type y_variable: Model.binary_var
            :param facilities: list of facility locations
            :type facilities: list
            :return: The land usage linear expression from the class docplex.mp.linear.LinearExpr
            """
           
            return self.model.sum(self._parameters.land_stress_ratios[type_of_facility][l] * y_variable[(i,l)] for i in facilities for l in self._parameters._range_of_facility_sizes)
        _land_usage_objectives = [_create_land_usage_sum(tuple_[0], tuple_[1], tuple_[2]) for tuple_ in [self._sorting_tuple, self._incinerator_tuple, self._landfill_tuple]]
        _total_land_usage_sum = _land_usage_objectives[self._S] + _land_usage_objectives[self._I] + _land_usage_objectives[self._L]
        self.minimization = "Land Usage Objective"
        if minimize == False: return _total_land_usage_sum
        self.model.minimize(_total_land_usage_sum)

    def minimize_health_impact(self, minimize : Optional[bool] = True) -> None:
        """
        The function `minimize_health_impact` takes in a boolean value `minimize` and returns `None`. If
        `minimize` is `True`, the function minimizes the health impact of the transportation and
        facilities on the population. If `minimize` is `False`, the function returns the health impact
        of the transportation and facilities on the population
        
        :param minimize: Whether to minimize the objective or not, defaults to True
        :type minimize: Optional[bool] (optional)
        :return: The total health impact sum.
        """
        
        def _create_facility_health_impact_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            The function creates a linear expression that sums the product of the population near a
            facility, the DALY per person, and the decision variable for the facility
            
            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill
            :type type_of_facility: int
            :param y_variable: The y decision variable to be used
            :type y_variable: Model.binary_var
            :param facilities: Facility locations to loop over
            :type facilities: list
            :return: The facility health impact linear expression from the class
            docplex.mp.linear.LinearExpr
            """
            return self.model.sum(self._parameters.population_near_facilities[i][type_of_facility][l] * self._parameters.facility_daly_per_person[i][l] * y_variable[(i,l)] for i in facilities for l in self._parameters._range_of_facility_sizes)

        def _create_transport_health_impact_sum(x_variable : Model.integer_var, edge_list : list) -> docplex.mp.linear.LinearExpr:
            """
            The function takes in a decision variable and a list of edges, and returns a linear
            expression that sums over the population of each edge times the DALYs of each edge times the
            decision variable
            
            :param x_variable: The x decision variable to be used
            :type x_variable: Model.integer_var
            :param edge_list: list of tuples of the form (i,j,k) where i,j,k are nodes in the network
            :type edge_list: list
            :return: The sum of the population of the link times the dalys of the link times the x
            variable of the link.
            """
            return self.model.sum(self._parameters.link_populations[(i,j)]  * self._parameters.link_dalys[(i,j)][1] * x_variable[(i,j)] for i,j,_ in edge_list)

        _sorting_transport_health_tuple = (self._x_ij, self._ij_list)
        _incinerator_transport_health_tuple = (self._x_jk, self._jk_list)
        _landfill_transport_health_tuple = (self._x_jkp, self._jkp_list)
        _transport_health_objectives = [_create_transport_health_impact_sum(tuple_[0], tuple_[1]) for tuple_ in [_sorting_transport_health_tuple, _incinerator_transport_health_tuple, _landfill_transport_health_tuple]]
        _facility_health_objectives = [_create_facility_health_impact_sum(tuple_[0], tuple_[1], tuple_[2]) for tuple_ in [self._sorting_tuple, self._incinerator_tuple, self._landfill_tuple]]
        _total_health_impact_sum = _facility_health_objectives[self._S] + _facility_health_objectives[self._I] + _facility_health_objectives[self._L] \
            + _transport_health_objectives[self._S] + _transport_health_objectives[self._I] + _transport_health_objectives[self._L]
        self.minimization = "Health Impact Objective"
        if minimize == False: return _total_health_impact_sum
        self.model.minimize(_total_health_impact_sum)
    
    def solve_model(self, list_of_functions : List[Any]) -> pd.DataFrame:
        """
        This function solves the model and returns a dataframe with the results
        
        :param list_of_functions: A list of functions that will be used to solve the model
        :type list_of_functions: List[Any]
        :return: A dataframe with the results of the model.
        """
        if not isinstance(list_of_functions, list): list_of_functions = [list_of_functions]
        self.model.parameters.mip.tolerances.mipgap = 0.000
        df = pd.DataFrame(columns=["Objective Name","Cost Objective", "Land Usage Objective", "Health Impact Objective"])
        model_df = pd.DataFrame()
        _figs = []
        log = False
        if self._verbose: log=True
        minimization_names = []
        for minimize_function in list_of_functions:
            minimize_function()
            minimization_names.append(self.minimization)
            self.model.print_information()
            tic = time.perf_counter()
            self.solved_model = self.model.solve(clean_before_solve=True, log_output = log)
            toc = time.perf_counter()
            assert self.solved_model, {f"Solution could not be found for this model. Got {self.solved_model}."}
            time_spent = toc - tic
            print(f"Elapsed time for {self.minimization} was {time_spent:0.4f} seconds")
            if self._verbose:
                self.solved_model.display()
                self.model.export_as_lp("./")
            self.model.remove_objective()
            self.solved_graph = nx.DiGraph()
            df_solved_model = self.solved_model.as_df(name_key = self.minimization)
            model_df = pd.concat([model_df,df_solved_model], axis=1)
            self.df_solved_model_list = [(key.split('_'), value) for key, value in df_solved_model.values if round(value) > 0]
            self._data_locations = [key for key, _ in self.df_solved_model_list]
            # Get the distance for all cities between all cities as our cost edges.
            for i in range(len(self._data_locations)):
                if 'f' in self._data_locations[i]:
                    first_node = ast.literal_eval(self._data_locations[i][1])
                    second_node = ast.literal_eval(self._data_locations[i][2])
                    # Eucledian distance calculation.
                    distance = np.round(((first_node[0] - second_node[0])**2 + (first_node[1] - second_node[1])**2)**0.5)
                    # Add the edge to a graph with the distance as an edge weight.
                    self.solved_graph.add_edge(first_node, second_node, weight=distance)
            df = self.create_dataframe(df)
            if self._plot_graph: _figs.append(self.plot_graph())
        if self._plot_graph:
            if len(list_of_functions) == 1:
                _figs[0].show()
                return df
            _num_rows = round(len(list_of_functions)/2)
            _num_cols = (len(list_of_functions)//2) + 1
            _total_fig = make_subplots(rows = _num_rows, cols = _num_cols, shared_xaxes= False,start_cell="top-left", subplot_titles=minimization_names,vertical_spacing=0.1, horizontal_spacing=0.1)
            col_num = 1
            row_num = 1
            for idx, fig in enumerate(_figs):
                if col_num > 2:
                    row_num = 2
                    col_num = 1
                for item in fig['data']:
                    if idx > 0: item['showlegend'] = False
                    _total_fig.add_trace(item, row = row_num, col = col_num)
                col_num +=1
            _total_fig.show()
        return model_df, df
    
    def _calculate_cost(self, y_decisions : List[Tuple[Any, int]], x_decisions : Dict[tuple, Any]) -> Tuple[float, float]:
        """
        For each facility, if it is a sorting facility, add the opening cost of a sorting facility to
        the total opening cost. If it is an incinerator facility, add the opening cost of an incinerator
        facility to the total opening cost. If it is a landfill facility, add the opening cost of a
        landfill facility to the total opening cost. For each edge, add the operational cost of the edge
        to the total operational cost
        
        :param y_decisions: a list of tuples, where each tuple is a facility and its level of operation
        :type y_decisions: List[Tuple[Any, int]]
        :param x_decisions: a dictionary of tuples to floats, where the tuples are the edges and the
        floats are the amount of waste that is sent along that edge
        :type x_decisions: Dict[tuple, Any]
        :return: The opening cost and the operational cost.
        """
        _opening_cost = 0
        _opening_cost_array = self._parameters.opening_costs
        for j, l in y_decisions:
            if j in self._parameters._sorting_facilities:  _opening_cost += _opening_cost_array[0][l]
            elif j in self._parameters._incinerator_facilities: _opening_cost += _opening_cost_array[1][l]
            elif j in self._parameters._landfill_facilities: _opening_cost += _opening_cost_array[2][l]
        _operational_cost = 0
        _operational_cost_array = self._parameters.operational_costs
        for index, edge_list in enumerate([self._ij_data, self._jk_data, self._jkp_data]):
            for i, j, t_ij in edge_list:
                _operational_cost += (t_ij + _operational_cost_array[index]) * x_decisions[(i,j)]
        return (_opening_cost, _operational_cost)
    
    def _calculate_land_usage(self, y_decisions : List[Tuple[Any, int]]) -> float:
        """
        The function calculates the land usage of the decision variables
        
        :param y_decisions: a list of tuples, where each tuple is (facility, location)
        :type y_decisions: List[Tuple[Any, int]]
        :return: The land usage is being returned.
        """
        land_usage = 0
        land_stress_ratio_array = self._parameters.land_stress_ratios
        for j, l in y_decisions:
            if j in self._parameters._sorting_facilities:  land_usage += land_stress_ratio_array[0][l]
            elif j in self._parameters._incinerator_facilities: land_usage += land_stress_ratio_array[1][l]
            elif j in self._parameters._landfill_facilities: land_usage += land_stress_ratio_array[2][l]
        return land_usage

    def _calculate_health_impact(self, y_decisions : List[Tuple[Any, int]], x_decisions : Dict[tuple, Any]) -> Tuple[float, float]:
        """
        For each facility, we multiply the population near the facility by the dalys per person for that
        facility. 
        
        For each transport link, we multiply the population near the link by the dalys per person for
        that link. 
        
        We then sum all of these values to get the total health impact.
        
        :param y_decisions: a list of tuples, where each tuple is a facility and a facility size.
        :type y_decisions: List[Tuple[Any, int]]
        :param x_decisions: a dictionary of the decisions made for the transport links. The keys are
        tuples of the form (i,j) where i and j are the nodes that the link connects. The values are the
        decisions made for that link.
        :type x_decisions: Dict[tuple, Any]
        :return: the health impact of the transport and facility decisions.
        """
        facility_health_impact = 0
        pop_near_facilities = self._parameters.population_near_facilities
        facility_dalys = self._parameters.facility_daly_per_person
        for j, l in y_decisions:
            if j in self._parameters._sorting_facilities:  facility_health_impact += pop_near_facilities[j][0][l] * facility_dalys[j][l]
            elif j in self._parameters._incinerator_facilities: facility_health_impact += pop_near_facilities[j][1][l] * facility_dalys[j][l]
            elif j in self._parameters._landfill_facilities: facility_health_impact += pop_near_facilities[j][2][l] * facility_dalys[j][l]

        transport_health_impact = 0
        population_for_edge = self._parameters.link_populations
        daly_for_edge = self._parameters.link_dalys
        for edge_list in [self._ij_data, self._jk_data, self._jkp_data]:
            for i,j,_ in edge_list:
                transport_health_impact += population_for_edge[(i,j)] * daly_for_edge[(i,j)][1] * x_decisions[(i,j)] 
        return (transport_health_impact, facility_health_impact)

    def create_dataframe(self, df : pd.DataFrame, name : Optional[str] = None) -> pd.DataFrame:
        """
        This function takes in a dataframe and a name, and returns a dataframe with the name and the
        objective values of the model
        
        :param df: The dataframe to append the results to
        :type df: pd.DataFrame
        :param name: The name of the model
        :type name: Optional[str]
        :return: The dataframe is being returned.
        """
        self._ij_data = [(i,j,w['weight']) for i, j, w in self.solved_graph.edges(data=True) if i in self._parameters._G.collection_locations and j in self._parameters._sorting_facilities]
        self._jk_data = [(j,k,w['weight']) for j, k, w in self.solved_graph.edges(data=True) if (j in self._parameters._sorting_facilities and k in self._parameters._incinerator_facilities)]
        self._jkp_data = [(j,kp,w['weight']) for j, kp, w in self.solved_graph.edges(data=True) if (j in self._parameters._sorting_facilities and kp in self._parameters._landfill_facilities)]
        _y_decisions = [(ast.literal_eval(key[1]), int(key[2])) for key, _ in self.df_solved_model_list if 'y' in key]
        _x_decisions = {(ast.literal_eval(key[1]), ast.literal_eval(key[2])): value for key, value in self.df_solved_model_list if 'x' in key}
        _cost_objective : Tuple[float, float] = self._calculate_cost(_y_decisions, _x_decisions)
        _land_objective : float = self._calculate_land_usage(_y_decisions)
        _health_objective : Tuple[float, float] = self._calculate_health_impact(_y_decisions, _x_decisions)
        obj_name = self.minimization
        if name is not None: obj_name = name
        df.loc[len(df.index)] = [obj_name, f"{sum(_cost_objective):8f}", _land_objective, f"{sum(_health_objective):8f}"]
        return df

    def plot_graph(self, figure_name : Optional[str] = None):
        """
        This function takes in a solved model and plots the graph with the edges coloured based on the
        edge weights
        
        :param figure_name: The name of the figure
        :type figure_name: Optional[str]
        :return: The plotly figure object.
        """
        def _edge_colours(G, value1: int, value2: int) -> Tuple[list, list, str, int, str]:
            """
            This function takes in a graph, and two values, and returns the x and y coordinates of the
            edges, the color of the edges, the width of the edges, and the name of the edges

            :param G: The graph object
            :param value1: The low value of the range
            :type value1: int
            :param value2: The high value of the range
            :type value2: int
            :return: The edge_x and edge_y coordinates, the color, the width and the name of the line.
            """
            edge_x = []
            edge_y = []
            for i,j,w in G.edges(data=True):
                x0, y0 = i
                x1, y1 = j
                weight = w['weight']
                if weight >= value1 and weight < value2:
                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
            if value1 == 60:
                color = "#DF4E4F"
                width = 1
                name = "Distance > 60"
            elif value1 == 40:
                color = '#FDB813'
                width = 2
                name = "Distance > 40"
            elif value1 == 0:
                color = '#4E9B47'
                width = 4
                name = "Distance < 40"
            return edge_x, edge_y, color, width, name

        CATEGORIES = 3
        VALUES = [0, 40, 60, 101]
        assert self.solved_model, {f"Solution could not be found for this model. Got {self.solved_model}."}
        if figure_name == None: figure_name= f"Solved solution for objective: {self.minimization}"
        fig = go.Figure(layout=go.Layout(
                        title=figure_name,
                        title_x = 0.5,
                        legend=dict(
                            x=1,
                            y=1,
                            traceorder="reversed",
                            title_font_family="Times New Roman",
                            font=dict(
                                family="Courier",
                                size=12,
                                color="black"
                            ),
                            bordercolor="Black",
                            borderwidth=2
                        ),
                        annotations=[ dict(
                                text= f"<b>Total Unsorted Supply:</b> {sum([us for us,_  in self._parameters._G.supplies.values()])}",
                                showarrow=False,
                                align = 'left',
                                x=0.005, y=-0.002 ) ],
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=True, zeroline=False, showticklabels=True))
                        )
        for i in range(CATEGORIES):
            x_edges, y_edges, colors, widths, name = _edge_colours(self.solved_graph,VALUES[i], VALUES[i+1])
            fig.add_trace(go.Scatter(
                x=x_edges, 
                y=y_edges,
                showlegend=True,
                name= name,
                line = dict(
                    color = colors,
                    width = widths),
                hoverinfo='none',
                mode='lines'))
        _node_colours = dict()
        _custom_node_attrs = dict()
        _data_locations_y = {ast.literal_eval(key[1]):key[2] for key in self._data_locations if 'y' in key}
        dictionary_values = {0 : "Small", 1: "Medium", 2 : "Large"}
        for node in self.solved_graph.nodes():
            if node in self._parameters._G.collection_locations: 
                _node_colours[node] = ["Collection Center", '#D7D2CB']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Unsorted Supply: {self._parameters._G.supplies[node][0]}"
            elif node in self._parameters._sorting_facilities: 
                _node_colours[node] = ["Sorting Facility", '#6AC46A']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(_data_locations_y[node])]}"
            elif node in self._parameters._incinerator_facilities: 
                _node_colours[node] = ["Incinerator Facility", '#952E25']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(_data_locations_y[node])]}"
            elif node in self._parameters._landfill_facilities: 
                _node_colours[node] = ["Landfill Facility", '#00C0F0']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(_data_locations_y[node])]}"
        
        seen_node_colours = []
        for node in self.solved_graph.nodes():
            if _node_colours[node] not in seen_node_colours:
                temp_x = []
                temp_y = []
                temp_attr = []
                current_node_colour = _node_colours[node]
                seen_node_colours.append(current_node_colour)
                for key, value in _node_colours.items():
                    if value == current_node_colour:
                        temp_x.append(key[0])
                        temp_y.append(key[1])
                        temp_attr.append(_custom_node_attrs[key])
                fig.add_trace(go.Scatter(
                    x=temp_x, y=temp_y,
                    mode='markers',
                    hoverinfo='text',
                    text = temp_attr,
                    name=f"{current_node_colour[0]}",
                    marker=dict(
                        color=current_node_colour[1],
                        size=20,
                        line_width=2)))
        return fig

class Multiobjective_model(Model_Baseline):
    """
    Class Multiobjective_model. Inherits Model_Baseline.
    Solve a min-max version of the single objective functions combined.
    """
    def __init__(self, parameters : Parameters, df : pd.DataFrame) -> None:
        """
        Multiobjective_model constructor.
        Initializes the class, creates the deviations, and adds the minimizing variable.

        :param parameters: The parameters for the optimization
        :type parameters: Parameters
        :param df: The pandas dataframe from the single optimizations
        :type df: pd.DataFrame
        """
        super().__init__(parameters)
        assert len(df) > 1, f"Must be atleast two functions being compared for a multiobjective optimization. Got {len(df)}"
        self.org_df : pd.DataFrame = df.copy()
        self.df : pd.DataFrame = df.copy()
        self.df.set_index('Objective Name', inplace=True)
        cols = ["Cost Objective", "Land Usage Objective", "Health Impact Objective"]
        self.df[cols] = self.df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        self.column_dict = {"Cost Objective" : [0, self.minimize_cost],
                            "Land Usage Objective" : [1, self.minimize_land_usage],
                            "Health Impact Objective" : [2, self.minimize_health_impact]}
        self.deviations = self._create_deviations()
        self._z = self.model.continuous_var(name="z") 

    def _create_deviations(self) -> List[docplex.mp.linear.LinearExpr]:
        """
        Private helper function.
        Creates the deviations listed in Olapiriyakul, Sun & Pannakkong, Warut & Kachapanya, Warith & Starita, Stefano. (2019)
        :return: A list of linear expressions.
        """
        # Optimal Values show minimum value for each objective. Worst values show maximum value for each objective.
        self._optimal_values = {index: row[self.column_dict[index][0]] for index, row in self.df.iterrows()}
        _worst_values = {index: self.df[index].max() for index in self._optimal_values.keys()}
        deviations = [((self.column_dict[key][1](minimize = False) - self._optimal_values[key])/(_worst_values[key] - self._optimal_values[key])) for key in self._optimal_values.keys()]
        return deviations
    def _create_multi_constraints(self, combination : Tuple[int, int]) -> None:
        """
        For each combination of the deviations, add a constraint that the deviation must be smaller or
        equal to the continuous variable z
        
        :param combination: A combination created by itertools.combinations
        :type combination: Tuple[int, int]
        """
        for comb in combination:
            self.model.add_constraint(self.deviations[comb] <= self._z, ctname = f"constraint_{comb}")
    def _remove_multi_constraints(self, combination: Tuple[int, int]) -> None:
        """
        Removes the constraints created in _create_multi_constraints() such that there is no overlap
        in constraints
        
        :param combination: A combination created by itertools.combinations
        :type combination: Tuple[int, int]
        """
        self.model.remove_objective()
        for comb in combination:
            self.model.remove_constraint(f"constraint_{comb}")

    def solve_multi_objective(self, plot_graph : Optional[bool] = False, verbose : Optional[bool] = False) -> pd.DataFrame:
        """
        The function solves the multi objective min-max problem. It goes through all combinations of
        single optimization functions and solves each one.
        
        :param plot_graph: Plot the created graph for each combination, defaults to False
        :type plot_graph: Optional[bool] (optional)
        :param verbose: Prints the model results and exports the problem to lp, defaults to False
        :type verbose: Optional[bool] (optional)
        :return: The dataframe object which includes all of the calculations for the various objectives.
        """
        
        def _get_key(dict : Dict[str, list], val : int) -> str:
            """
            It loops through a dictionary and returns the key of the value that is passed in
            
            :param dict: The dictionary which is looped through
            :type dict: Dict[str, list]
            :param val: The value which is searched for
            :type val: int
            :return: The key of the value.
            """
            for key, value in dict.items():
                if val in value:
                    return key
        _figures = []
        plot_names = []
        log = False
        if verbose: log=True
        all_double_combinations = list(combinations([i for i in range(0,len(self._optimal_values))], 2))
        if len(self._optimal_values) == 3: all_double_combinations.insert(0,(0,1,2))
        for combination in all_double_combinations:  
            self.model.minimize(self._z)
            self._create_multi_constraints(combination)
            self.model.print_information()
            self.model.round_solution = True
            tic = time.perf_counter()
            self.solved_model = self.model.solve(clean_before_solve=True,log_output=log)
            toc = time.perf_counter()

            assert self.solved_model, {f"Solution could not be found for this model. Got {self.solved_model}."}
            _name = ""
            for idx, comb in enumerate(combination):
                if idx < len(combination)-1 : _name += f"{_get_key(self.column_dict, comb)}, "
                else: _name += f"and {_get_key(self.column_dict, comb)}"
            print(f"Elapsed time for {_name} was {toc - tic:0.4f} seconds")
            if verbose:
                print(self.model.solve_details)
                self.solved_model.display()
                self.model.export_as_lp("./")
            self._remove_multi_constraints(combination)
            self.solved_graph = nx.DiGraph()
            df_solved_model = self.solved_model.as_df()
            self.df_solved_model_list = [(key.split('_'), value) for key, value in df_solved_model.values if round(value) > 0]
            self._data_locations = [key for key, _ in self.df_solved_model_list]
            
            # Get the distance for all cities between all cities as our cost edges.
            for i in range(len(self._data_locations)):
                if 'f' in self._data_locations[i]:
                    first_node = ast.literal_eval(self._data_locations[i][1])
                    second_node = ast.literal_eval(self._data_locations[i][2])
                    # Eucledian distance calculation.
                    distance = np.round(((first_node[0] - second_node[0])**2 + (first_node[1] - second_node[1])**2)**0.5)
                    # Add the edge to a graph with the distance as an edge weight.
                    self.solved_graph.add_edge(first_node, second_node, weight=distance)
            
            self.org_df = self.create_dataframe(self.org_df, name = _name)
            plot_names.append(_name)
            if plot_graph: _figures.append(self.plot_graph(f"Solution for {_name}"))
        self.org_df.set_index('Objective Name', inplace=True)
        if plot_graph:
            _num_of_cols_rows = len(all_double_combinations)//2
            if _num_of_cols_rows == 0: 
                _figures[0].show()
                return self.org_df
            _total_fig = make_subplots(rows = _num_of_cols_rows, cols = _num_of_cols_rows, shared_xaxes= False, start_cell="top-left", subplot_titles=plot_names, vertical_spacing=0.1, horizontal_spacing=0.1)
            col_num = 1
            row_num = 1
            for idx, fig in enumerate(_figures):
                if col_num > _num_of_cols_rows: 
                    col_num = 1
                    row_num = 2
                for item in fig['data']:
                    if idx > 0: item['showlegend'] = False
                    _total_fig.add_trace(item, row = row_num, col = col_num)
                col_num += 1
            _total_fig.show()
        return self.org_df

if __name__ == '__main__':
    set_seed = 0
    RandomGraph = Graph(8,baseline=True,plot_graph=False, seed=set_seed)
    parameters = Parameters(RandomGraph, set_seed)
    model_baseline = Model_Baseline(parameters, plot_graph=False, verbose=False)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_health_impact]
    data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(parameters, df = data) 
    df = mo.solve_multi_objective()
    print(df)