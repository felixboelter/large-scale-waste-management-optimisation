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
IntFloat = TypeVar("IntFloat", int, float)

class Parameters():
    """
    Class Parameters. Creates a all the parameters used in the baseline.
    Parameter amounts based on: Olapiriyakul, Sun & Pannakkong, Warut &
    Kachapanya, Warith & Starita, Stefano. (2019). Multiobjective Optimization
    Model for Sustainable Waste Management Network Design. Journal of Advanced
    Transportation. 2019. 1-15. 10.1155/2019/3612809. 
    """
    def __init__(self, G: Graph, seed : IntFloat = 0) -> None:
        """
        Parameters Constructor. 
        Public access to parameters:
        facility_storage_capacities : np.ndarray, maximum_amount_transport : np.ndarray, 
        operational_costs : np.ndarray, land_stress_ratios : np.ndarray, 
        opening_costs : np.ndarray, link_populations : Dict[tuple, int],
        population_near_facilities : Dict[tuple, np.float64], link_dalys : Dict[tuple, float],
        facility_daly_per_person : Dict[tuple, List[float]]

        :param G: Graph object from generate_graph.py
        :param seed: The seed to generate random numbers from.
        """
        np.random.seed(seed)
        self._G = G
        self._sorting_facilities = [k for k, v in self._G.special_locations.items() if 'J' in v]
        self._incinerator_facilities = [k for k, v in self._G.special_locations.items() if 'K' in v]
        self._landfill_facilities = [k for k, v in self._G.special_locations.items() if "K'" in v]
        self._direct_land_usage =  np.array([4800,8000,16000, 8000,16000, 24000, 80000,160000,192000]).reshape(3,3)
        self._indirect_land_usage =  np.array([6191,10780,21559, 11971,23941,35911, 127770,255525,335295]).reshape(3,3)
        self._range_of_facility_sizes = range(0, self._direct_land_usage.shape[1])
        self.facility_storage_capacities : np.ndarray= self._G.add_custom_parameter(name= 'facility_storage', size=(3,3), fixed_number=[50,100,300,50,100,150,50,100,150])
        self.maximum_amount_transport : np.ndarray= self._G.add_custom_parameter(name = 'maximum_amount_transport', size = 2, fixed_number=[16,32])
        _random_increasing = [np.random.randint(low=l,high=h) for l,h in [(100000,200000),(200000,400000),(100000,200000)]]
        self.operational_costs : np.ndarray = self._G.add_custom_parameter(name='operational_costs',size=(3),fixed_number=_random_increasing)
        self.land_stress_ratios : np.ndarray = self._create_land_usage_stress_ratios()
        self.opening_costs : np.ndarray = self._create_opening_costs()
        self.link_populations : Dict[tuple, int] = self._create_population_near_links()
        self.population_near_facilities : Dict[tuple, np.float64] = self._create_population_near_facilities()
        self.link_dalys : Dict[tuple, float] = self._create_DALY_for_links()
        self.facility_daly_per_person : Dict[tuple, List[float]] = self._create_DALY_for_facilities()

    def _create_DALY_for_facilities(self) -> Dict[tuple, List[float]]:
        """
        Helper function.
        Creates DALY per person values for all facilities of length l.
        :return: DALYs per person for all facilities of length l.
        """
        facility_daly = np.array([[0.07, 0.14, 0.28],
                                [5.95, 11.9, 17.85],
                                [3.89, 7.78, 11.66]])
        daly_per_person = dict()
        for ix, facilities in enumerate([self._sorting_facilities, self._incinerator_facilities, self._landfill_facilities]): 
            for node in facilities:
                daly_per_person.update({node: [facility_daly[ix][l] for l in self._range_of_facility_sizes]})
        self._G.custom_parameters['facility_daly_per_person'] = daly_per_person
        return daly_per_person

    def _create_DALY_for_links(self) -> Dict[tuple, float]:
        """
        Helper function.
        Creates DALY per person values for all links (i,j), (j,k), and (j,k').
        :return: DALYs per person for all links (i,j), (j,k), and (j,k').
        """
        DALY_per_vehicle = {16 : [5.82e-07, 5.62e-08],
                            32:  [1.16e-06, 1.12e-07]}
        link_DALY = dict()
        for i,j,w in self._G.G.edges(data=True):
            weight = w['weight']
            if i == j: link_DALY[(i,j)] = (DALY_per_vehicle[16][0], DALY_per_vehicle[32][0])
            else: link_DALY[(i,j)] = (DALY_per_vehicle[16][1] * weight, DALY_per_vehicle[32][1] * weight)
        self._G.custom_parameters['daly_per_person_links'] = link_DALY
        return link_DALY

    def _create_land_usage_stress_ratios(self) -> np.ndarray:
        """
        Helper function.
        Creates the land usage stress ratios according to the formula given in Olapiriyakul, Sun & Pannakkong, Warut & Kachapanya, Warith & Starita, Stefano. (2019). P.7
        :return: Land stress ratios of all facilities of size l.
        """
        land_stress_ratios = (self._direct_land_usage+self._indirect_land_usage)/100**2
        self._G.custom_parameters['land_stress_ratios'] = land_stress_ratios
        return land_stress_ratios
    
    def _create_opening_costs(self) -> np.ndarray:
        """
        Helper function. Creates opening costs for all facilities of size l.
        Data for costs gathered from Azienda cantonale dei rifiuti Giubiasco
        (2010), Elrabaya, Daker & Marchenko, Valentina. (2021) 
        :return: Opening costs for all facilities of size l.
        """
        # Elrabaya, Daker & Marchenko, Valentina. (2021). Landfill development cost of 6'482'949 in 2019. Inflation adjusted through (https://www.bls.gov/data/inflation_calculator.htm : 7'241'085.63)
        _SHARJAH_LANDFILL_M_SQRD = 126500
        _LANDFILL_DEVELOPMENT_COST_DOLLAR= 7241085.63
        # Exchange according to (https://www.bloomberg.com/quote/USDCHF:CUR : 0.9952 May 2022)
        _LANDFILL_DEVELOPMENT_COST_CHF = _LANDFILL_DEVELOPMENT_COST_DOLLAR * 0.9952
        _LANDFILL_COST_PER_M_SQRD = _LANDFILL_DEVELOPMENT_COST_CHF/_SHARJAH_LANDFILL_M_SQRD
        _opening_cost_landfill = lambda x: x * _LANDFILL_COST_PER_M_SQRD
        # Estimate for equipment cost (Conveyor belt, Air seperation system, etc.)
        EQUIPMENT_COST = 500000
        _opening_cost_sorting = lambda x: (x * _LANDFILL_COST_PER_M_SQRD)+EQUIPMENT_COST
        # Giubiasco incinerator cost: 40 Million CHF in 2004 (Inflation adjusted through: https://lik-app.bfs.admin.ch/en/lik/rechner : 42’050’290 CHF) for an area of 40,000 m^
        _opening_cost_incinerator = lambda x: x * (42050290/40000)
        _opening_cost_list = np.array([function(self._direct_land_usage[ix]) for ix,function in enumerate([_opening_cost_sorting,_opening_cost_incinerator,_opening_cost_landfill])]).tolist()
        opening_costs = self._G.add_custom_parameter(name='opening_costs',size=(3,3), fixed_number=_opening_cost_list)
        return opening_costs
    
    def _create_population_near_links(self) -> Dict[tuple, int]:
        """
        Helper function. Creates population for all edges in graph (i,j). 
        Based on the population at location i and j and the distance between them.
        :return: Population for all edges in graph G.
        :rtype: Dictionary[(i,j), int]
        """
        link_populations = dict()
        HIGH = 0.05
        LOW = 0.01
        for i,j,w in self._G.G.edges(data=True):
            weight = w['weight']
            random_num = np.random.random_sample() * (HIGH - LOW) + LOW
            if i == j: link_populations[(i,j)] = int(self._G.city_population[i])
            else: link_populations[(i,j)] = round(((self._G.city_population[i] * random_num) + (self._G.city_population[j] * random_num)) * (weight * (1-random_num)))
        self._G.custom_parameters['population_near_links'] = link_populations
        return link_populations
    
    def _create_population_near_facilities(self) -> Dict[tuple, np.float64]:
        """
        Helper function. Creates population for all nodes in graph G. 
        Based on the population at node i and the amount of space the facility is using up in km^2.
        :return: Subpopulation for all nodes in graph G.
        :rtype: Dictionary[(i,j), np.float64]
        """
        people_living_near_facility = dict()
        m_sqr_to_km_sqr = lambda m_sqr: m_sqr / 1e6
        for node in self._G.G.nodes: people_living_near_facility.update({node : np.round(self._G.city_population[node] * m_sqr_to_km_sqr(self._direct_land_usage))})
        self._G.custom_parameters['people_living_near_facility'] = people_living_near_facility
        return people_living_near_facility

class Model_Baseline(Parameters):
    """
    Class Model_Baseline, inherits Parameters.
    Creates a baseline model object, which is based on Olapiriyakul, Sun & Pannakkong, Warut & Kachapanya, Warith & Starita, Stefano. (2019).
    """
    def __init__(self, G: Graph, seed : IntFloat = 0, plot_graph : Optional[bool] = False,verbose : Optional[bool] = False) -> None:
        """
        Model_Baseline constructor. Public access to solved_model : docplex.mp.solution.SolveSolution | None
        :param G: Graph object from generate_graph.py
        :param seed: The seed to generate random numbers from.
        """
        super().__init__(G, seed)
        self.model = Model(name="Baseline")
        self._verbose = verbose
        self._plot_graph = plot_graph
        self._ij_list = [(i,j,w['weight']) for i, j, w in self._G.G.edges(data=True) if i in self._G.collection_locations and j in self._sorting_facilities]
        self._jk_list = [(j,k,w['weight']) for j, k, w in self._G.G.edges(data=True) if (j in self._sorting_facilities and k in self._incinerator_facilities)]
        self._jkp_list = [(j,kp,w['weight']) for j, kp, w in self._G.G.edges(data=True) if j in self._sorting_facilities and kp in self._landfill_facilities]
        self._S = 0
        self._I = 1
        self._L = 2
        self._create_decision_variables()
        self._sorting_tuple = (self._S, self._y_sorting, self._sorting_facilities)
        self._incinerator_tuple = (self._I, self._y_incinerator, self._incinerator_facilities)
        self._landfill_tuple = (self._L, self._y_landfill, self._landfill_facilities)
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
        self._y_sorting = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._sorting_facilities for j in self._range_of_facility_sizes}
        self._y_incinerator = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._incinerator_facilities for j in self._range_of_facility_sizes}
        self._y_landfill = {(i,j): self.model.binary_var(name=f"y_{i}_{j}") for i in self._landfill_facilities for j in self._range_of_facility_sizes}
        self._x_ij = {(i,j) : self.model.integer_var(name=f'x_{i}_{j}') for i,j,_ in self._ij_list}
        self._x_jk = {(j,k) : self.model.integer_var(name=f'x_{j}_{k}') for j,k,_ in self._jk_list}
        self._x_jkp = {(j,kp) : self.model.integer_var(name=f'x_{j}_{kp}') for j,kp,_ in self._jkp_list}
        self._f_ij = {(i,j) : self.model.continuous_var(name=f'f_{i}_{j}') for i,j,_ in self._ij_list}
        self._f_jk = {(j,k) : self.model.continuous_var(name=f'f_{j}_{k}') for j,k,_ in self._jk_list}
        self._f_jkp = {(j,kp) : self.model.continuous_var(name=f'f_{j}_{kp}') for j,kp,_ in self._jkp_list}
    
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
            Private helper function. Creates a dictionary out of an array.
            :param array: List with tuple, to be made a dictionary out of.
            :return: The created dictionary
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
            Private helper function, used for the creation of constraint 3,4, and 5. 

            :param type_of_constraint: 0 for Sorting, 1 for Incinerator, and 2 for Landfill 
            :param y_variable: The y decision variable to be used. 
            :param size_dependent_f: List of sums of the decision variable f: 
            collection points i for sorting, sorting facilities j for incinerators, sorting facilities j for landfills.
            :param facilities: Facility locations to loop over.
            """
            size_capacity= [self.model.sum(self.facility_storage_capacities[type_of_constraint][l] * y_variable[(j,l)] \
                 for l in self._range_of_facility_sizes) for j in facilities]
            for ix in range(0, len(size_dependent_f)):
                self.model.add_constraint(size_dependent_f[ix] <= size_capacity[ix])
       
        def _transportation_capacity_limitations(edge_list : list, x_variable : Model.integer_var, f_variable : Model.continuous_var) -> None:
            """
            Private helper function, used for the creation of constraint 6,7, and 8. 
            
            :param edge_list: Edges to loop over (i,j), (j,k), or (j,k')
            :param x_variable: The x decision variable to be used. 
            :param f_variable: The f decision variable to be used.
            """
            for i,j,_ in edge_list:
                for l in range(0,len(self.maximum_amount_transport)):
                    self.model.add_constraint(f_variable[(i,j)] <= self.maximum_amount_transport[l] * x_variable[(i,j)])
        
        def _one_size_selection_constraint(facilities : list, y_variable : Model.binary_var) -> None:
            """

            Private helper function, used for the creation of constraint 9, 10, and 11. 
            :param facilities: Facility locations to loop over.
            :param y_variable: The y decision variable to be used. 
            """
            for i in facilities:
                self.model.add_constraint(self.model.sum(y_variable[(i,l)] for l in self._range_of_facility_sizes) <= 1)
        
        _ij_for_i_dict = _dictionary_array(self._ij_list)
        _jk_for_j_dict = _dictionary_array(self._jk_list)
        _jkp_for_j_dict = _dictionary_array(self._jkp_list)
        # Inverse dictionaries
        _ij_for_j_dict = _dictionary_array(self._ij_list,inverse=True)
        _jk_for_k_dict = _dictionary_array(self._jk_list,inverse=True)
        _jkp_for_kp_dict = _dictionary_array(self._jkp_list,inverse=True)

        # Constraints 
        # Constraint 1: Outflow of waste from any collection center i must be equal to the amount of available waste at i
        for key, val in _ij_for_i_dict.items():
            self.model.add_constraint(self.model.sum(self._f_ij[(key,j)] for j in val) == self._G.supplies[key][0])
        # Constraint 2: Flow balance, inflow of waste at sorting facility j is ENTIRELY forwarded to incinerator k or landfill k'
        _sum_f_ij_for_j= [self.model.sum(self._f_ij[(i,j)] for i in i_values) for j, i_values in _ij_for_j_dict.items()]
        _sum_f_jk_for_j = [self.model.sum(self._f_jk[(j,k)] for k in k_values) for j, k_values in _jk_for_j_dict.items()]
        _sum_f_jkp_for_j = [self.model.sum(self._f_jkp[(j,kp)] for kp in kp_values) for j, kp_values in _jkp_for_j_dict.items()] 
        for ix in range(0, len(_sum_f_jk_for_j)):
            self.model.add_constraint(_sum_f_ij_for_j[ix] == _sum_f_jk_for_j[ix])
        for ix in range(0, len(_sum_f_jkp_for_j)):
            self.model.add_constraint(_sum_f_ij_for_j[ix] == _sum_f_jkp_for_j[ix])
        # Constraint 3: Size dependent capacity constraint for sorting facilities
        _size_dependent_capacity_constraint(self._S, self._y_sorting, _sum_f_ij_for_j, self._sorting_facilities)
        # Constraint 4: Size dependent capcity constraint for incinerator facilities
        _sum_f_jk_for_k = [self.model.sum(self._f_jk[(j,k)] for j in j_value) for k, j_value in _jk_for_k_dict.items()] 
        _size_dependent_capacity_constraint(self._I, self._y_incinerator, _sum_f_jk_for_k, self._incinerator_facilities)
        # Constraint 5: Size dependent capcity constraint for landfill facilities
        _sum_f_jk_for_kp = [self.model.sum(self._f_jkp[(j,kp)]for j in j_value) for kp, j_value in _jkp_for_kp_dict.items()] 
        _size_dependent_capacity_constraint(self._L, self._y_landfill, _sum_f_jk_for_kp, self._landfill_facilities)
        # Constraint (6, 7, 8) : Transportation capacity limitations for ((i, j), (j, k), (j, kp))
        _ij_tuple = (self._ij_list, self._x_ij, self._f_ij)
        _jk_tuple = (self._jk_list, self._x_jk, self._f_jk)
        _jkp_tuple = (self._jkp_list, self._x_jkp, self._f_jkp)
        for edge_list, x_decision_value, f_decision_value in [_ij_tuple, _jk_tuple, _jkp_tuple]:
            _transportation_capacity_limitations(edge_list, x_decision_value, f_decision_value)
        # Constraint (9, 10, 11): One size selection for (sorting, incinerator, landfill) facilities
        _sorting_tuple = (self._sorting_facilities, self._y_sorting)
        _incinerator_tuple = (self._incinerator_facilities, self._y_incinerator)
        _landfill_tuple = (self._landfill_facilities, self._y_landfill)
        for facilities, decision_values in [_sorting_tuple, _incinerator_tuple, _landfill_tuple]:
            _one_size_selection_constraint(facilities, decision_values)
    
    def minimize_cost(self, minimize : Optional[bool] = True) -> None:
        """
        Public objective minimization function Fc. 
        Minimizes the overall cost which is the sum of the fixed costs to open a facility of size l plus the operational
        costs of transporting and managing solid waste flow across the network. Subsequently, solves the model according to this function.
        :return: The solved model.
        """
        def _create_opening_costs_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            Private helper function. Used to create the linear expression of sums to open a facility of size l.

            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill.
            :param y_variable: The y decision variable to be used. 
            :param facilities: Facility locations to loop over.
            :return: The opening costs linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum(y_variable.get((i,j)) * self.opening_costs[type_of_facility][j] for i in facilities for j in self._range_of_facility_sizes)
        
        def _create_operational_costs_sum(type_of_facility : int, x_variable : Model.integer_var, edge_list : list) -> docplex.mp.linear.LinearExpr:
            """
            Private helper function. Used to create the linear expression of sums of costs to transport and manage the solid waste flow.

            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill.
            :param x_variable: The x decision variable to be used. 
            :param edge_list:  Edges to loop over (i,j), (j,k), or (j,k')
            :return: The operational costs linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum((t_ij + self.operational_costs[type_of_facility]) * x_variable[(i,j)] for i, j, t_ij in edge_list)

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
        Public objective minimization function Fu.
        Used to measure the average land-use stress. Sum of all land-use ratios across all candidate locations.
        Subsequently, solves the model.
        :return: The solved model.
        """
        def _create_land_usage_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            Private helper function. Used to create the land usage linear expression.

            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill.
            :param y_variable: The y decision variable to be used. 
            :param facilities: Facility locations to loop over.
            :return: The land usage linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum(self.land_stress_ratios[type_of_facility][l] * y_variable[(i,l)] for i in facilities for l in self._range_of_facility_sizes)
        _land_usage_objectives = [_create_land_usage_sum(tuple_[0], tuple_[1], tuple_[2]) for tuple_ in [self._sorting_tuple, self._incinerator_tuple, self._landfill_tuple]]
        _total_land_usage_sum = _land_usage_objectives[self._S] + _land_usage_objectives[self._I] + _land_usage_objectives[self._L]
        self.minimization = "Land Usage Objective"
        if minimize == False: return _total_land_usage_sum
        self.model.minimize(_total_land_usage_sum)

    def minimize_health_impact(self, minimize : Optional[bool] = True) -> None:
        """
        Public objective minimization function Fh.
        Minimize the impact of transportation and facilities on the population's health.
        Subsequently, solve the model.
        :return: The solved model.
        """
        def _create_facility_health_impact_sum(type_of_facility : int, y_variable : Model.binary_var, facilities : list) -> docplex.mp.linear.LinearExpr:
            """
            Private helper function. Creates the linear expression for the facility health impact.

            :param type_of_facility: 0 for Sorting, 1 for Incinerator, and 2 for Landfill.
            :param y_variable: The y decision variable to be used. 
            :param facilities: Facility locations to loop over.
            :return: The facility health impact linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum(self.population_near_facilities[i][type_of_facility][l] * self.facility_daly_per_person[i][l] * y_variable[(i,l)] for i in facilities for l in self._range_of_facility_sizes)
        def _create_transport_health_impact_sum(x_variable : Model.integer_var, edge_list : list) -> docplex.mp.linear.LinearExpr:
            """
            Private helper function. Creates the linear expression for the transportation health impact.

            :param x_variable: The x decision variable to be used. 
            :param edge_list:  Edges to loop over (i,j), (j,k), or (j,k')
            :return: The transportation health impact linear expression from the class docplex.mp.linear.LinearExpr
            """
            return self.model.sum(self.link_populations[(i,j)]  * self.link_dalys[(i,j)][1] * x_variable[(i,j)] for i,j,_ in edge_list)
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
        Public function to solve the created model.
        :return: The solved model.
        """
        if not isinstance(list_of_functions, list): list_of_functions = [list_of_functions]
        df = pd.DataFrame(columns=["Objective Name","Cost Objective", "Land Usage Objective", "Health Impact Objective"])
        _figs = []
        minimization_names = []
        for minimize_function in list_of_functions:
            minimize_function()
            minimization_names.append(self.minimization)
            self.model.print_information()
            self.solved_model = self.model.solve()
            assert self.solved_model, {f"Solution could not be found for this model. Got {self.solved_model}."}
            if self._verbose:
                self.solved_model.display()
                self.model.export_as_lp("./")
            self.model.remove_objective()
            self.solved_graph = nx.DiGraph()
            df_solved_model = self.solved_model.as_df()
            self.df_solved_model_list = [(key.split('_'), value) for key, value in df_solved_model.values if round(value) > 0]
            self._data_locations = [key for key, _ in self.df_solved_model_list]
            # Get the distance for all cities between all cities as our cost edges.
            for i in range(len(self._data_locations)):
                if 'f' in self._data_locations[i]:
                    first_node = eval(self._data_locations[i][1])
                    second_node = eval(self._data_locations[i][2])
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
        return df
    
    def _calculate_cost(self, y_decisions : List[Tuple[Any, int]], x_decisions : Dict[tuple, Any]) -> Tuple[float, float]:
        _opening_cost = 0
        _opening_cost_array = self.opening_costs
        for j, l in y_decisions:
            if j in self._sorting_facilities:  _opening_cost += _opening_cost_array[0][l]
            elif j in self._incinerator_facilities: _opening_cost += _opening_cost_array[1][l]
            elif j in self._landfill_facilities: _opening_cost += _opening_cost_array[2][l]
        _operational_cost = 0
        _operational_cost_array = self.operational_costs
        for index, edge_list in enumerate([self._ij_data, self._jk_data, self._jkp_data]):
            for i, j, t_ij in edge_list:
                _operational_cost += (t_ij + _operational_cost_array[index]) * x_decisions[(i,j)]
        return (_opening_cost, _operational_cost)
    
    def _calculate_land_usage(self, y_decisions : List[Tuple[Any, int]]) -> float:
        land_usage = 0
        land_stress_ratio_array = self.land_stress_ratios
        for j, l in y_decisions:
            if j in self._sorting_facilities:  land_usage += land_stress_ratio_array[0][l]
            elif j in self._incinerator_facilities: land_usage += land_stress_ratio_array[1][l]
            elif j in self._landfill_facilities: land_usage += land_stress_ratio_array[2][l]
        return land_usage

    def _calculate_health_impact(self, y_decisions : List[Tuple[Any, int]], x_decisions : Dict[tuple, Any]) -> Tuple[float, float]:
        facility_health_impact = 0
        pop_near_facilities = self.population_near_facilities
        facility_dalys = self.facility_daly_per_person
        for j, l in y_decisions:
            if j in self._sorting_facilities:  facility_health_impact += pop_near_facilities[j][0][l] * facility_dalys[j][l]
            elif j in self._incinerator_facilities: facility_health_impact += pop_near_facilities[j][1][l] * facility_dalys[j][l]
            elif j in self._landfill_facilities: facility_health_impact += pop_near_facilities[j][2][l] * facility_dalys[j][l]

        transport_health_impact = 0
        population_for_edge = self.link_populations
        daly_for_edge = self.link_dalys
        for edge_list in [self._ij_data, self._jk_data, self._jkp_data]:
            for i,j,_ in edge_list:
                transport_health_impact += population_for_edge[(i,j)] * daly_for_edge[(i,j)][1] * x_decisions[(i,j)] 
        return (transport_health_impact, facility_health_impact)

    def create_dataframe(self, df : pd.DataFrame, name : Optional[str] = None) -> pd.DataFrame:
        self._ij_data = [(i,j,w['weight']) for i, j, w in self.solved_graph.edges(data=True) if i in self._G.collection_locations and j in self._sorting_facilities]
        self._jk_data = [(j,k,w['weight']) for j, k, w in self.solved_graph.edges(data=True) if j in self._sorting_facilities and k in self._incinerator_facilities]
        self._jkp_data = [(j,kp,w['weight']) for j, kp, w in self.solved_graph.edges(data=True) if j in self._sorting_facilities and kp in self._landfill_facilities]
        _y_decisions = [(eval(key[1]), int(key[2])) for key, _ in self.df_solved_model_list if 'y' in key]
        _x_decisions = {(eval(key[1]), eval(key[2])): value for key, value in self.df_solved_model_list if 'x' in key}
        _cost_objective : Tuple[float, float] = self._calculate_cost(_y_decisions, _x_decisions)
        _land_objective : float = self._calculate_land_usage(_y_decisions)
        _health_objective : Tuple[float, float] = self._calculate_health_impact(_y_decisions, _x_decisions)
        obj_name = self.minimization
        if name is not None: obj_name = name
        df.loc[len(df.index)] = [obj_name, f"{sum(_cost_objective):8f}", _land_objective, f"{sum(_health_objective):8f}"]
        return df

    def plot_graph(self, figure_name : Optional[str] = None):
        def _edge_colours(G, value1: int, value2: int):
            """
            private helper Function _edge_colours
            Creates the edge colours based on the edge weights
            :param value1: The low value of the range.
            :param value2: The high value of the range.
            :type value1: int
            :type value2: int
            :return: Edge X coordinates, Edge Y coordinates, Color HEX, line width, Name in Legend
            :rtype: list, list, str, int, str
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
                                text= f"<b>Total Unsorted Supply:</b> {sum([us for us,_  in self._G.supplies.values()])}",
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
        _data_locations_y = {eval(key[1]):key[2] for key in self._data_locations if 'y' in key}
        dictionary_values = {0 : "Small", 1: "Medium", 2 : "Large"}
        for node in self.solved_graph.nodes():
            if node in self._G.collection_locations: 
                _node_colours[node] = ["Collection Center", '#D7D2CB']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Unsorted Supply: {self._G.supplies[node][0]}"
            elif node in self._sorting_facilities: 
                _node_colours[node] = ["Sorting Facility", '#6AC46A']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(_data_locations_y[node])]}"
            elif node in self._incinerator_facilities: 
                _node_colours[node] = ["Incinerator Facility", '#952E25']
                _custom_node_attrs[node] = f"Node: {node} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(_data_locations_y[node])]}"
            elif node in self._landfill_facilities: 
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
    def __init__(self, Graph : nx.Graph, df : pd.DataFrame, seed : IntFloat = 0) -> None:
        """
        Multiobjective_model constructor.
        Initializes the class, creates the deviations, and adds the minimizing variable.
        :param Graph: The generated graph created by generate_graph.graph
        :param df: The pandas dataframe from the single optimizations.
        :seed: The seed for random number generation.
        """
        super().__init__(Graph, seed)
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
        Private helper function.
        Adds the constraint that the deviation must be smaller or equal to the continuous variable z.
        :param combination: A combination created by itertools.combinations
        """
        for comb in combination:
            self.model.add_constraint(self.deviations[comb] <= self._z, ctname = f"constraint_{comb}")
    def _remove_multi_constraints(self, combination: Tuple[int, int]) -> None:
        """
        Private helper function.
        Removes the constraints created in _create_multi_constraints() such that there is no overlap in constraints.
        :param combination: A combination created by itertools.combinations
        """
        self.model.remove_objective()
        for comb in combination:
            self.model.remove_constraint(f"constraint_{comb}")

    def solve_multi_objective(self, plot_graph : Optional[bool] = False, verbose : Optional[bool] = False) -> pd.DataFrame:
        """
        Public function.
        Solves the multi objective min-max problem.
        Goes through all combinations of single optimization functions and solves each one.
        :param plot_graph: Plot the created graph for each combination.
        :param verbose: Displays the model results and exports the problem to lp.
        :return: The dataframe object which includes all of the calculations for the various objectives.
        """
        def _get_key(dict : Dict[str, list], val : int) -> str:
            """
            Private helper function.
            Used to get a key from a value array in a dictionary.
            :param dict: The dictionary which is looped through.
            :param val: The value which is searched for.
            :return: The key of the value.
            """
            for key, value in dict.items():
                if val in value:
                    return key
        _figures = []
        plot_names = []
        all_double_combinations = list(combinations([i for i in range(0,len(self._optimal_values))], 2))
        if len(self._optimal_values) == 3: all_double_combinations.append((0,1,2))
        for combination in all_double_combinations:  
            self.model.minimize(self._z)
            self._create_multi_constraints(combination)
            self.model.print_information()
            self.solved_model = self.model.solve()
            assert self.solved_model, {f"Solution could not be found for this model. Got {self.solved_model}."}
            if verbose:
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
                    first_node = eval(self._data_locations[i][1])
                    second_node = eval(self._data_locations[i][2])
                    # Eucledian distance calculation.
                    distance = np.round(((first_node[0] - second_node[0])**2 + (first_node[1] - second_node[1])**2)**0.5)
                    # Add the edge to a graph with the distance as an edge weight.
                    self.solved_graph.add_edge(first_node, second_node, weight=distance)
            _name = ""
            for idx, comb in enumerate(combination):
                if idx < len(combination)-1 : _name += f"{_get_key(self.column_dict, comb)}, "
                else: _name += f"and {_get_key(self.column_dict, comb)}"
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
    model_baseline = Model_Baseline(RandomGraph, plot_graph=False, seed=set_seed, verbose=False)
    list_of_functions = [model_baseline.minimize_cost, model_baseline.minimize_health_impact]
    data = model_baseline.solve_model(list_of_functions)
    mo = Multiobjective_model(RandomGraph, df = data, seed=set_seed) 
    df = mo.solve_multi_objective()
    print(df)