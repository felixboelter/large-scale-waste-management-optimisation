from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
import numpy as np
from parameters import Parameters
import pymoo
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems.constr_as_penalty import ConstraintsAsPenalty


class Vectorized_heuristic(Problem):
    def __init__(self, parameters : Parameters, **kwargs) -> None:
        """
        The function creates the variables for the problem, and then creates the problem with the number
        of variables, objectives, and constraints
        
        :param parameters: The parameters object that contains all the data for the problem
        :type parameters: Parameters
        """

        self._parameters = parameters
        self.num_binary_vars, self.num_integer_vars, self.num_continuous_vars = self._create_variables()
        _n_constr = len(self.supplies) + (len(self._ij_list)//len(self.supplies) ) *2 + len(self._parameters._sorting_facilities)*2 + len(self._parameters._incinerator_facilities)*2 + len(self._parameters._landfill_facilities) * 2 + len(self._ij_list)* len(self._parameters.maximum_amount_transport) + len(self._jk_list)* len(self._parameters.maximum_amount_transport) + len(self._jkp_list)* len(self._parameters.maximum_amount_transport)

        print(f"Number of Variables: {self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars}")
        print(f"Binary: {self.num_binary_vars},  Integer: {self.num_integer_vars},   Continuous: {self.num_continuous_vars}")

        print(f"Number of constraints: {_n_constr}.")
        xu_bin = np.ones(self.num_binary_vars)
        xu_int = np.ones(self.num_integer_vars) * self._parameters._G._number_of_cities
        xu_cont = np.ones(self.num_continuous_vars) * np.sum(self.supplies)
        super().__init__(n_var = self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars,
                                    n_obj = 3,
                                    n_constr = _n_constr,
                                    xl=0,
                                    xu=np.concatenate([xu_bin,xu_int,xu_cont]),
                                    **kwargs)
    def _evaluate(self, x, out, *args, **kwargs):
        """
        The function takes in the decision variables, and returns the objective function values and the
        constraint violation
        :param x: the decision variables
        :param out: the output dictionary
        """
        # Objective 1
        _sorting_x = x[:,self.binary_sorting_slice]
        _incinerator_x = x[:,self.binary_incinerator_slice]
        _landfill_x = x[:,self.binary_landfill_slice]
        _facility_lengths = [len(self._parameters._sorting_facilities), len(self._parameters._incinerator_facilities), len(self._parameters._landfill_facilities)]
        _opening_costs = [np.hstack((self._parameters.opening_costs[i],) * l) for i,l in enumerate(_facility_lengths)]

        _sorting_opening_costs = x[:,self.binary_sorting_slice] @ _opening_costs[0]
        _incinerator_opening_costs = x[:,self.binary_incinerator_slice] @ _opening_costs[1]
        _landfill_opening_costs = x[:,self.binary_landfill_slice] @ _opening_costs[2]
        total_opening_cost = _sorting_opening_costs + _incinerator_opening_costs + _landfill_opening_costs
        _ij_operational_cost = x[:,self.integer_ij_slice] @ (self._ij_list + self._parameters.operational_costs[0]) 
        _jk_operational_cost = x[:,self.integer_jk_slice] @ (self._jk_list + self._parameters.operational_costs[1]) 
        _jkp_operational_cost = x[:,self.integer_jkp_slice] @ (self._jkp_list + self._parameters.operational_costs[2])
        total_operational_cost = _ij_operational_cost + _jk_operational_cost + _jkp_operational_cost
        objective_1 = total_opening_cost + total_operational_cost
        # Objective 2
        _land_stress_ratios = [np.hstack((self._parameters.land_stress_ratios[i],) * l) for i,l in enumerate(_facility_lengths)]
        _sorting_land_usage = x[:,self.binary_sorting_slice] @ _land_stress_ratios[0]
        _incinerator_land_usage = x[:,self.binary_incinerator_slice] @ _land_stress_ratios[1]
        _landfill_land_usage = x[:,self.binary_landfill_slice] @ _land_stress_ratios[2]
        objective_2 = _sorting_land_usage + _incinerator_land_usage + _landfill_land_usage

        #Objective 3
        # _population_list = [np.hstack((self._parameters.population_list[i],) * l) for i,l in enumerate(_facility_lengths)]
        _facility_daly_matrix = [np.hstack((self._parameters.facility_daly_matrix[i],) * l) for i,l in enumerate(_facility_lengths)]
        _sorting_health_impact = np.sum(x[:,self.binary_sorting_slice] * self._parameters.population_list[0].reshape(-1) * _facility_daly_matrix[0], axis = 1)
        _incinerator_health_impact = np.sum(x[:,self.binary_incinerator_slice] * self._parameters.population_list[1].reshape(-1) * _facility_daly_matrix[1], axis = 1)
        _landfill_health_impact = np.sum(x[:,self.binary_landfill_slice] * self._parameters.population_list[2].reshape(-1) * _facility_daly_matrix[2], axis = 1)
        total_facility_health_impact = _sorting_health_impact + _incinerator_health_impact + _landfill_health_impact

        _sorting_link_health_impact = self._parameters.link_populations_list[0] * self._parameters.link_dalys_list[0] * x[:,self.integer_ij_slice]
        _incinerator_link_health_impact = self._parameters.link_populations_list[1] * self._parameters.link_dalys_list[1] * x[:,self.integer_jk_slice]
        _landfill_link_health_impact = self._parameters.link_populations_list[2] * self._parameters.link_dalys_list[2] * x[:,self.integer_jkp_slice]
        total_link_health_impact = np.sum(_sorting_link_health_impact, axis=1) + np.sum(_incinerator_link_health_impact, axis = 1) + np.sum(_landfill_link_health_impact, axis=1)
        objective_3 = total_facility_health_impact + total_link_health_impact
        # Constraints
        _num_sorting_facilities = len(self._parameters._sorting_facilities)
        _num_collection_facilities = len(self._parameters._G.collection_locations)
        _num_incinerator_facilities = len(self._parameters._incinerator_facilities)
        _num_landfill_facilities = len(self._parameters._landfill_facilities)
        constraint_1 = np.abs(np.sum(x[:,self.continuous_ij_slice].reshape(-1,_num_sorting_facilities,_num_collection_facilities),axis=1) - self.supplies)
        # print(x[:,self.continuous_ij_slice].reshape(-1,_num_sorting_facilities,_num_collection_facilities).shape)
        _ij_x_sum_for_j = np.sum(x[:,self.continuous_ij_slice].reshape(-1,_num_sorting_facilities,_num_collection_facilities), axis = 2)
        _jk_x_sum_for_j = np.sum(x[:,self.continuous_jk_slice].reshape(-1,_num_sorting_facilities,_num_incinerator_facilities), axis = 2)
        _jkp_x_sum_for_j = np.sum(x[:,self.continuous_jkp_slice].reshape(-1,_num_sorting_facilities, _num_landfill_facilities), axis = 2)
        constraint_2 = np.abs(_ij_x_sum_for_j - _jk_x_sum_for_j)
        constraint_3 = np.abs(_ij_x_sum_for_j - _jkp_x_sum_for_j)
        _y_sorting_for_j = x[:,self.binary_sorting_slice].reshape(x.shape[0],-1, len(self._parameters.facility_storage_capacities[0]))
        constraint_4 = _ij_x_sum_for_j - np.sum(_y_sorting_for_j * self._parameters.facility_storage_capacities[0], axis = 2)
        _jk_x_sum_for_k = np.sum(x[:,self.continuous_jk_slice].reshape(x.shape[0],-1,_num_sorting_facilities), axis = 2)
        _jkp_x_sum_for_kp = np.sum(x[:,self.continuous_jkp_slice].reshape(x.shape[0],-1,_num_sorting_facilities), axis = 2)
        _y_incinerator_for_k = x[:,self.binary_incinerator_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[1]))
        _y_landfill_for_kp = x[:,self.binary_landfill_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[2]))
        _y_landfill_for_kp = x[:,self.binary_landfill_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[2]))
        constraint_5 = _jk_x_sum_for_k - np.sum(_y_incinerator_for_k * self._parameters.facility_storage_capacities[1], axis = 2)
        constraint_6 = _jkp_x_sum_for_kp - np.sum(_y_landfill_for_kp * self._parameters.facility_storage_capacities[2], axis = 2)
        constraint_7 = np.column_stack([x[:,self.continuous_ij_slice] - x[:,self.integer_ij_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        # print(constraint_7.shape)
        constraint_8 = np.column_stack([x[:,self.continuous_jk_slice] - x[:,self.integer_jk_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_9 = np.column_stack([x[:,self.continuous_jkp_slice] - x[:,self.integer_jkp_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_10 = np.sum(_y_sorting_for_j, axis=2) - 1
        constraint_11 = np.sum(_y_incinerator_for_k, axis=2) - 1
        constraint_12 = np.sum(_y_landfill_for_kp, axis=2) - 1
        out["F"] = np.column_stack([objective_1, objective_2, objective_3])
        out["G"] = np.column_stack([constraint_1, constraint_2, constraint_3, constraint_4,constraint_5,constraint_6, constraint_7, constraint_8, constraint_9, constraint_10, constraint_11, constraint_12])
    
    def _create_variables(self):
        """
        The function creates a list of the weights of the edges between the collection locations and the
        sorting facilities, the sorting facilities and the incinerator facilities, and the sorting
        facilities and the landfill facilities. It also creates a list of the supplies of the collection
        locations. It then creates slices for the binary variables, the integer variables, and the
        continuous variables.
        :return: The number of binary, integer and continuous variables
        """

        self._ij_list = [(w['weight']) for i, j, w in self._parameters._G.G.edges(data=True) if (i in self._parameters._G.collection_locations and j in self._parameters._sorting_facilities)]
        self._jk_list = [(w['weight']) for j, k, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and k in self._parameters._incinerator_facilities) or (j in self._parameters._incinerator_facilities and k in self._parameters._sorting_facilities)]
        self._jkp_list = [(w['weight']) for j, kp, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and kp in self._parameters._landfill_facilities) or (j in self._parameters._landfill_facilities and kp in self._parameters._sorting_facilities)]
        
        self.supplies = np.array(list(self._parameters._G.supplies.values()))[:,0]

        # Multiply by 3 for 3 sizes per sorting facility
        _sorting_length = len(self._parameters._sorting_facilities)*3
        _incinerator_length = len(self._parameters._incinerator_facilities) *3
        _landfill_length = len(self._parameters._landfill_facilities) * 3
        
        _ij_length = len(self._ij_list)
        _jk_length = len(self._jk_list)
        _jkp_length = len(self._jkp_list)
        num_binary_vars = _sorting_length + _incinerator_length + _landfill_length
        num_integer_vars = _ij_length + _jk_length + _jkp_length
        num_continuous_vars = num_integer_vars

        self.binary_sorting_slice = slice(0,_sorting_length, 1)
        self.binary_incinerator_slice = slice(_sorting_length, _incinerator_length + _sorting_length, +  1)
        self.binary_landfill_slice = slice(_incinerator_length + _sorting_length , num_binary_vars , 1)
        
        self.integer_ij_slice = slice(num_binary_vars, num_binary_vars + _ij_length, 1)
        self.integer_jk_slice = slice(num_binary_vars + _ij_length, (num_integer_vars + num_binary_vars) - _jkp_length , 1)
        self.integer_jkp_slice = slice((num_integer_vars + num_binary_vars) - _jkp_length, (num_integer_vars + num_binary_vars) , 1)

        self.continuous_ij_slice = slice((num_integer_vars + num_binary_vars) , (num_integer_vars + num_binary_vars) + _ij_length, 1)
        self.continuous_jk_slice = slice((num_integer_vars + num_binary_vars) + _ij_length, (num_integer_vars + num_continuous_vars + num_binary_vars) - _jkp_length , 1)
        self.continuous_jkp_slice = slice((num_integer_vars+ num_continuous_vars + num_binary_vars) - _jkp_length, (num_integer_vars+ num_continuous_vars + num_binary_vars), 1)
        
        return num_binary_vars,num_integer_vars,num_continuous_vars

class Elementwise_heuristic(ElementwiseProblem):
    def __init__(self, parameters : Parameters, **kwargs) -> None:
        """
        The function creates the variables for the problem, and then creates the problem with the number
        of variables, objectives, and constraints
        
        :param parameters: The parameters object that contains all the data for the problem
        :type parameters: Parameters
        """

        self._parameters = parameters
        self.num_binary_vars, self.num_integer_vars, self.num_continuous_vars = self._create_variables()
        _n_constr = len(self.supplies) + (len(self._ij_list)//len(self.supplies) ) *2 + len(self._parameters._sorting_facilities)*2 + len(self._parameters._incinerator_facilities)*2 + len(self._parameters._landfill_facilities) * 2 + len(self._ij_list)* len(self._parameters.maximum_amount_transport) + len(self._jk_list)* len(self._parameters.maximum_amount_transport) + len(self._jkp_list)* len(self._parameters.maximum_amount_transport)

        print(f"Number of Variables: {self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars}")
        print(f"Binary: {self.num_binary_vars},  Integer: {self.num_integer_vars},   Continuous: {self.num_continuous_vars}")

        print(f"Number of constraints: {_n_constr}.")
        xu_bin = np.ones(self.num_binary_vars)
        xu_int = np.ones(self.num_integer_vars) * self._parameters._G._number_of_cities
        xu_cont = np.ones(self.num_continuous_vars) * np.sum(self.supplies)
        super().__init__(n_var = self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars,
                                    n_obj = 3,
                                    n_constr = _n_constr,
                                    xl=0,
                                    xu=np.concatenate([xu_bin,xu_int,xu_cont]),
                                    **kwargs)
    def _evaluate(self, x, out, *args, **kwargs):
        """
        The function takes in the decision variables, and returns the objective function values and the
        constraint violation
        
        :param x: the decision variables
        :param out: the output dictionary
        """
        # Objective 1
        _sorting_x = x[self.binary_sorting_slice].reshape(-1,self._parameters._direct_land_usage.shape[1])
        _incinerator_x = x[self.binary_incinerator_slice].reshape(-1,self._parameters._direct_land_usage.shape[1])
        _landfill_x = x[self.binary_landfill_slice].reshape(-1,self._parameters._direct_land_usage.shape[1])

        _sorting_opening_costs = np.sum(_sorting_x @ self._parameters.opening_costs[0])
        _incinerator_opening_costs = np.sum(_incinerator_x @ self._parameters.opening_costs[1])
        _landfill_opening_costs = np.sum(_landfill_x @ self._parameters.opening_costs[2])

        total_opening_cost = _sorting_opening_costs + _incinerator_opening_costs + _landfill_opening_costs
        
        _ij_operational_cost = (self._ij_list + self._parameters.operational_costs[0]) @ x[self.integer_ij_slice]
        _jk_operational_cost = (self._jk_list + self._parameters.operational_costs[1]) @ x[self.integer_jk_slice]
        _jkp_operational_cost = (self._jkp_list + self._parameters.operational_costs[2]) @ x[self.integer_jkp_slice]
        total_operational_cost = _ij_operational_cost + _jk_operational_cost + _jkp_operational_cost
        objective_1 = total_opening_cost + total_operational_cost

        # Objective 2
        _sorting_land_usage = np.sum(_sorting_x @ self._parameters.land_stress_ratios[0])
        _incinerator_land_usage = np.sum(_incinerator_x @ self._parameters.land_stress_ratios[1])
        _landfill_land_usage = np.sum(_landfill_x @ self._parameters.land_stress_ratios[2])
        objective_2 = _sorting_land_usage + _incinerator_land_usage + _landfill_land_usage

        #Objective 3
        _sorting_health_impact = np.sum(_sorting_x * self._parameters.population_list[0] * self._parameters.facility_daly_matrix[0])
        _incinerator_health_impact = np.sum(_incinerator_x * self._parameters.population_list[1] * self._parameters.facility_daly_matrix[1])
        _landfill_health_impact = np.sum(_landfill_x * self._parameters.population_list[2] * self._parameters.facility_daly_matrix[2])
        total_facility_health_impact = _sorting_health_impact + _incinerator_health_impact + _landfill_health_impact
        
        _sorting_link_health_impact = np.sum(self._parameters.link_populations_list[0] * self._parameters.link_dalys_list[0] * x[self.integer_ij_slice])
        _incinerator_link_health_impact = np.sum(self._parameters.link_populations_list[1] * self._parameters.link_dalys_list[1] * x[self.integer_jk_slice])
        _landfill_link_health_impact = np.sum(self._parameters.link_populations_list[2] * self._parameters.link_dalys_list[2] * x[self.integer_jkp_slice])
        total_link_health_impact = _sorting_link_health_impact + _incinerator_link_health_impact + _landfill_link_health_impact
        objective_3 = total_facility_health_impact + total_link_health_impact

        # Constraints
        _num_sorting_facilities = len(self._parameters._sorting_facilities)
        constraint_1 = np.abs(np.sum(x[self.continuous_ij_slice].reshape(-1,_num_sorting_facilities),axis=1) - self.supplies)
        _ij_x_sum_for_j = np.sum(x[self.continuous_ij_slice].reshape(-1,_num_sorting_facilities), axis = 0)
        _jk_x_sum_for_j = np.sum(x[self.continuous_jk_slice].reshape(-1,_num_sorting_facilities), axis = 0)
        _jkp_x_sum_for_j = np.sum(x[self.continuous_jkp_slice].reshape(-1,_num_sorting_facilities), axis = 0)
        constraint_2 = np.abs(_ij_x_sum_for_j - _jk_x_sum_for_j)
        constraint_3 = np.abs(_ij_x_sum_for_j - _jkp_x_sum_for_j)
        _y_sorting_for_j = x[self.binary_sorting_slice].reshape(-1,len(self._parameters.facility_storage_capacities[0]))
        constraint_4 = _ij_x_sum_for_j - np.sum(_y_sorting_for_j * self._parameters.facility_storage_capacities[0], axis = 1)
        _jk_x_sum_for_k = np.sum(x[self.continuous_jk_slice].reshape(-1,_num_sorting_facilities), axis = 1)
        _jkp_x_sum_for_kp = np.sum(x[self.continuous_jkp_slice].reshape(-1,_num_sorting_facilities), axis = 1)
        _y_incinerator_for_k = x[self.binary_incinerator_slice].reshape(-1,len(self._parameters.facility_storage_capacities[1]))
        _y_landfill_for_kp = x[self.binary_landfill_slice].reshape(-1,len(self._parameters.facility_storage_capacities[2]))
        constraint_5 = _jk_x_sum_for_k - np.sum(_y_incinerator_for_k * self._parameters.facility_storage_capacities[1], axis = 1)
        constraint_6 = _jkp_x_sum_for_kp - np.sum(_y_landfill_for_kp * self._parameters.facility_storage_capacities[2], axis = 1)
        constraint_7 = np.concatenate([x[self.continuous_ij_slice] - x[self.integer_ij_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_8 = np.concatenate([x[self.continuous_jk_slice] - x[self.integer_jk_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_9 = np.concatenate([x[self.continuous_jkp_slice] - x[self.integer_jkp_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_10 = np.sum(_y_sorting_for_j, axis=1) - 1
        constraint_11 = np.sum(_y_incinerator_for_k, axis=1) - 1
        constraint_12 = np.sum(_y_landfill_for_kp, axis=1) - 1
        out["F"] = [objective_1, objective_2,objective_3]
        out["G"] = np.concatenate([constraint_1,constraint_2,constraint_3,constraint_4,constraint_5,constraint_6, constraint_7,constraint_8,constraint_9, constraint_10, constraint_11, constraint_12])

    def _create_variables(self):
        """
        The function creates a list of the weights of the edges between the collection locations and the
        sorting facilities, the sorting facilities and the incinerator facilities, and the sorting
        facilities and the landfill facilities. It also creates a list of the supplies of the collection
        locations. It then creates slices for the binary variables, the integer variables, and the
        continuous variables.
        :return: The number of binary, integer and continuous variables
        """
        self._ij_list = [(w['weight']) for i, j, w in self._parameters._G.G.edges(data=True) if (i in self._parameters._G.collection_locations and j in self._parameters._sorting_facilities)]
        self._jk_list = [(w['weight']) for j, k, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and k in self._parameters._incinerator_facilities) or (j in self._parameters._incinerator_facilities and k in self._parameters._sorting_facilities)]
        self._jkp_list = [(w['weight']) for j, kp, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and kp in self._parameters._landfill_facilities) or (j in self._parameters._landfill_facilities and kp in self._parameters._sorting_facilities)]
        
        self.supplies = np.array(list(self._parameters._G.supplies.values()))[:,0]

        # Multiply by 3 for 3 sizes per sorting facility
        _sorting_length = len(self._parameters._sorting_facilities)*3
        _incinerator_length = len(self._parameters._incinerator_facilities) *3
        _landfill_length = len(self._parameters._landfill_facilities) * 3
        
        _ij_length = len(self._ij_list)
        _jk_length = len(self._jk_list)
        _jkp_length = len(self._jkp_list)
        num_binary_vars = _sorting_length + _incinerator_length + _landfill_length
        num_integer_vars = _ij_length + _jk_length + _jkp_length
        num_continuous_vars = num_integer_vars

        self.binary_sorting_slice = slice(0,_sorting_length, 1)
        self.binary_incinerator_slice = slice(_sorting_length, _incinerator_length + _sorting_length, +  1)
        self.binary_landfill_slice = slice(_incinerator_length + _sorting_length , num_binary_vars , 1)
        
        self.integer_ij_slice = slice(num_binary_vars, num_binary_vars + _ij_length, 1)
        self.integer_jk_slice = slice(num_binary_vars + _ij_length, (num_integer_vars + num_binary_vars) - _jkp_length , 1)
        self.integer_jkp_slice = slice((num_integer_vars + num_binary_vars) - _jkp_length, (num_integer_vars + num_binary_vars) , 1)

        self.continuous_ij_slice = slice((num_integer_vars + num_binary_vars) , (num_integer_vars + num_binary_vars) + _ij_length, 1)
        self.continuous_jk_slice = slice((num_integer_vars + num_binary_vars) + _ij_length, (num_integer_vars + num_continuous_vars + num_binary_vars) - _jkp_length , 1)
        self.continuous_jkp_slice = slice((num_integer_vars+ num_continuous_vars + num_binary_vars) - _jkp_length, (num_integer_vars+ num_continuous_vars + num_binary_vars), 1)
        
        return num_binary_vars,num_integer_vars,num_continuous_vars
    
    


class Minimize():
    """
    The class `Minimize` is a wrapper for the `NSGA2` and `NSGA3` algorithms from the `pymoo` library.
    It takes in a `Multiobjective_heuristic` object, a population size, a number of generations, and a
    boolean for verbosity. It returns a `Result` object from `pymoo` that contains the Pareto front and
    the corresponding decision variables. 
    """
    def __init__(self, 
                problem : Elementwise_heuristic, 
                population_size : int, 
                termination : pymoo.util.termination, 
                verbose = True, 
                nsga3 = True):
        """
        The function `__init__` is a constructor for the class `Minimize`. It takes in the problem,
        population size, number of generations, verbose, and nsga3 as arguments. It then sets the
        problem, population size, number of generations, verbose, and nsga3 as attributes of the class.
        It also creates the mixed variables sampling, crossover, and mutation.
        
        :param problem: The problem to be solved
        :type problem: Multiobjective_heuristic
        :param population_size: The number of individuals in the population
        :type population_size: int
        :param number_of_generations: The number of generations to run the algorithm for
        :type number_of_generations: int
        :param verbose: If True, prints the progress of the algorithm, defaults to True (optional)
        :param nsga3: If True, the algorithm will use NSGA-III. If False, it will use NSGA-II, defaults
        to True (optional)
        """
        self._problem = problem
        self._pop_size = population_size
        self._termination = termination
        self._verbose = verbose
        self._nsga3 = nsga3
        self.sampling, self.crossover, self.mutation = self._create_mixed_variables()

    def minimize_heuristic(self):
        """
        The `minimize_heuristic` method is the main method of the class. It takes in the
        `Multiobjective_heuristic` object and returns a `Result` object from `pymoo`. 
        """
        if self._nsga3:
            ref_dirs = get_reference_directions("das-dennis", n_dim = 3, n_partitions=12)
            algorithm = NSGA3(pop_size=self._pop_size,
                    sampling=self.sampling,
                    crossover=self.crossover,
                    mutation=self.mutation,
                    ref_dirs = ref_dirs,
                    eliminate_duplicates=True)
        else: 
            algorithm = NSGA2(pop_size=self._pop_size,
                    sampling=self.sampling,
                    crossover=self.crossover,
                    mutation=self.mutation,
                    eliminate_duplicates=True)
        self._problem = ConstraintsAsPenalty(self._problem, penalty=1e6)
        res = minimize(self._problem,
            algorithm,
            self._termination,
            seed=1,
            verbose=self._verbose,
            save_history= False)

        return res

    def _create_mixed_variables(self):
        """
         We create a mixed variable sampling, crossover, and mutation function that uses the
        `bin_random`, `int_random`, and `real_random` sampling functions, the `bin_hux`, `int_sbx`, and
        `real_sbx` crossover functions, and the `bin_bitflip`, `int_pm`, and `real_pm` mutation
        functions
        :return: The sampling, crossover, and mutation methods for the mixed variables.
        """
        _mask_binary = np.array(["bin" for _ in range(self._problem.num_binary_vars)])
        _mask_integer = np.array(["int" for _ in range(self._problem.num_integer_vars)])
        _mask_continuous = np.array(["real" for _ in range(self._problem.num_continuous_vars)])
        _masks = np.hstack((_mask_binary, _mask_integer, _mask_continuous))
        _sampling = MixedVariableSampling(_masks, {
            "bin" : get_sampling("bin_random"),
            "int": get_sampling("int_random"),
            "real": get_sampling("real_random")
        })

        _crossover = MixedVariableCrossover(_masks, {
            "bin": get_crossover("bin_hux", prob=0.4),
            "int": get_crossover("int_sbx", prob=0.4),
            "real": get_crossover("real_sbx", prob=0.4)
        })

        _mutation = MixedVariableMutation(_masks, {
            "bin": get_mutation("bin_bitflip"),
            "int": get_mutation("int_pm"),
            "real": get_mutation("real_pm")
        })
        return _sampling, _crossover, _mutation
if __name__ == "__main__":
    from generate_graph import Graph
    num_of_collection_centers = 5
    set_seed = 1
    verbose = True 
    nsga3 = False
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=True, seed=set_seed, baseline_scaler=3)
    parameters = Parameters(RandomGraph, set_seed)
    three_objective_problem = Vectorized_heuristic(parameters)
    minimization = Minimize(problem = three_objective_problem, population_size = 1000, number_of_generations = 400, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print(result.F, result.X)
    three_objective_problem = Elementwise_heuristic(parameters)
    minimization = Minimize(problem = three_objective_problem, population_size = 1000, number_of_generations = 400, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print(result.F, result.X)