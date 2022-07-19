from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
import numpy as np
from parameters import Parameters
import networkx as nx
import pymoo
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems.constr_as_penalty import ConstraintsAsPenalty
from pymoo.core.repair import Repair
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


class Vectorized_heuristic(Problem):
    def __init__(self, parameters : Parameters, verbose = True, **kwargs) -> None:
        """
        The function creates the variables for the problem, and then creates the problem with the number
        of variables, objectives, and constraints
        
        :param parameters: The parameters object that contains all the data for the problem
        :type parameters: Parameters
        """

        self._parameters = parameters
        self.num_binary_vars, self.num_integer_vars, self.num_continuous_vars = self._create_variables()
        _n_obj = 3
        _n_constr = len(self.supplies) + (len(self._ij_list)//len(self.supplies) ) *2+ len(self._ij_list)* len(self._parameters.maximum_amount_transport) + len(self._jk_list)* len(self._parameters.maximum_amount_transport) + len(self._jkp_list)* len(self._parameters.maximum_amount_transport)
        _n_constr += (len(self._parameters._sorting_facilities) + len(self._parameters._incinerator_facilities) +len(self._parameters._landfill_facilities))*2
        if verbose:
            print(f"Number of Variables: {self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars}")
            print(f"Binary: {self.num_binary_vars},  Integer: {self.num_integer_vars},   Continuous: {self.num_continuous_vars}")
            print(f"Number of constraints: {_n_constr}.")

        xu_bin = np.ones(self.num_binary_vars)
        xu_int = np.ones(self.num_integer_vars) * np.ceil(np.sum(self.supplies)/self._parameters.maximum_amount_transport[0])
        xu_cont = np.ones(self.num_continuous_vars) * np.sum(self.supplies)

        super().__init__(n_var = self.num_binary_vars + self.num_integer_vars + self.num_continuous_vars,
                                    n_obj = _n_obj,
                                    n_constr = _n_constr,
                                    xl=0,
                                    xu=np.concatenate([xu_bin,xu_int,xu_cont]),
                                    **kwargs)
    def _evaluate(self, x, out, t = False, *args, **kwargs):
        """
        The function takes in the decision variables, and returns the objective function values and the
        constraint violation
        :param x: the decision variables
        :param out: the output dictionary
        """
        # Objective 1
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
        _epsilon = 0
        _num_sorting_facilities = len(self._parameters._sorting_facilities)
        _num_collection_facilities = len(self._parameters._G.collection_locations)
        _num_incinerator_facilities = len(self._parameters._incinerator_facilities)
        _num_landfill_facilities = len(self._parameters._landfill_facilities)
        constraint_1 = np.abs(np.sum(x[:,self.continuous_ij_slice].reshape(-1,_num_collection_facilities,_num_sorting_facilities),axis=2) - self.supplies) - _epsilon
        _ij_f_sum_for_j = np.sum(x[:,self.continuous_ij_slice].reshape(-1,_num_collection_facilities, _num_sorting_facilities), axis = 1)
        _jk_kp_f_sum_for_j = np.sum(x[:,self.continuous_end_link_slice].reshape(-1,_num_sorting_facilities,_num_incinerator_facilities + _num_landfill_facilities), axis = 2)
        constraint_2 = np.abs(_ij_f_sum_for_j - _jk_kp_f_sum_for_j)  - _epsilon
        _y_sorting_for_j = x[:,self.binary_sorting_slice].reshape(x.shape[0],-1, len(self._parameters.facility_storage_capacities[0]))
        constraint_4 = _ij_f_sum_for_j - np.sum(_y_sorting_for_j * self._parameters.facility_storage_capacities[0], axis = 2)
        _jk_x_sum_for_k = np.sum(x[:,self.continuous_jk_slice].reshape(x.shape[0],_num_sorting_facilities,-1), axis = 1)
        _jkp_x_sum_for_kp = np.sum(x[:,self.continuous_jkp_slice].reshape(x.shape[0],_num_sorting_facilities,-1), axis = 1)
        _y_incinerator_for_k = x[:,self.binary_incinerator_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[1]))
        _y_landfill_for_kp = x[:,self.binary_landfill_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[2]))
        _y_landfill_for_kp = x[:,self.binary_landfill_slice].reshape(x.shape[0],-1,len(self._parameters.facility_storage_capacities[2]))
        constraint_5 = _jk_x_sum_for_k - np.sum(_y_incinerator_for_k * self._parameters.facility_storage_capacities[1], axis = 2)
        constraint_6 = _jkp_x_sum_for_kp - np.sum(_y_landfill_for_kp * self._parameters.facility_storage_capacities[2], axis = 2)
        constraint_7 = np.column_stack([x[:,self.continuous_ij_slice] - x[:,self.integer_ij_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_8 = np.column_stack([x[:,self.continuous_jk_slice] - x[:,self.integer_jk_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_9 = np.column_stack([x[:,self.continuous_jkp_slice] - x[:,self.integer_jkp_slice] * self._parameters.maximum_amount_transport[l] for l in range(len(self._parameters.maximum_amount_transport))])
        constraint_10 = np.sum(_y_sorting_for_j, axis=2) - 1
        constraint_11 = np.sum(_y_incinerator_for_k, axis=2) - 1
        constraint_12 = np.sum(_y_landfill_for_kp, axis=2) - 1
        
        out["F"] = np.column_stack([objective_1, objective_2, objective_3])
        out["G"] = np.column_stack([constraint_1, constraint_2, constraint_4,constraint_5,constraint_6, constraint_7, constraint_8, constraint_9, constraint_10, constraint_11, constraint_12])
        if t == True:
            print(out["G"])
            
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
        self._jk_list = [(w['weight']) for j, k, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and k in self._parameters._incinerator_facilities)]
        self._jkp_list = [(w['weight']) for j, kp, w in self._parameters._G.G.edges(data=True) if (j in self._parameters._sorting_facilities and kp in self._parameters._landfill_facilities)]
        
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
        self.binary_end_facility_slice = slice(_sorting_length, num_binary_vars, 1)
        
        self.integer_ij_slice = slice(num_binary_vars, num_binary_vars + _ij_length, 1)
        self.integer_jk_slice = slice(num_binary_vars + _ij_length, (num_integer_vars + num_binary_vars) - _jkp_length , 1)
        self.integer_jkp_slice = slice((num_integer_vars + num_binary_vars) - _jkp_length, num_integer_vars + num_binary_vars , 1)
        self.integer_end_link_slice = slice(num_binary_vars + _ij_length, num_integer_vars + num_binary_vars)

        self.continuous_ij_slice = slice((num_integer_vars + num_binary_vars) , (num_integer_vars + num_binary_vars) + _ij_length, 1)
        self.continuous_jk_slice = slice((num_integer_vars + num_binary_vars) + _ij_length, (num_integer_vars + num_continuous_vars + num_binary_vars) - _jkp_length , 1)
        self.continuous_jkp_slice = slice((num_integer_vars+ num_continuous_vars + num_binary_vars) - _jkp_length, (num_integer_vars+ num_continuous_vars + num_binary_vars), 1)
        self.continuous_end_link_slice = slice((num_integer_vars + num_binary_vars) + _ij_length, (num_integer_vars+ num_continuous_vars + num_binary_vars), 1)
        
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
        _sorting_length = len(self._parameters._sorting_facilities) * 3
        _incinerator_length = len(self._parameters._incinerator_facilities) * 3
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

def update_by_shrinking(main_to_update, main_update_from, main_mask, shrink_with, shrink_mask, shrink_factor : int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the main array with the update array, but shrink the update array first.
    
    :param main_to_update: the array that will be updated
    :param main_update_from: the array that we want to update the main array with
    :param main_mask: the mask for the main array
    :param shrink_with: the array to shrink
    :param shrink_mask: the mask of the shrink_with array that we want to shrink
    :param shrink_factor: the amount to shrink the numbers by, defaults to 10
    :type shrink_factor: int (optional)
    :return: The updated main_to_update and main_update_from
    """
    def _shrink_numbers(to_update, update_with, update_mask, shrink_factor):
        to_update[update_mask] /= np.sum(update_with[update_mask], axis=1)[:,np.newaxis] + shrink_factor
        return to_update
    def _update_array(to_update, update_with, mask):
        if to_update.ndim == 4:
            to_update[0][mask] = update_with[0][mask]
            to_update[1][mask] = update_with[1][mask]
        elif to_update.ndim == 3:
            to_update[mask] = update_with[mask]
        return to_update
    main_update_from = _shrink_numbers(main_update_from, shrink_with, shrink_mask, shrink_factor)
    main_to_update = _update_array(main_to_update, main_update_from, main_mask)
    return main_to_update, main_update_from

class CustomMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)
        
        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None
    def _normalize(self, X, num_facilities, link_slice, normalize_supplies):
            normalize = lambda x_reshaped, in_supplies: x_reshaped/in_supplies
            # Reshape for j
            _link_f_for_j = X[:, link_slice].reshape(X.shape[0], -1, num_facilities)
            _n_link_f_for_j = normalize(_link_f_for_j, normalize_supplies)
            return _n_link_f_for_j

    def _fix_mutation(self, X : np.ndarray, num_facilities : int, _link_slice : slice, normalize_supplies : np.ndarray, mutation_mask : np.ndarray):
        """
        If the sum of the original values is greater than 1, then shrink the original values until the
        sum is less than or equal to 1. If the sum of the original values is less than 1, then shrink
        the mutation values until the sum is greater than or equal to 1. Lastly, add the difference to 1 to mutated values
        s.t. the final sum is 1.
        
        :param X: the matrix of flows
        :type X: np.ndarray
        :param num_facilities: the number of facilities in the problem
        :type num_facilities: int
        :param _link_slice: the slice of the X matrix that contains the link values
        :type _link_slice: slice
        :param normalize_supplies: the supplies for each facility
        :type normalize_supplies: np.ndarray
        :param mutation_mask: a boolean array of shape (n_samples, n_features)
        :type mutation_mask: np.ndarray
        :return: the X matrix with the mutated values.
        """
        
        edited_supplies = normalize_supplies.copy()
        if len(edited_supplies[edited_supplies == 0]) > 0: edited_supplies[edited_supplies == 0] += (np.sum(edited_supplies)/edited_supplies.size) * 10
        _n_link_f_for_j = self._normalize(X, num_facilities, _link_slice, edited_supplies)
        _mutation_link_for_j = mutation_mask[:, _link_slice].reshape(mutation_mask.shape[0], -1, num_facilities)
        
        only_mutations = _n_link_f_for_j.copy()
        only_mutations[~_mutation_link_for_j] = 0
        only_originals = _n_link_f_for_j.copy()
        only_originals[_mutation_link_for_j] = 0
        _mutation_link_for_j[_n_link_f_for_j == 0] = False

        org_greater_1 = np.sum(only_originals, axis = 2) > 1
        while len(only_originals[org_greater_1]) > 0:
            _n_link_f_for_j, only_originals = update_by_shrinking(_n_link_f_for_j, only_originals, ~_mutation_link_for_j, only_originals, org_greater_1)
            org_greater_1 = np.sum(only_originals, axis = 2) > 1
        
        all_greater_1 = np.sum(_n_link_f_for_j, axis = 2) > 1
        while len(only_mutations[all_greater_1]) > 0:
            _n_link_f_for_j, only_mutations = update_by_shrinking(_n_link_f_for_j, only_mutations, _mutation_link_for_j, only_mutations, all_greater_1)
            all_greater_1 = np.sum(_n_link_f_for_j, axis = 2) > 1
        
        non_zero_count = np.count_nonzero(_mutation_link_for_j, axis=2)
        non_zero_count[non_zero_count == 0] += 1
        only_mutations += ((1-np.sum(_n_link_f_for_j, axis=2))/non_zero_count)[:,:,np.newaxis]
        _n_link_f_for_j[_mutation_link_for_j] = only_mutations[_mutation_link_for_j]
        denormalize = lambda x_reshaped, in_supplies: x_reshaped * in_supplies
        _link_f_for_j = denormalize(_n_link_f_for_j, normalize_supplies)
        X[:, _link_slice] = _link_f_for_j.reshape(X[:, _link_slice].shape)
        return X

    def _do(self, problem, X, **kwargs):
        """
        The function takes in a matrix of decision variables, and for each row, it randomly selects a
        decision variable to mutate. The mutation is done by adding a random number to the decision
        variable. The random number is generated by a formula that is based on the decision variable's
        upper and lower bounds
        
        :param problem: the problem instance
        :param X: the population
        :return: The mutated values of the input array X.
        """
        num_sorting = len(problem._parameters._sorting_facilities) 
        num_incinerators = len(problem._parameters._incinerator_facilities) 
        num_landfill = len(problem._parameters._landfill_facilities)
        _ij_slice = slice(0,len(problem._ij_list))
        _jk_kp_slice = slice(len(problem._ij_list), X.shape[1])

        X = X.astype(float)
        Y = np.full(X.shape, np.inf)
        
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = np.random.random(X.shape) < self.prob

        Y[:, :] = X
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]
        # set the values for output
        Y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        Y = set_to_bounds_if_outside_by_problem(problem, Y)

        Y = self._fix_mutation(Y, num_sorting, _ij_slice, problem.supplies[:, np.newaxis], do_mutation)
        _supplies_j = np.sum(Y[:,_ij_slice].reshape(Y.shape[0], -1, num_sorting), axis=1)[:,:,np.newaxis]
        Y = self._fix_mutation(Y, num_incinerators + num_landfill, _jk_kp_slice, _supplies_j, do_mutation)
        
        return Y



class CustomCrossover(Crossover):
    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points
    def _fix_crossover(self, X : np.ndarray, normalize_supplies : np.ndarray, link_slice : slice, mask : np.ndarray, num_facilities : int) -> np.ndarray:
        """
        The function takes the population, and for each individual, it fixes the transport variables
        that are not part of the original individual, so that they sum to 1.
        
        :param X: the current population
        :type X: np.ndarray
        :param normalize_supplies: The supplies of the problem normalized to sum to 1
        :type normalize_supplies: np.ndarray
        :param link_slice: the slice of the 3d array that contains the transport variables
        :type link_slice: slice
        :param inv_mask: a boolean array of shape (2, num_customers, num_facilities)
        :type inv_mask: np.ndarray
        :param num_facilities: The number of facilities in the problem
        :type num_facilities: int
        :return: The fixed population according to the hard constraints of the problem.
        """
        
        def apply_mask(X_3d : np.ndarray, M_3d : np.ndarray) -> np.ndarray:
            _X_3d = X_3d.copy()
            _X_3d[0][M_3d] = 0
            _X_3d[1][M_3d] = 0
            return _X_3d


        normalize = lambda x_reshaped, in_supplies: x_reshaped/in_supplies
        denormalize = lambda x_reshaped, in_supplies: x_reshaped * in_supplies
        # Reshape for j
        _link_f_for_j = X[:,:,link_slice].reshape(X.shape[0], X.shape[1], -1, num_facilities)
        edited_supplies = normalize_supplies.copy()
        if len(edited_supplies[edited_supplies == 0]) > 0: edited_supplies[edited_supplies == 0] += (np.sum(edited_supplies)/edited_supplies.size) * 10
        _n_link_f_for_j = normalize(_link_f_for_j, edited_supplies)
        _M_ij_3d = mask.reshape(mask.shape[0], -1, num_facilities)
        # Apply the masks for the originals and crossover candidates
        only_originals = apply_mask(_n_link_f_for_j, _M_ij_3d)
        only_crossovers = apply_mask(_n_link_f_for_j, ~_M_ij_3d)

        # Fix originals that sum above 1 and fix crossover candiates that sum above 1
        large_originals_mask = np.sum(only_originals, axis = 3) >= 1
        while len(only_originals[large_originals_mask]) > 0:
            _n_link_f_for_j, only_originals = update_by_shrinking(main_to_update = _n_link_f_for_j, 
                                                        main_update_from = only_originals, 
                                                        main_mask = ~_M_ij_3d, 
                                                        shrink_with = only_originals, 
                                                        shrink_mask = large_originals_mask)
            large_originals_mask = np.sum(only_originals, axis = 3) >= 1

        large_rows_mask = np.sum(_n_link_f_for_j, axis = 3) > 1
        while len(only_originals[large_rows_mask]) > 0:
            _n_link_f_for_j, only_crossovers = update_by_shrinking(main_to_update =_n_link_f_for_j, 
                                                         main_update_from = only_crossovers, 
                                                         main_mask = _M_ij_3d, 
                                                         shrink_with = only_originals, 
                                                         shrink_mask = large_rows_mask)
            large_rows_mask = np.sum(_n_link_f_for_j, axis = 3) > 1
        

        # Add difference to 1 to the crossover candidates, s.t. sum is 1.
        non_zero_count = np.count_nonzero(_M_ij_3d, axis=2)
        only_crossovers += ((1-np.sum(_n_link_f_for_j, axis=3))/non_zero_count)[:,:,:,np.newaxis]
        _n_link_f_for_j[0][_M_ij_3d] = only_crossovers[0][_M_ij_3d]
        _n_link_f_for_j[1][_M_ij_3d] = only_crossovers[1][_M_ij_3d]
        _link_f_for_j = denormalize(_n_link_f_for_j, normalize_supplies)
        X[:,:,link_slice] = _link_f_for_j.reshape(X[:,:,link_slice].shape)
        return X

    def _do(self, problem, X, **kwargs):
        """
        The function creates a mask for each of the three types of facilities (sorting, incineration,
        and landfill) and then uses that mask to create a new population of children
        
        :param problem: the problem instance
        :param X: the input array of parents
        :return: The crossover mask is being returned.
        """

        def crossover_mask(X, M):
            # convert input to output by flatting along the first axis
            _X = np.copy(X)
            _X[0][M] = X[1][M]
            _X[1][M] = X[0][M]
            return _X
        
        def create_ranges(out_facilities, n_matings, n_var):
            """
            Given the number of mating pairs, the number of variables, and the number of output
            facilities, create a matrix of mating pairs by variables, where each mating pair has a
            random number of contiguous True values, and the rest are False.
            
            :param out_facilities: the number of facilities in the output problem
            :param n_matings: number of matings
            :param n_var: the number of variables in the problem
            :return: A boolean array of size (n_matings, n_var)
            """
            
            in_facilities = n_var//out_facilities
            new_matings = n_matings * in_facilities
            r = np.row_stack([np.random.permutation(out_facilities - 1) + 1  for _ in range(new_matings)])[:, :self.n_points]
            r.sort(axis=1)
            r = np.column_stack([r, np.full(new_matings, out_facilities)])
            M = np.full((new_matings, out_facilities), False)
            # create for each individual the crossover range
            for i in range(new_matings):
                j = 0
                while j < r.shape[1] - 1:
                    a, b = r[i, j], r[i, j + 1]
                    M[i, a:b] = True
                    j += 2
            M = M.reshape(n_matings, -1)
            return M

         # get the X of parents and count the matings
        _, n_matings, n_var = X.shape
        num_sorting = len(problem._parameters._sorting_facilities) 
        num_incinerators = len(problem._parameters._incinerator_facilities) 
        num_landfill = len(problem._parameters._landfill_facilities)
        _ij_slice = slice(0,len(problem._ij_list))
        _jk_kp_slice = slice(len(problem._ij_list), n_var)
        M_ij = create_ranges(num_sorting,n_matings, X[:,:,_ij_slice].shape[2])
        M_jk_kp = create_ranges(num_incinerators + num_landfill,n_matings, X[:,:,_jk_kp_slice].shape[2])
        M = np.concatenate([M_ij, M_jk_kp], axis = 1)
        _X = crossover_mask(X, M)
        _X = self._fix_crossover(_X, problem.supplies[:, np.newaxis], _ij_slice, M_ij, num_sorting)
        # Sum over sorting centers
        supplies_j = np.sum(_X[:,:,_ij_slice].reshape(_X.shape[0], _X.shape[1], -1, num_sorting), axis=2)[:,:,:,np.newaxis]
        _X = self._fix_crossover(_X, supplies_j, _jk_kp_slice, M_jk_kp, num_incinerators + num_landfill)
        return _X

class RepairGraph(Repair):
    def _add_zeroes(self, 
                    Z : np.ndarray, 
                    binary_slice : slice,
                    integer_slice : slice,
                    continuous_slice : slice,
                    num_facilities : int,
                    supplies : np.ndarray) -> np.ndarray:
        """
        If a binary variable is zero, then the corresponding integer and continuous variables are
        zeroed out. If the sum of the continuous variables is less than one, then the continuous
        variables are normalized to sum to one.
        
        :param Z: the population of crossover and mutated solutions.
        :type Z: np.ndarray
        :param binary_slice: the slice of the decision vector that corresponds to the binary variables
        :type binary_slice: slice
        :param integer_slice: the slice of the decision vector that contains the integer variables
        :type integer_slice: slice
        :param continuous_slice: the slice of the decision vector that contains the continuous variables
        :type continuous_slice: slice
        :param num_facilities: number of facilities
        :type num_facilities: int
        :param supplies: the supplies for each facility
        :type supplies: np.ndarray
        :return: The return value is the new population, with added zeroes.
        """
        normalize = lambda x_4d, in_supplies: x_4d/in_supplies
        denormalize = lambda x_4d, in_supplies: x_4d * in_supplies
        _binary_for_j = Z[:, binary_slice].reshape(Z.shape[0], -1, 3)
        _integer_for_j = Z[:, integer_slice].reshape(Z.shape[0], -1, num_facilities)
        _continuous_for_j = Z[:, continuous_slice].reshape(Z.shape[0], -1, num_facilities)
        edited_supplies = supplies.copy()
        if len(edited_supplies[edited_supplies == 0]) > 0: edited_supplies[edited_supplies == 0] += (np.sum(edited_supplies)/edited_supplies.size) * 10
        _norm_continuous_for_j = normalize(_continuous_for_j, edited_supplies)
        _indices_binary_zero = np.where(np.sum(_binary_for_j, axis = 2) == 0)
        _integer_for_j[_indices_binary_zero[0],:,_indices_binary_zero[1]] = 0
        _norm_continuous_for_j[_indices_binary_zero[0],:,_indices_binary_zero[1]] = 0
        _integer_for_j[ _norm_continuous_for_j == 0] = 0
        _norm_continuous_for_j[_integer_for_j == 0] = 0

        _non_zero_mask = _norm_continuous_for_j != 0
        only_originals = _norm_continuous_for_j.copy()
        only_originals[~_non_zero_mask] = 0
        only_originals_ix = np.sum(only_originals, axis = 2) > 1
        while len(only_originals[only_originals_ix]) > 0:
            _norm_continuous_for_j, only_originals = update_by_shrinking(_norm_continuous_for_j, only_originals, _non_zero_mask,only_originals,only_originals_ix)
            only_originals_ix = np.sum(only_originals, axis = 2) > 1
        
        _mask_less_than_1 = np.sum(_norm_continuous_for_j, axis = 2) < 1
        f_copy = _norm_continuous_for_j.copy()
        non_zero_count = np.count_nonzero(_non_zero_mask[_mask_less_than_1], axis = 1)
        non_zero_count[non_zero_count == 0] += 1
        f_copy[_mask_less_than_1] += ((1-np.sum(f_copy[_mask_less_than_1], axis=1))/non_zero_count)[:,np.newaxis]
        _norm_continuous_for_j[_non_zero_mask] = f_copy[_non_zero_mask]
        _continuous_for_j = denormalize(_norm_continuous_for_j, supplies)
        Z[:, continuous_slice] = _continuous_for_j.reshape(Z[:, continuous_slice].shape)
        Z[:, integer_slice] = _integer_for_j.reshape(Z[:, integer_slice].shape)
        return Z

    def _do(self, problem, pop : np.ndarray, **kwargs):
        """
        It takes the population, and adds zeroes to the columns of the population matrix that correspond
        to the sorting facilities, and then adds zeroes to the columns of the population matrix that
        correspond to the incinerator/landfill facilities.

        :param problem: the problem instance
        :param pop: the population of solutions
        :type pop: np.ndarray
        :return: The population with the added zeroes.
        """
        Z = pop.get("X")
        num_sorting = len(problem._parameters._sorting_facilities)
        num_incinerator = len(problem._parameters._incinerator_facilities)
        num_landfill = len(problem._parameters._landfill_facilities)
        Z = self._add_zeroes(Z, problem.binary_sorting_slice, problem.integer_ij_slice, problem.continuous_ij_slice, num_sorting, problem.supplies[:, np.newaxis])
        _supplies_j = np.sum(Z[:, problem.continuous_ij_slice].reshape(Z.shape[0], -1, num_sorting), axis = 1)[:,:,np.newaxis]
        Z = self._add_zeroes(Z, problem.binary_end_facility_slice, problem.integer_end_link_slice, problem.continuous_end_link_slice, num_incinerator+num_landfill, _supplies_j)
        return pop.set("X", Z)

        

class Minimize():
    """
    The class `Minimize` is a wrapper for the `NSGA2` and `NSGA3` algorithms from the `pymoo` library.
    It takes in a `Multiobjective_heuristic` object, a population size, a number of generations, and a
    boolean for verbosity. It returns a `Result` object from `pymoo` that contains the Pareto front and
    the corresponding decision variables. 
    """
    def __init__(self, 
                problem : Elementwise_heuristic, 
                termination : pymoo.util.termination,
                population_size : int = 100, 
                reference_directions : pymoo.util.reference_direction = [],
                verbose = True, 
                algorithm : str = 'nsga3',  #nsga2, nsga3, unsga3, rnsga3, moead, ctaea 
                seed = 1):
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
        self._ref_dir = reference_directions
        self._termination = termination
        self._verbose = verbose
        self._algorithm = algorithm
        self._seed = seed
        self.sampling, self.crossover, self.mutation = self._create_mixed_variables()
    def select_algorithm(self):
        if len(self._ref_dir) == 0: self._ref_dir = get_reference_directions("das-dennis", n_dim = 3, n_partitions=15)
        if self._algorithm == 'nsga3':
            print(f"Number of reference directions: {len(self._ref_dir)}")
            algorithm = NSGA3(pop_size = self._pop_size,
                    sampling = self.sampling,
                    crossover = self.crossover,
                    mutation = self.mutation,
                    ref_dirs = self._ref_dir,
                    repair = RepairGraph(),
                    eliminate_duplicates = True)
        elif self._algorithm == 'nsga2': 
            algorithm = NSGA2(pop_size = self._pop_size,
                    sampling = self.sampling,
                    crossover = self.crossover,
                    mutation = self.mutation,
                    repair = RepairGraph(),
                    eliminate_duplicates = True)
        elif self._algorithm == 'moead':
            print(f"Number of reference directions: {len(self._ref_dir)}")
            algorithm = MOEAD(
                    ref_dirs = self._ref_dir,
                    n_neighbors=15,
                    prob_neighbor_mating=0.7,
                    sampling = self.sampling,
                    crossover = self.crossover,
                    mutation = self.mutation,
                    repair = RepairGraph())
        elif self._algorithm == "ctae":
            print(f"Number of reference directions: {len(self._ref_dir)}")
            algorithm = CTAEA(ref_dirs=self._ref_dir,
                            sampling = self.sampling,
                            crossover = self.crossover,
                            mutation = self.mutation,
                            repair = RepairGraph(),
                            eliminate_duplicates = True)
        elif self._algorithm == "agemoea":
            algorithm = AGEMOEA(pop_size = self._pop_size,
                            sampling = self.sampling,
                            crossover = self.crossover,
                            mutation = self.mutation,
                            repair = RepairGraph(),
                            eliminate_duplicates = True)
        elif self._algorithm == "unsga3":
            print(f"Number of reference directions: {len(self._ref_dir)}")
            algorithm = UNSGA3(ref_dirs=self._ref_dir,
                            pop_size = self._pop_size,
                            sampling = self.sampling,
                            crossover = self.crossover,
                            mutation = self.mutation,
                            repair = RepairGraph(),
                            eliminate_duplicates = True)
        else:
            raise ValueError(f"{self._algorithm} is an invalid algorithm. Use one of nsga3, nsga2, unsga3, agemoea, moead, or cteae.")
        return algorithm
    def minimize_heuristic(self):
        """
        The `minimize_heuristic` method is the main method of the class. It takes in the
        `Multiobjective_heuristic` object and returns a `Result` object from `pymoo`. 
        """
        algorithm = self.select_algorithm()
        print(f"Running {self._algorithm.upper()} heuristic...")
        self._problem = ConstraintsAsPenalty(self._problem, penalty=1e10)
        res = minimize(self._problem,
            algorithm,
            self._termination,
            return_least_infeasible=True,
            seed=self._seed,
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
        _num_facilities = len(self._problem._parameters._sorting_facilities) + len(self._problem._parameters._incinerator_facilities) + len(self._problem._parameters._landfill_facilities)
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
            "bin": get_crossover("bin_k_point", n_points = _num_facilities),
            "int": get_crossover("int_k_point", n_points = 1, prob=0.4),
            "real": CustomCrossover(n_points = 1, prob = 0.4)
        })

        _mutation = MixedVariableMutation(_masks, {
            "bin": get_mutation("bin_bitflip"),
            "int": get_mutation("int_pm"),
            "real": CustomMutation(eta = 20)
        })
        return _sampling, _crossover, _mutation
if __name__ == "__main__":
    from generate_graph import Graph
    from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
    from pymoo.factory import get_termination
    termination = MultiObjectiveSpaceToleranceTermination(tol=1,
                                                        n_last=25,
                                                        nth_gen=5,
                                                        n_max_gen=2000,
                                                        n_max_evals=None)
    num_of_collection_centers = 2
    set_seed = 1
    verbose = True 
    nsga3 = False
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=True, seed=set_seed, baseline_scaler=3)
    parameters = Parameters(RandomGraph, set_seed)
    three_objective_problem = Vectorized_heuristic(parameters)
    minimization = Minimize(problem = three_objective_problem, population_size = 5, termination = termination, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print(result.F, result.X)
    three_objective_problem = Elementwise_heuristic(parameters)
    minimization = Minimize(problem = three_objective_problem, population_size = 1000, number_of_generations = 400, verbose = verbose, nsga3 = nsga3)
    result = minimization.minimize_heuristic()
    print(result.F, result.X)