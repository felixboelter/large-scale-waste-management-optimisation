from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
import numpy as np
from parameters import Parameters
import networkx as nx
import pymoo
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems.constr_as_penalty import ConstraintsAsPenalty
from pymoo.core.repair import Repair


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
    def _evaluate(self, x, out, t = None, *args, **kwargs):
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
        _num_sorting_facilities = len(self._parameters._sorting_facilities)
        _num_collection_facilities = len(self._parameters._G.collection_locations)
        _num_incinerator_facilities = len(self._parameters._incinerator_facilities)
        _num_landfill_facilities = len(self._parameters._landfill_facilities)
        constraint_1 = np.abs(np.sum(x[:,self.continuous_ij_slice].reshape(-1,_num_sorting_facilities,_num_collection_facilities),axis=1) - self.supplies)
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
        if t != None:
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
    
    
class RepairFunction(Repair):
    def _set_additional_zero(self, Z : np.ndarray, problem : Union[ElementwiseProblem, Problem], indices : Tuple[np.ndarray, np.ndarray]):
        """
        It takes a matrix of solutions, and sets the values of the matrix to zero at the indices
        specified by the tuple
        
        :param Z: the decision variables
        :type Z: np.ndarray
        :param problem: the problem object
        :type problem: Union[ElementwiseProblem, Problem]
        :param indices: a tuple of two arrays, the first one is the row indices, the second one is the
        column indices
        :type indices: Tuple[np.ndarray, np.ndarray]
        """
        _jk_x_for_k = Z[:,problem.integer_jk_slice].reshape(Z.shape[0], -1, Z[:,problem.binary_incinerator_slice].shape[1]//3)
        _jkp_x_for_kp = Z[:,problem.integer_jkp_slice].reshape(Z.shape[0], -1, Z[:,problem.binary_landfill_slice].shape[1]//3)
        _jk_f_for_k = Z[:,problem.continuous_jk_slice].reshape(Z.shape[0], -1, Z[:,problem.binary_incinerator_slice].shape[1]//3)
        _jkp_f_for_kp = Z[:,problem.continuous_jkp_slice].reshape(Z.shape[0], -1, Z[:,problem.binary_landfill_slice].shape[1]//3)
        _jk_x_for_k[indices[0],:,indices[1]] = 0
        _jkp_x_for_kp[indices[0],:,indices[1]] = 0
        _jk_f_for_k[indices[0],:,indices[1]] = 0
        _jkp_f_for_kp[indices[0],:,indices[1]] = 0
    def _set_to_zero(self, Z : np.ndarray, y_slice : slice, x_link_slice : slice, f_link_slice : slice, problem : Union[ElementwiseProblem, Problem] = None):
        """
        > If the sum of the elements in the third dimension of the array is zero, then set the
        corresponding elements in the first two dimensions to zero
        
        :param Z: the decision variable matrix
        :type Z: np.ndarray
        :param y_slice: the slice of the Z vector that corresponds to the y variables
        :type y_slice: slice
        :param x_link_slice: the slice of the decision vector that corresponds to the x_ij values
        :type x_link_slice: slice
        :param f_link_slice: the slice of the decision vector that corresponds to the f_ij variables
        :type f_link_slice: slice
        :param problem: the problem we're solving
        :type problem: Union[ElementwiseProblem, Problem]
        :return: The return value is the Z matrix.
        """
        _y_sorting_for_j = Z[:,y_slice].reshape(Z.shape[0],-1,3)
        _ij_x_for_j = Z[:,x_link_slice].reshape(Z.shape[0], -1, Z[:,y_slice].shape[1]//3)
        _ij_f_for_j = Z[:,f_link_slice].reshape(Z.shape[0], -1, Z[:,y_slice].shape[1]//3)
        _indices_y_zero = np.where(np.sum(_y_sorting_for_j, axis=2) == 0)
        _indices_x_zero = np.where(_ij_x_for_j == 0)
        _indices_f_low = np.where(_ij_f_for_j < 0.5)
        if len(_indices_y_zero[0]) > 0 and Z[:,x_link_slice].shape[1] > 1:
            _ij_x_for_j[_indices_y_zero[0],:,_indices_y_zero[1]] = 0
            _ij_f_for_j[_indices_y_zero[0],:,_indices_y_zero[1]] = 0
            if problem != None: self._set_additional_zero(Z, problem, _indices_y_zero)
        if len(_indices_x_zero[0]) > 0 and Z[:,x_link_slice].shape[1] > 1:
            _ij_f_for_j[_indices_x_zero] = 0
        if len(_indices_f_low[0]) > 0 and Z[:,x_link_slice].shape[1] > 1:
            _ij_x_for_j[_indices_f_low] = 0
            _ij_f_for_j[_indices_f_low] = 0
        return Z
    def _repair_supply(self, Z : np.ndarray, y_slice : slice, f_link_slice : slice, supplies : np.ndarray, facility_sizes : np.ndarray):
        """
        For each population, we construct a network flow graph with a source node, a sink node, and a
        node for each facility. We then find the maximum flow from the source to the sink, and use the
        flow values to update the facility-population link variables
        
        :param Z: the current solution
        :type Z: np.ndarray
        :param y_slice: the slice of the Z matrix that corresponds to the y variables
        :type y_slice: slice
        :param f_link_slice: This is the slice of the Z matrix that corresponds to the facility links
        :type f_link_slice: slice
        :param supplies: the supply of each facility
        :type supplies: np.ndarray
        :param facility_sizes: the size of each facility
        :type facility_sizes: np.ndarray
        :return: The Z matrix with the repaired supply.
        """
        _y_sorting_for_j = Z[:,y_slice].reshape(Z.shape[0],-1,3)
        _ij_f_for_j = Z[:,f_link_slice].reshape(Z.shape[0], -1, Z[:,y_slice].shape[1]//3)
        y_ix = np.where(_y_sorting_for_j == 1)
        facility_sizes_for_y = _y_sorting_for_j*facility_sizes
        for pop in range(_ij_f_for_j.shape[0]):
            G = nx.DiGraph()
            lookup = np.where(y_ix[0] == pop)
            for i in range(0,len(supplies)):
                G.add_edge("s",i,capacity=supplies[i])
                for j,cap in enumerate(facility_sizes_for_y[y_ix][lookup]):
                    G.add_edge(i,j+len(supplies),capacity=supplies[i])
                    G.add_edge(j+len(supplies),"e", capacity=cap)
            _, flow_dict = nx.maximum_flow(G, "s", "e")
            facility_positions = y_ix[1][lookup]
            for i in range(len(supplies)):
                for ix, j in enumerate(flow_dict[i].values()):
                    _ij_f_for_j[pop,i,facility_positions[ix]] = j
        return Z
    def _repair_placements(self, Z : np.ndarray, y_slice : slice, facility_sizes : np.ndarray, supplies : np.ndarray):
        """
        If the sum of the facility sizes for a given customer is less than the supply, then add the
        smallest facility size that will make the sum equal to the supply
        
        :param Z: the solution matrix
        :type Z: np.ndarray
        :param y_slice: the slice of the Z matrix that corresponds to the y variables
        :type y_slice: slice
        :param facility_sizes: the sizes of the facilities
        :type facility_sizes: np.ndarray
        :param supplies: the amount of supplies that each facility can handle
        :type supplies: np.ndarray
        :return: the Z matrix with the repaired sorting placements.
        """
        
        y = Z[:,y_slice].reshape(Z.shape[0],-1,3)
        ix = np.where(np.sum(y,axis=2) > 1)
        y[ix] = 0 
        facility_sizes_for_y = y*facility_sizes
        smaller_than_sum = np.where(np.sum(np.sum(facility_sizes_for_y,axis=1),axis=1) < np.sum(supplies))
        while len(smaller_than_sum[0]) > 0:
            difference = np.ravel(np.sum(supplies) - np.sum(np.sum(facility_sizes_for_y[smaller_than_sum,:,:],axis=2),axis=2))
            for i,diff in enumerate(difference):
                if diff <= facility_sizes[0]:
                    x = np.sum(y[smaller_than_sum[0][i],:,:],axis=1).tolist().index(0)
                    y[smaller_than_sum[0][i],:,:][x,0] = 1
                elif diff <= facility_sizes[1]:
                    x = np.sum(y[smaller_than_sum[0][i],:,:],axis=1).tolist().index(0)
                    y[smaller_than_sum[0][i],:,:][x,1] = 1
                else:
                    x = np.sum(y[smaller_than_sum[0][i],:,:],axis=1).tolist().index(0)
                    y[smaller_than_sum[0][i],:,:][x,2] = 1
            facility_sizes_for_y = y*facility_sizes
            smaller_than_sum = np.where(np.sum(np.sum(facility_sizes_for_y,axis=1),axis=1) < np.sum(supplies))
            difference = np.ravel(np.sum(supplies) - np.sum(np.sum(facility_sizes_for_y[smaller_than_sum,:,:],axis=2),axis=2))
        return Z
    def _check_modify(self, Z : np.ndarray, y_slice : slice, facility_sizes : np.ndarray, supplies : np.ndarray):
        """
        If the sum of the facility sizes for a given y is less than the sum of the supplies, then we
        need to modify y.
        
        :param Z: the solution matrix
        :type Z: np.ndarray
        :param y_slice: the slice of the Z matrix that corresponds to the y variables
        :type y_slice: slice
        :param facility_sizes: the number of units of each facility type
        :type facility_sizes: np.ndarray
        :param supplies: the number of people that need to be assigned to a facility
        :type supplies: np.ndarray
        """
        y = Z[:,y_slice].reshape(Z.shape[0],-1,3)
        ix = np.where(np.sum(y,axis=2) > 1)
        y[ix] = 0 
        facility_sizes_for_y = y*facility_sizes
        _modify = np.where(np.sum(np.sum(facility_sizes_for_y,axis=1),axis=1) < np.sum(supplies))
        if len(_modify[0]) > 0:
            return True, _modify
        else:
            return False, None

    def _do(self, problem : Union[ElementwiseProblem, Problem], pop : np.ndarray, **kwargs):
        """
        It checks if the sorting facilities have the least number of supply, if they don't, it repairs the solution by setting the
        sorting facilitiy placements that are bigger than one to zero and then repairs the supply.
        
        :param problem: The problem object
        :type problem: Union[ElementwiseProblem, Problem]
        :param pop: the population of solutions
        :type pop: np.ndarray
        :return: The modified population.
        """
        
        Z = pop.get("X")
        _sorting_input = [problem.binary_sorting_slice, problem._parameters.facility_storage_capacities[0], problem.supplies]
        _incinerator_input = [problem.binary_incinerator_slice, problem._parameters.facility_storage_capacities[1], problem.supplies]
        _landfill_input = [problem.binary_landfill_slice, problem._parameters.facility_storage_capacities[2], problem.supplies]
        _bool, _modify = self._check_modify(Z, problem.binary_sorting_slice, problem._parameters.facility_storage_capacities[0], problem.supplies)
        if _bool:
            _z_modify = Z[_modify]
            for inputs in [_sorting_input, _incinerator_input, _landfill_input]:
                _z_modify = self._repair_placements(_z_modify, *inputs)
            _z_modify = self._repair_supply(_z_modify, problem.binary_sorting_slice, problem.continuous_ij_slice, problem.supplies, problem._parameters.facility_storage_capacities[0])
            _z_modify = self._set_to_zero(_z_modify, problem.binary_sorting_slice, problem.integer_ij_slice, problem.continuous_ij_slice, problem = problem)
            # _z_modify = self._set_to_zero(_z_modify, problem.binary_incinerator_slice, problem.integer_jk_slice, problem.continuous_jk_slice)
            # _z_modify = self._set_to_zero(_z_modify, problem.binary_landfill_slice, problem.integer_jkp_slice, problem.continuous_jkp_slice)
            Z[_modify] = _z_modify
        pop.set("X", Z)
        return pop

        

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
                    repair = RepairFunction(),
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
            "bin": get_crossover("bin_hux", prob=0.01),
            "int": get_crossover("int_two_point", prob=0.2),
            "real": get_crossover("real_two_point", prob=0.2)
        })

        _mutation = MixedVariableMutation(_masks, {
            "bin": get_mutation("bin_bitflip"),
            "int": get_mutation("int_pm"),
            "real": get_mutation("real_pm")
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