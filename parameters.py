from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
import numpy as np
from generate_graph import Graph
FND = TypeVar("FND", float, np.ndarray)

class Parameters():
    """
    Class Parameters. Creates a all the parameters used in the baseline.
    Parameter amounts based on: Olapiriyakul, Sun & Pannakkong, Warut &
    Kachapanya, Warith & Starita, Stefano. (2019). Multiobjective Optimization
    Model for Sustainable Waste Management Network Design. Journal of Advanced
    Transportation. 2019. 1-15. 10.1155/2019/3612809. 
    """
    def __init__(self, G: Graph, seed : int = 0) -> None:
        """
        Parameters Constructor. 
        Public access to parameters:
        facility_storage_capacities : np.ndarray, maximum_amount_transport : np.ndarray, 
        operational_costs : np.ndarray, land_stress_ratios : np.ndarray, 
        opening_costs : np.ndarray, link_populations : Dict[tuple, int],
        population_near_facilities : Dict[tuple, np.float64], link_dalys : Dict[tuple, float],
        facility_daly_per_person : Dict[tuple, List[float]]

        :param G: Graph object from generate_graph.py
        :type G: Graph
        :param seed: The seed to generate random numbers from, defaults to 0
        :type seed: int (optional)
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
        self.link_populations_list : Tuple[np.ndarray, np.ndarray, np.ndarray] = self._population_near_links_tolist(self.link_populations)
        self.population_near_facilities : Dict[tuple, np.ndarray] = self._create_population_near_facilities()
        self.population_list : Tuple[np.ndarray, np.ndarray, np.ndarray] = self._population_near_facilities_list(self.population_near_facilities)
        self.link_dalys : Dict[tuple, float] = self._create_DALY_for_links()
        self.link_dalys_list : Tuple[np.ndarray, np.ndarray, np.ndarray] = self.DALY_for_links_tolist(self.link_dalys)
        self.facility_daly_per_person : Dict[tuple, List[float]] = self._create_DALY_for_facilities()

    def _create_DALY_for_facilities(self) -> Dict[tuple, List[float]]:
        """
        This function creates a dictionary of DALYs per person for each facility of length l
        :return: A dictionary with the keys being the nodes and the values being the DALYs per person
        for all facilities of length l.
        """
        self.facility_daly_matrix = np.array([[0.07, 0.14, 0.28],
                                [5.95, 11.9, 17.85],
                                [3.89, 7.78, 11.66]])
        daly_per_person = dict()
        for ix, facilities in enumerate([self._sorting_facilities, self._incinerator_facilities, self._landfill_facilities]): 
            for node in facilities:
                daly_per_person.update({node: [self.facility_daly_matrix[ix][l] for l in self._range_of_facility_sizes]})
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
    
    def DALY_for_links_tolist(self, link_daly : Dict[tuple, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        It takes a dictionary of link dalys and returns three lists of link dalys: one for sorting
        links, one for incinerator links, and one for landfill links.
        
        :param link_daly: a dictionary of tuples (key) and floats (value)
        :type link_daly: Dict[tuple, float]
        :return: the dalys for the links between the supplies and the sorting facilities, the dalys for
        the links between the sorting facilities and the incinerators, and the dalys for the links
        between the sorting facilities and the landfills.
        """
        _sorting_link_dalys = np.array([val[1] for key, val in link_daly.items() if key[0] in self._G.supplies.keys() and key[1] in self._sorting_facilities])
        _incinerator_link_dalys = np.array([val[1] for key, val in link_daly.items() if key[0] in self._sorting_facilities and key[1] in self._incinerator_facilities ])
        _landfill_link_dalys = np.array([val[1] for key, val in link_daly.items() if (key[0] in self._sorting_facilities and key[1] in self._landfill_facilities) ])
        return _sorting_link_dalys, _incinerator_link_dalys, _landfill_link_dalys

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
        EQUIPMENT_COST = 750000
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
    def _population_near_links_tolist(self, link_populations: Dict[tuple,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper function. Converts a dictionary of link_populations to lists of link populations.
        :param link_populations: Link populations created in _create_population_near_links()
        :return: The three lists for sorting, incinerator, and landfill link populations.
        """
        _sorting_link_population = np.array([val for key, val in link_populations.items() if (key[0] in self._G.supplies.keys() and key[1] in self._sorting_facilities)])
        _incinerator_link_population = np.array([val for key, val in link_populations.items() if (key[0] in self._sorting_facilities and key[1] in self._incinerator_facilities)])
        _landfill_link_population = np.array([val for key, val in link_populations.items() if (key[0] in self._sorting_facilities and key[1] in self._landfill_facilities)])
        return _sorting_link_population, _incinerator_link_population, _landfill_link_population
    
    def _create_population_near_facilities(self) -> Dict[tuple, np.ndarray]:
        """
        Helper function. Creates population for all nodes in graph G. 
        Based on the population at node i and the amount of space the facility is using up in km^2.
        :return: Subpopulation for all nodes in graph G.
        :rtype: Dictionary[(i,j), np.ndarray]
        """
        people_living_near_facility = dict()
        m_sqr_to_km_sqr = lambda m_sqr: m_sqr / 1e6
        for node in self._G.G.nodes: people_living_near_facility.update({node : np.round(self._G.city_population[node] * m_sqr_to_km_sqr(self._direct_land_usage))})
        self._G.custom_parameters['people_living_near_facility'] = people_living_near_facility
        return people_living_near_facility
    
    def _population_near_facilities_list(self, population_dict: Dict[tuple, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper function. Creates three lists from the population dictionary, used for the metaheuristic.
        :param population_dict: dictionary created using the function _create_population_near_facilities().
        :return: Three lists for sorting, incinerator, and landfill population near facilities.
        """
        _sorting_population = np.array([val[0] for key, val in population_dict.items() if key in self._sorting_facilities])
        _incinerator_population = np.array([val[1] for key, val in population_dict.items() if key in self._incinerator_facilities])
        _landfill_population = np.array([val[2] for key, val in population_dict.items() if key in self._landfill_facilities])
        return _sorting_population, _incinerator_population, _landfill_population
if __name__ == '__main__':
    from generate_graph import Graph
    num_of_collection_centers = 5
    set_seed = 1
    verbose = True 
    nsga3 = False
    RandomGraph = Graph(num_of_collection_centers,baseline=True,plot_graph=False, seed=set_seed, baseline_scaler=3)
    parameters = Parameters(RandomGraph, set_seed)