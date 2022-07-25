import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import random
from typing import TypeVar,Dict, List, Tuple, Any

StrInt = TypeVar('StrInt', str, int)
TupleInt = TypeVar('TupleInt', tuple, int)
ListInt = TypeVar('ListInt', list, int)
IntFloat = TypeVar("IntFloat", int, float)

class Graph():
    """
    The Graph class is a container for a set of vertices and a set of edges..
    """
    def __init__(self, 
        
                number_of_cities: int, 
                number_list: List[int] = [0,0,0], 
                baseline: bool = False, 
                plot_graph : bool = False, 
                seed : IntFloat = 1,
                baseline_scaler : int = 3) -> None:
        """
        Graph constructor Public access to G (Networkx graph), special_locations
        (dict), supplies (dict), demands (dict), custom_parameters (dict),
        city_population (dict), and city_sizes (dict)

        :param number_of_cities: The number of cities to create
        :type number_of_cities: int
        :param number_list: [Num. of demand cities, Num. of incinerator cities, Num. of recycling
        cities]
        :type number_list: List[int]
        :param baseline: Use the baseline (Optional): Olapiriyakul, Sun & Pannakkong, Warut &
        Kachapanya, Warith & Starita, Stefano. (2019), defaults to False
        :type baseline: bool (optional)
        :param plot_graph: To plot the graph when creating a new graph (Optional), defaults to False
        :type plot_graph: bool (optional)
        :param seed: The seed for the random number generator, defaults to 1
        :type seed: IntFloat (optional)
        :param baseline_scaler: This is the scaler for the baseline. The baseline is a graph that is
        created by the paper. The scaler is used to scale the graph, defaults to 3
        :type baseline_scaler: int (optional)
        """
        super().__init__()
        assert all(x <= number_of_cities and x >= 0 for x in number_list), (f"All numbers in the number list must be less than or equal to the number of cities: {number_of_cities} and greater than or equal to zero. Got {number_list}.")
        assert baseline_scaler > 0, f"Baseline scaler must be strictly positive and greater than 0. Got {baseline_scaler}"
        self._baseline = baseline
        self._scaler = baseline_scaler
        self._number_of_cities = number_of_cities
        self._number_list = number_list
        self._value_strings = ['D', 'I', 'R']
        self._city_supplies = dict()
        self._demand_ranges = (20,80)
        self._supply_ranges = (10,40)
        self._locations = tuple()
        if baseline: 
            self._value_strings = ['J','K',"K'"]
            self.collection_locations = list()
        self.special_locations = dict() # key = node : value = special value
        self.supplies = dict() # key = node : value = [unsorted supply, sorted supply]
        self.demands= dict() # key = node : value = demand value
        self.custom_parameters= dict() # key = parameter name: value = numbers of parameter
        self.city_population= dict() #key = city (tuple) : value = population (int)
        self.city_sizes= dict() #key = city (tuple) : value = city size (float)
        self.node_translator= dict() #key = node location : value = node number
        self.G = None
        np.random.seed(seed)
        self.create_graph(plot_graph=plot_graph)
    def add_custom_parameter(self, 
                    name: StrInt,
                    size: TupleInt = 1,
                    low: int = 10, 
                    high: int = None,
                    random: bool = False,
                    integer: bool = False, 
                    fixed_number: ListInt = 0) -> Dict[StrInt,Any]:
        """
        Add a custom parameter to the dictionary self.custom_parameters.

        :param name: Name of the dictionary entry
        :type name: StrInt
        :param size: The size of the parameter, defaults to 1
        :type size: TupleInt (optional)
        :param low: The lower bound of the random number generation, defaults to 10
        :type low: int (optional)
        :param high: The highest value that the random number can be
        :type high: int
        :param random: If random number generation should be used. (Optional), defaults to False
        :type random: bool (optional)
        :param integer: If the random number generation should use Integers or Floats. (Optional),
        defaults to False
        :type integer: bool (optional)
        :param fixed_number: The fixed number/numbers that should be used in the parameter. (Optional),
        defaults to 0
        :type fixed_number: ListInt (optional)
        :return: The custom parameter which is added to the dictionary.
        """

        if random:
            if integer:
                numbers = np.random.randint(low = low, high = high, size=size, dtype = int)
            else:
                numbers = np.random.random_sample(size=size) * (high - low) + low
        else:
            if isinstance(fixed_number, list) and np.array(fixed_number).size == np.prod(size): numbers = np.array(fixed_number).reshape(size)
            elif isinstance(fixed_number, list):
                numbers = np.zeros(shape=size, dtype = float)
                for ix in range(len(fixed_number)):
                    numbers[..., ix] = fixed_number[ix]
            else:
                numbers = np.zeros(shape=size,dtype=type(fixed_number))+fixed_number
        self.custom_parameters.update({name: numbers})
        return self.custom_parameters[name]
    def plot_graph(self):
        """
        Function plot_graph
        Using plotly to plot the randomized graph.

        :return: Returns a plotly Figure object.
        """
        def _edge_colours(value1: int, value2: int):
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
            for i,j,w in self.G.edges(data=True):
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
        node_x = [_x for _x, __ in self.G.nodes()]
        node_y = [_y for __, _y in self.G.nodes()]
        node_colours = [["Node", '#D7D2CB'] for __ in range(len(self.G.nodes))]

        if not self._baseline:
            custom_node_attrs = [f"Node: {_i+1} <br> Unsorted Supply: {self.supplies[_v][0]} <br> Sorted Supply: {self.supplies[_v][1]}" for _i,_v in enumerate(self.G.nodes)]
            for ix, kv in enumerate(self.special_locations.items()):
                node = kv[0]
                attr = kv[1]
                custom_node_attrs[ix] = f"Node: {ix+1} Attr: {attr} <br> Unsorted Supply: {self.supplies[node][0]} <br> Sorted Supply: {self.supplies[node][1]}"
                if 'D' in attr:
                    custom_node_attrs[ix] = f"Node: {ix+1} Attr: {attr} <br> Demand: {self.demands[node]} <br>Unsorted Supply: {self.supplies[node][0]} <br> Sorted Supply: {self.supplies[node][1]}"
                if len(attr) == 3:
                    node_colours[ix] = ["D, I, R",'#D95D67']
                elif 'D' in attr and len(attr) == 2:
                    node_colours[ix] = ["D, I/R",'#C5B4E3']
                elif 'I' in attr and 'R' in attr:
                    node_colours[ix] = ["I, R",'#FFB673']
                elif 'I' in attr:
                    node_colours[ix] = ["I", '#952E25']
                elif 'D' in attr:
                    node_colours[ix] = ["D", '#00C0F0']
                elif 'R' in attr:
                    node_colours[ix] = ["R", '#6AC46A']
        elif self._baseline:
            self.demands.update({i: 0 for i in self.G.nodes})
            custom_node_attrs = [f"Node: {_i+1}" for _i in range(0,len(self.G.nodes))]
            for ix, kv in enumerate(self.special_locations.items()):
                index_displaced = ix+self._number_of_cities
                node = kv[0]
                attr = kv[1]
                self.node_translator.update({node: index_displaced + 1})
                if 'K' in attr:
                    node_colours[index_displaced] = ["Incinerator Candidate", '#952E25']
                elif "K'" in attr:
                    node_colours[index_displaced] = ["Landfill Candidate", '#00C0F0']
                elif 'J' in attr:
                    node_colours[index_displaced] = ["Sorting Candidate", '#6AC46A']
                custom_node_attrs[index_displaced] = f"Node: {index_displaced+1} Attr: {node_colours[index_displaced][0]}"
            for ix in range(0,len(self.collection_locations)):
                node = self.collection_locations[ix]
                self.node_translator.update({node: ix + 1})
                node_colours[ix] = ["Collection Center", '#D7D2CB']
                custom_node_attrs[ix] = f"Node: {ix+1} Attr: {node_colours[ix][0]} <br> Unsorted Supply: {self.supplies[node][0]} <br> Sorted Supply: {self.supplies[node][1]}"

        fig = go.Figure(layout=go.Layout(
                        title="Incinerator, Recycling, Demand Randomised Graph",
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
                                text=f" <b>Total Demand:</b> {sum(self.demands.values())}" + \
                                    f"<br> <b>Total Unsorted Supply:</b> {np.round(sum([us for us,_  in self.supplies.values()]), 3)}" + \
                                    f"<br> <b>Total Sorted Supply:</b> {sum([s for _, s in self.supplies.values()])} ",
                                showarrow=False,
                                align = 'left',
                                # xref="paper", yref="paper",
                                x=0.005, y=-0.002 ) ],
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                        )
        # Add edge colours and edge lines to plot
        for i in range(CATEGORIES):
            x_edges, y_edges, colors, widths, name = _edge_colours(VALUES[i], VALUES[i+1])
            fig.add_trace(go.Scatter(
                x=x_edges, y=y_edges,
                showlegend=True,
                name= name,
                line = dict(
                    color = colors,
                    width = widths),
                hoverinfo='none',
                mode='lines'))
        # Add node colours and nodes to plot
        seen_node_colours = []
        for i in range(len(node_x)):
            if node_colours[i] not in seen_node_colours:
                temp_x = []
                temp_y = []
                temp_attr = []
                current_node_colour = node_colours[i]
                seen_node_colours.append(current_node_colour)
                for j in range(0, len(node_colours)):
                    if node_colours[j] == current_node_colour:
                        temp_x.append(node_x[j])
                        temp_y.append(node_y[j])
                        temp_attr.append(custom_node_attrs[j])
                fig.add_trace(go.Scatter(
                    x=temp_x, y=temp_y,
                    mode='markers',
                    hoverinfo='text',
                    text = temp_attr,
                    name=f"{node_colours[i][0]}",
                    marker=dict(
                        color=node_colours[i][1],
                        size=20,
                        line_width=2)))
        return fig
    def create_graph(self, plot_graph: bool = False):
        """
        Public create_graph Function.
        Creates a randomised graph. This uses K-means to find the various X,Y locations of the cities.
        Libraries used are networkx, numpy, and scikit-learn.
        :param plot_graph: Will this function call the plot_graph function for plotting the generated graph.
        :type plot_graph: boolean
        :return: 
        """
        def _add_edge_to_graph(G: nx.Graph(), e1: Tuple, e2: Tuple, w: float) -> None:
            """
            Adds an edge to a networkx graph
            :param G: Graph to add an edge
            :param e1: The first node X,Y coordinates
            :param e2: The second node X,Y coordinates
            :param w: Weight of the edge
            :return:
            """
            G.add_edge(e1, e2, weight=w)
        def _create_special_locations(nodes: List[List[int]], value: List[str]) -> Tuple[Dict, Dict]:
            """
            Create the Incinerator, Recycling, and Demand node locations. Aswell as generating the numbers for the demand nodes.
            :param nodes: The nodes which are split in [[Demand], [Incinerator], [Recycling]] or [[Sorting], [Incinerator], [Landfill], [Collection]]
            :param value: The value strings of the split (Either D,I,R or J,K,K',I)
            :type nodes: 2-D list
            :type value: 1-D list
            :return locs: special node locations for Incinerator nodes, Recycling nodes, and Demand nodes.
            :return demands: Demands in the shape (node: demand amount)
            """
            locs = dict()
            demands = dict()
            if not self._baseline:
                _demands = _generate_numbers(low= self._demand_ranges[0], high= self._demand_ranges[1], n= self._number_list[0])
            for i in range(0, len(value)):
                for node_number in nodes[i]:
                    node_tuple = self._locations[node_number]
                    if node_tuple not in locs:
                        locs[node_tuple] = [value[i]]
                    else:
                        locs[node_tuple].append(value[i])
                    if value[i] == value[0] and not self._baseline: 
                        if isinstance(_demands, list):
                            _random_number = _demands.pop()
                        else:
                            _random_number = _demands
                        demands.update({node_tuple: _random_number})
                        
            return locs, demands
        def _generate_numbers(
                            high: int, 
                            low: int = 0, 
                            n: int = 1):
            """
            Private _generate_numbers helper function to generate integer numbers between two values.
            :param high: The highest value to generate (not included)
            :param low: The lowest value to generate (included) (Default: 0)
            :param n: Number of values to generate (Default: 1)
            :return numbers: The number/numbers generated.
            :rtype: int/list
            """
            if n == 1: numbers = np.random.random_sample() * (high - low) + low
            else: numbers = [ np.random.random_sample() * (high - low) + low for _ in range(n)]
            return np.round(numbers, 3)
        def _generate_city_sizes() -> dict:
            """
            The function takes the population of each city and divides it by the density of the city
            to get the area of the city
            :return: A dictionary of city names and their respective sizes.
            """
            DENSITY = 4800 # Population/KM^2
            #pop/density = Area
            _city_sizes = dict()
            for index in range(0, self._number_of_cities):
                city = self._locations[index]
                _city_sizes.update({city : round(self.city_population[city] / DENSITY, 3)})
            return _city_sizes


        def _generate_city_population() -> dict:
            """
            The function generates a dictionary of city populations based on the number of nodes in
            the graph, the number of nodes in the special locations, and a random number generator
            :return: A dictionary of the city names and their populations.
            """
            POPULATION_SCALE = 45000
            HIGH_FIXED = 80000
            LOW_FIXED = 35000
            _populations = dict()
            for nodes in range(0, len(self.G.nodes)): _populations.update({self._locations[nodes] : round(np.random.random_sample() * (HIGH_FIXED - LOW_FIXED) + LOW_FIXED)})
            
            for key, value in self.special_locations.items():
                length_value = len(value)
                if 'D' in value: length_value -= 1
                if length_value == 1: length_value = np.sqrt(3)
                random_number = (((length_value**2)*POPULATION_SCALE)*np.random.random_sample()) + _populations[key]
                _populations.update({key : round(random_number)})
            return _populations
        # TODO: create same graph locations/supplies for both baseline and new versions (ofc after finishing the baseline)
        self.G = nx.DiGraph()
        # Create random points in range 0 - 100.
        pts = 100*np.random.random((self._number_of_cities*20,2))
        # Apply K-means to the random points.
        if self._baseline: scaler = self._scaler
        else: scaler = 1
        kmean = KMeans(n_clusters=self._number_of_cities*scaler).fit(pts)
        # The centroids are the nodes of the cities, as such they are spaced out.
        self._locations = tuple(map(tuple,kmean.cluster_centers_.astype(int)))
        # Get the distance for all cities between all cities as our cost edges.
        for i in range(len(self._locations)):
            for j in range(len(self._locations)):
                # Eucledian distance calculation.
                distance = ((self._locations[i][0] - self._locations[j][0])**2 + (self._locations[i][1] - self._locations[j][1])**2)**0.5
                # Add the edge to a graph with the distance as an edge weight.
                _add_edge_to_graph(self.G, self._locations[i], self._locations[j], distance)
        # Generate the supply amount for each node.
        if not self._baseline:    
            for i in range(0,self._number_of_cities):
                _rand_num = _generate_numbers(low= self._supply_ranges[0], high = self._supply_ranges[1], n= 1)
                self.supplies.update({self._locations[i] :[_rand_num, _rand_num//3]})
            # Generate Demand, Incinerator, and Recycling node locations
            special_nodes = [random.sample(range(0,self._number_of_cities), self._number_list[i]) for i in range(len(self._number_list))]
            self.special_locations, self.demands = _create_special_locations(special_nodes, self._value_strings)
        elif self._baseline:
            # Creating a list of the collection locations, and then creating a list of the special locations.
            self.collection_locations = [self._locations[i] for i in range(0,self._number_of_cities)]
            special_cities = [self._locations[i] for i in range(self._number_of_cities, len(self.G.nodes))]
            splits = []
            _temp = []
            difference = (len(self.G.nodes) - self._number_of_cities)%3
            for ix in range(self._number_of_cities,len(self.G.nodes)):
                if ix > len(self.G.nodes) - difference - 1 and difference != 0: splits[np.random.randint(0,3)].append(ix)
                else: _temp.append(ix)
                if len(_temp) == int(len(special_cities)/3): splits.append(_temp); _temp = []
            self.special_locations, self.demands = _create_special_locations(splits, self._value_strings)
            for loc in self.collection_locations:
                _rand_num = _generate_numbers(low= self._supply_ranges[0], high = self._supply_ranges[1], n= 1)
                self.supplies.update({loc: [_rand_num, 0]})
        self.city_population = _generate_city_population()
        self.city_sizes = _generate_city_sizes()
        # If plot_graph = True. We create a plot of the generated graph
        if plot_graph:
            figure = self.plot_graph()
            figure.show()
if __name__ == "__main__":
    RG = Graph(12,[2,2,11])
    Graph = RG.create_graph(plot_graph=True)
        