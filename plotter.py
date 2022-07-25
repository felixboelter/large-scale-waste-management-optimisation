from plotly import graph_objs as go
from typing import TypeVar, Dict,List, Any, Union, Tuple, Optional
import ast
import networkx as nx
from parameters import Parameters

class Plotter():
    def __init__(self, parameters : Parameters, solved_graph : nx.DiGraph, facility_sizes : dict) -> None:
        self._parameters = parameters
        self._solved_graph = solved_graph
        self._facility_sizes = facility_sizes
    def plot_graph(self, figure_name : Optional[str] = None) -> go.Figure:
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
        # if figure_name == None: figure_name= f"Solved solution for objective: {self.minimization}"
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
                                text= f"<b>Total Unsorted Supply:</b> {round(sum([us for us,_  in self._parameters.G.supplies.values()]), 3)}",
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
            x_edges, y_edges, colors, widths, name = _edge_colours(self._solved_graph,VALUES[i], VALUES[i+1])
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
        dictionary_values = {0 : "Small", 1: "Medium", 2 : "Large"}
        for node in self._solved_graph.nodes():
            node_name = self._parameters.G.node_translator[node]
            if node in self._parameters.G.collection_locations: 
                _node_colours[node] = ["Collection Center", '#D7D2CB']
                _custom_node_attrs[node] = f"Node: {node_name} Attr: {_node_colours[node][0]} <br> Unsorted Supply: {self._parameters.G.supplies[node][0]}"
            elif node in self._parameters.sorting_facilities: 
                _node_colours[node] = ["Sorting Facility", '#6AC46A']
                _custom_node_attrs[node] = f"Node: {node_name} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(self._facility_sizes[node_name])]}"
            elif node in self._parameters.incinerator_facilities: 
                _node_colours[node] = ["Incinerator Facility", '#952E25']
                _custom_node_attrs[node] = f"Node: {node_name} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(self._facility_sizes[node_name])]}"
            elif node in self._parameters.landfill_facilities: 
                _node_colours[node] = ["Landfill Facility", '#00C0F0']
                _custom_node_attrs[node] = f"Node: {node_name} Attr: {_node_colours[node][0]} <br> Size: {dictionary_values[int(self._facility_sizes[node_name])]}"
        
        seen_node_colours = []
        for node in self._solved_graph.nodes():
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