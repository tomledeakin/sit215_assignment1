#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import time
import problem
import importlib
importlib.reload(problem)
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import lines
from problem import Problem, Node, GraphProblem, Graph, UndirectedGraph
import math
import numpy as np
import heapq
import functools
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import heapq
import functools
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import time

# %matplotlib inline


# In[9]:


# --------------------------
# TASK 1: ENVIRONMENT CREATION AND PROBLEM FORMULATION (Advanced)
# --------------------------
# Simulate a wheelchair-friendly campus map with at least 20-25 unique path segments
# and key amenities: wheelchair-accessible restrooms, parking spaces, shops, and lifts.

# Define nodes and coordinates (for visualization)
campus_locations = {
    "Entrance": (0, 0),
    "Building_A": (1, 1),
    "Library": (2, 2),
    "Cafeteria": (0.5, 2),
    "Auditorium": (3, 2),
    "Parking": (-1, 0.5),
    "Building_B": (3, 1),
    "Building_C": (1, 3),
    "Garden": (1.5, 4),
    "Gym": (4, 3),
    "Dormitory": (3, 4),
    "Administration": (2, 0),
    "Medical_Center": (4, 0.5),
    "Recreation_Center": (1, 5),
    # Additional key amenities:
    "Accessible_Restroom": (2.5, 3.5),  # Wheelchair-accessible restroom
    "Shop": (1.8, 1.2),                # Shop
    "Lift": (1.2, 2.8)                 # Lift
}

# Define the campus graph with around 25 unique path segments (using estimated connections)
campus_graph_dict = {
    "Entrance": {"Building_A": 0.5, "Parking": 0.7, "Administration": 1.0, "Lift": 1.0},
    "Building_A": {"Entrance": 0.5, "Library": 0.3, "Auditorium": 0.8, "Shop": 0.7, "Cafeteria": 0.6},
    "Library": {"Building_A": 0.3, "Auditorium": 0.8, "Building_C": 0.5, "Accessible_Restroom": 0.4},
    "Cafeteria": {"Building_A": 0.6},
    "Auditorium": {"Building_A": 0.8, "Building_B": 0.7},
    "Parking": {"Entrance": 0.7, "Building_B": 0.6, "Dormitory": 0.9},
    "Building_B": {"Parking": 0.6, "Auditorium": 0.7, "Gym": 0.8},
    "Building_C": {"Library": 0.5, "Dormitory": 0.7, "Garden": 0.6},
    "Garden": {"Building_C": 0.6, "Gym": 0.5},
    "Gym": {"Building_B": 0.8, "Garden": 0.5, "Dormitory": 0.7},
    "Dormitory": {"Building_C": 0.7, "Parking": 0.9, "Gym": 0.7},
    "Administration": {"Entrance": 1.0, "Medical_Center": 0.6},
    "Medical_Center": {"Administration": 0.6, "Recreation_Center": 0.8},
    "Recreation_Center": {"Medical_Center": 0.8},
    "Shop": {"Building_A": 0.7, "Library": 0.5},
    "Lift": {"Entrance": 1.0, "Accessible_Restroom": 0.9},
    "Accessible_Restroom": {"Library": 0.4}  # Fewer connections to achieve around 25 segments
}

# Count unique path segments (for an undirected graph)
unique_edges = set()
for node, neighbors in campus_graph_dict.items():
    for neighbor, cost in neighbors.items():
        edge = tuple(sorted((node, neighbor)))
        unique_edges.add(edge)

print("Total unique path segments:", len(unique_edges))

# Create an undirected graph and assign coordinates
campus_map = UndirectedGraph(campus_graph_dict)
campus_map.locations = campus_locations

# --------------------------
# Visualization setup: Define colors for key amenities
# --------------------------
node_colors = {}
for node in campus_locations.keys():
    if "Restroom" in node:
        node_colors[node] = "red"        # Wheelchair-accessible restroom
    elif node == "Parking":
        node_colors[node] = "orange"       # Parking
    elif node == "Shop":
        node_colors[node] = "blue"         # Shop
    elif node == "Lift":
        node_colors[node] = "purple"       # Lift
    else:
        node_colors[node] = "white"        # Other nodes

# Define label positions (can be adjusted for better visualization)
node_positions = campus_map.locations
node_label_pos = { k: [v[0], v[1]-0.1] for k, v in node_positions.items() }

# Create edge weights from the graph
edge_weights = {(k, k2): v2 for k, v in campus_graph_dict.items() for k2, v2 in v.items()}

# Bundle data for visualization
campus_graph_data = {
    'graph_dict': campus_graph_dict,
    'node_colors': node_colors,
    'node_positions': node_positions,
    'node_label_positions': node_label_pos,
    'edge_weights': edge_weights
}

# --------------------------
# Visualization function using networkx and matplotlib
# --------------------------
def show_map(graph_data, node_colors=None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights = graph_data['edge_weights']

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()},
            node_color=[node_colors[node] for node in G.nodes()],
            linewidths=1, edgecolors='k', with_labels=True, node_size=800)
    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=12)

    # Add legend for markers
    red_circle = lines.Line2D([], [], marker='o', color='w', label='Accessible Restroom',
                               markerfacecolor='red', markersize=15)
    orange_circle = lines.Line2D([], [], marker='o', color='w', label='Parking',
                                  markerfacecolor='orange', markersize=15)
    blue_circle = lines.Line2D([], [], marker='o', color='w', label='Shop',
                                markerfacecolor='blue', markersize=15)
    purple_circle = lines.Line2D([], [], marker='o', color='w', label='Lift',
                                  markerfacecolor='purple', markersize=15)
    white_circle = lines.Line2D([], [], marker='o', color='w', label='Other',
                                 markerfacecolor='white', markersize=15)
    plt.legend(handles=[red_circle, orange_circle, blue_circle, purple_circle, white_circle],
               loc='upper right', fontsize=12)
    plt.title("Campus Environment with Wheelchair-Accessible Routes and Amenities", fontsize=16)
    plt.tight_layout()
    plt.show()

# Display the campus map with key amenity markers
show_map(campus_graph_data)


# In[10]:


# ---------- PART 1: Representing the Environment as an Adjacency Matrix ----------
def create_adjacency_matrix(graph):
    """
    Converts an undirected graph (in dictionary format) to an adjacency matrix.
    Returns a sorted list of nodes and a numpy matrix containing the cost between nodes.
    If there is no connection, the value is math.inf.
    """
    nodes = sorted(graph.nodes())  # Sort nodes to ensure consistent ordering
    n = len(nodes)
    matrix = np.full((n, n), math.inf)
    for i in range(n):
        matrix[i, i] = 0
    for i, node in enumerate(nodes):
        for j, neighbor in enumerate(nodes):
            cost = graph.get(node, neighbor)
            if cost is not None:
                matrix[i, j] = cost
    return nodes, matrix

def plot_adjacency_matrix(nodes, matrix):
    """
    Plots a heatmap of the adjacency matrix. Values of math.inf are masked and displayed as white.
    """
    masked_matrix = np.ma.masked_where(np.isinf(matrix), matrix)
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    
    plt.figure(figsize=(5,5))
    plt.imshow(masked_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar(label="Cost")
    plt.xticks(range(len(nodes)), nodes, rotation=90)
    plt.yticks(range(len(nodes)), nodes)
    plt.title("Adjacency Matrix of the Campus Graph")
    plt.tight_layout()
    plt.show()

# ---------- PART 2: Implementing the A* and Uniform Cost Search Algorithms ----------

def memoize(fn, slot=None, maxsize=32):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)
    return memoized_fn

class PriorityQueue:
    """
    Priority Queue with a counter to ensure unique ordering when f(x) values are equal.
    """
    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        self.counter = itertools.count()  # Unique counter for tie-breaker
        if order == 'min':
            self.f = f
        elif order == 'max':
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), next(self.counter), item))

    def extend(self, items):
        for item in items:
            self.append(item)

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[2]
        else:
            raise Exception("Trying to pop from empty PriorityQueue.")

    def __len__(self):
        return len(self.heap)

    def __contains__(self, key):
        return any(item == key for _, __, item in self.heap)

    def __getitem__(self, key):
        for value, count, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        try:
            index = [item == key for _, __, item in self.heap].index(True)
            del self.heap[index]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

def best_first_graph_search_for_vis(problem, f):
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}
    f = memoize(f, 'f')
    node = Node(problem.initial)
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return iterations, all_node_colors, node
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    explored = set()
    while frontier:
        node = frontier.pop()
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return iterations, all_node_colors, node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < incumbent:
                    del frontier[child]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = best_first_graph_search_for_vis(problem, lambda n: n.path_cost + h(n))
    return iterations, all_node_colors, node

def uniform_cost_search_graph(problem):
    iterations, all_node_colors, node = best_first_graph_search_for_vis(problem, lambda n: n.path_cost)
    return iterations, all_node_colors, node

# ---------- PART 3: Visualization Functions ----------
def show_map(graph_data, node_colors=None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights = graph_data['edge_weights']
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()},
            node_color=[node_colors[node] for node in G.nodes()],
            linewidths=0.3, edgecolors='k', with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=10)
    plt.show()

def final_path_colors(initial_node_colors, problem, solution):
    final_colors = dict(initial_node_colors)
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors

# ---------- PART 4: Prepare Data for Visualization ----------
# Using the campus_map defined in Task 1 (the campus environment)
node_colors = {node: 'white' for node in campus_map.locations.keys()}
node_positions = campus_map.locations
node_label_pos = {k: [v[0], v[1]-0.1] for k, v in node_positions.items()}
edge_weights = {(k, k2): v2 for k, v in campus_map.graph_dict.items() for k2, v2 in v.items()}

campus_graph_data = {
    'graph_dict': campus_map.graph_dict,
    'node_colors': node_colors,
    'node_positions': node_positions,
    'node_label_positions': node_label_pos,
    'edge_weights': edge_weights
}

# ---------- PART 5: Testing with Different Start-End Pairs ----------
test_cases = [
    ("Entrance", "Recreation_Center"),
    ("Building_A", "Gym"),
    ("Parking", "Library"),
    ("Dormitory", "Medical_Center"),
    ("Cafeteria", "Auditorium"),
    ("Accessible_Restroom", "Shop")  # Instead of "Lounge"
]

def test_search_algorithms(test_cases, search_algo, problem_class=GraphProblem, graph=campus_map, description="A* Search"):
    for start, goal in test_cases:
        prob = problem_class(start, goal, graph)
        iterations, node_colors_list, node = search_algo(prob)
        path = node.solution()
        cost = node.path_cost
        print(f"{description} from '{start}' to '{goal}':")
        print("  Path:", path)
        print("  Total cost:", cost)
        print("  Iterations:", iterations)
        print("-" * 50)

print("----- A* Search Test Results (Task 2) -----")
test_search_algorithms(test_cases, astar_search, problem_class=GraphProblem, graph=campus_map, description="A* Search")

print("----- Dijkstra/Uniform Cost Search Test Results (Task 2) -----")
test_search_algorithms(test_cases, uniform_cost_search_graph, problem_class=GraphProblem, graph=campus_map, description="Dijkstra/Uniform Cost Search")

# ---------- PART 6: Display the Adjacency Matrix ----------
nodes_list, adj_matrix = create_adjacency_matrix(campus_map)
print("Nodes:")
print(nodes_list)
print("\nAdjacency Matrix:")
print(adj_matrix)
plot_adjacency_matrix(nodes_list, adj_matrix)


# In[11]:


# -----------------------------------
# EXPANDED ENVIRONMENT: Create an extended graph with ≥30 unique path segments
# -----------------------------------
campus_graph_dict_extended = {
    "Entrance": {"Building_A": 0.5, "Parking": 0.7, "Administration": 1.0, "Lift": 1.0},
    "Building_A": {"Entrance": 0.5, "Library": 0.3, "Cafeteria": 0.6, "Auditorium": 0.8, "Shop": 0.7},
    "Library": {"Building_A": 0.3, "Auditorium": 0.8, "Building_C": 0.5, "Accessible_Restroom": 0.4, "Shop": 0.5},
    "Cafeteria": {"Building_A": 0.6, "Auditorium": 0.4},
    "Auditorium": {"Building_A": 0.8, "Building_B": 0.7, "Cafeteria": 0.4, "Dormitory": 0.9, "Recreation_Center": 1.0},
    "Parking": {"Entrance": 0.7, "Building_B": 0.6, "Dormitory": 0.9},
    "Building_B": {"Parking": 0.6, "Auditorium": 0.7, "Gym": 0.8, "Administration": 0.7, "Dormitory": 0.9},
    "Building_C": {"Library": 0.5, "Dormitory": 0.7, "Garden": 0.6, "Accessible_Restroom": 0.5},
    "Garden": {"Building_C": 0.6, "Gym": 0.5, "Recreation_Center": 0.4},
    "Gym": {"Building_B": 0.8, "Garden": 0.5, "Dormitory": 0.7},
    "Dormitory": {"Building_B": 0.9, "Gym": 0.7, "Building_C": 0.7, "Administration": 0.8, "Medical_Center": 0.9},
    "Administration": {"Entrance": 1.0, "Building_B": 0.7, "Dormitory": 0.8, "Medical_Center": 0.6},
    "Medical_Center": {"Administration": 0.6, "Recreation_Center": 0.8, "Dormitory": 0.9},
    "Recreation_Center": {"Medical_Center": 0.8, "Garden": 0.4, "Auditorium": 1.0},
    "Shop": {"Building_A": 0.7, "Library": 0.5, "Auditorium": 0.6},
    "Lift": {"Entrance": 1.0, "Accessible_Restroom": 0.9},
    "Accessible_Restroom": {"Library": 0.4, "Building_C": 0.5}
}

# Use a set to count unique path segments
unique_edges = set()
for node, neighbors in campus_graph_dict_extended.items():
    for neighbor in neighbors.keys():
        edge = tuple(sorted((node, neighbor)))
        unique_edges.add(edge)
print("Total unique path segments (extended environment):", len(unique_edges))
# Uncomment the following line if you want to print all unique edges:
# print("Unique path segments:", unique_edges)

# -----------------------------------
# Define coordinates for the nodes (campus environment)
# -----------------------------------
campus_locations_extended = {
    "Entrance": (0, 0),
    "Building_A": (1, 1),
    "Library": (2, 2),
    "Cafeteria": (0.5, 2),
    "Auditorium": (3, 2),
    "Parking": (-1, 0.5),
    "Building_B": (3, 1),
    "Building_C": (1, 3),
    "Garden": (1.5, 4),
    "Gym": (4, 3),
    "Dormitory": (3, 4),
    "Administration": (2, 0),
    "Medical_Center": (4, 0.5),
    "Recreation_Center": (1, 5),
    "Shop": (1.8, 1.2),
    "Lift": (1.2, 2.8),
    "Accessible_Restroom": (2.5, 3.5)
}

# Use a set to count unique edges again (for verification)
unique_edges = set()
for node, neighbors in campus_graph_dict_extended.items():
    for neighbor, cost in neighbors.items():
        edge = tuple(sorted((node, neighbor)))
        unique_edges.add(edge)

print("Total unique path segments:", len(unique_edges))

# Create an undirected graph and assign coordinates (extended environment)
campus_map_ext = UndirectedGraph(campus_graph_dict_extended)
campus_map_ext.locations = campus_locations_extended

# -----------------------------------
# Visualization Setup: Mark key amenities with colors
# -----------------------------------
node_colors_ext = {}
for node in campus_locations_extended.keys():
    if "Restroom" in node:      # Wheelchair-accessible restroom
        node_colors_ext[node] = "red"
    elif node == "Parking":
        node_colors_ext[node] = "orange"
    elif node == "Shop":
        node_colors_ext[node] = "blue"
    elif node == "Lift":
        node_colors_ext[node] = "purple"
    else:
        node_colors_ext[node] = "white"

node_positions_ext = campus_map_ext.locations
node_label_pos_ext = { k: [v[0], v[1]-0.1] for k, v in node_positions_ext.items() }
edge_weights_ext = {(k, k2): v2 for k, v in campus_graph_dict_extended.items() for k2, v2 in v.items()}

campus_graph_data_ext = {
    'graph_dict': campus_graph_dict_extended,
    'node_colors': node_colors_ext,
    'node_positions': node_positions_ext,
    'node_label_positions': node_label_pos_ext,
    'edge_weights': edge_weights_ext
}

def show_map(graph_data, node_colors=None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights = graph_data['edge_weights']
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()},
            node_color=[node_colors[node] for node in G.nodes()],
            linewidths=1, edgecolors='k', with_labels=True, node_size=800)
    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=12)
    plt.title("Extended Campus Environment", fontsize=16)
    plt.tight_layout()
    plt.show()

# Display the extended environment
show_map(campus_graph_data_ext)

# -----------------------------------
# PART 5: Heuristic Comparison
# -----------------------------------
# Define extra environmental constraints
campus_constraints = {
    tuple(sorted(("Entrance", "Building_A"))): 0.1,
    tuple(sorted(("Entrance", "Parking"))): 0.2,
    tuple(sorted(("Building_A", "Library"))): 0.0,
    tuple(sorted(("Building_A", "Cafeteria"))): 0.1,
    tuple(sorted(("Building_A", "Administration"))): 0.2,
    tuple(sorted(("Library", "Auditorium"))): 0.1,
    tuple(sorted(("Library", "Accessible_Restroom"))): 0.0,
    tuple(sorted(("Cafeteria", "Auditorium"))): 0.1,
    tuple(sorted(("Auditorium", "Building_B"))): 0.1,
    tuple(sorted(("Auditorium", "Dormitory"))): 0.2,
    tuple(sorted(("Parking", "Building_B"))): 0.1,
    tuple(sorted(("Parking", "Accessible_Restroom"))): 0.0,
    tuple(sorted(("Building_B", "Gym"))): 0.2,
    tuple(sorted(("Building_B", "Medical_Center"))): 0.3,
    tuple(sorted(("Accessible_Restroom", "Building_C"))): 0.0,
    tuple(sorted(("Building_C", "Garden"))): 0.1,
    tuple(sorted(("Building_C", "Recreation_Center"))): 0.2,
    tuple(sorted(("Dormitory", "Administration"))): 0.1,
    tuple(sorted(("Administration", "Medical_Center"))): 0.1,
    tuple(sorted(("Medical_Center", "Recreation_Center"))): 0.1,
    tuple(sorted(("Building_A", "Building_B"))): 0.2,
    tuple(sorted(("Building_B", "Administration"))): 0.1,
    tuple(sorted(("Dormitory", "Medical_Center"))): 0.2,
    tuple(sorted(("Auditorium", "Recreation_Center"))): 0.3,
    # Extra constraints for new edges:
    tuple(sorted(("Entrance", "Lift"))): 0.2,
    tuple(sorted(("Building_A", "Shop"))): 0.1,
    tuple(sorted(("Shop", "Library"))): 0.1,
    tuple(sorted(("Shop", "Auditorium"))): 0.1,
    tuple(sorted(("Lift", "Accessible_Restroom"))): 0.2,
}

# Define the Extended Campus Problem class integrating environmental constraints
class CampusProblemExtended(GraphProblem):
    def path_cost(self, c, A, action, B):
        base_cost = self.graph.get(A, B) or self.infinity
        extra = campus_constraints.get(tuple(sorted((A, B))), 0)
        return c + base_cost + extra

    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if isinstance(node, str):
                d = self.distance(locs[node], locs[self.goal])
            else:
                d = self.distance(locs[node.state], locs[self.goal])
            return d + 0.3
        else:
            return self.infinity

# Define the Basic Campus Problem class using the original Euclidean distance heuristic
class CampusProblemBasic(GraphProblem):
    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if isinstance(node, str):
                return self.distance(locs[node], locs[self.goal])
            return self.distance(locs[node.state], locs[self.goal])
        else:
            return self.infinity

# Create instances and run A* for both approaches
campus_problem_ext = CampusProblemExtended("Entrance", "Recreation_Center", campus_map_ext)
iterations_ext, all_node_colors_ext, node_ext = astar_search(campus_problem_ext)
print("\n----- A* Search Results with Extended Heuristic (CampusProblemExtended) -----")
print("Path found:", node_ext.solution())
print("Total cost (including extra constraints):", node_ext.path_cost)
print("Iterations:", iterations_ext)

campus_problem_basic = CampusProblemBasic("Entrance", "Recreation_Center", campus_map_ext)
iterations_basic, all_node_colors_basic, node_basic = astar_search(campus_problem_basic)
print("\n----- A* Search Results with Basic Heuristic (CampusProblemBasic) -----")
print("Path found:", node_basic.solution())
print("Total cost:", node_basic.path_cost)
print("Iterations:", iterations_basic)


# In[5]:


def test_algorithms_performance(test_cases, a_star_algo, dijkstra_algo, problem_class, graph):
    results = []
    for start, goal in test_cases:
        # Run A* and measure time and number of iterations
        prob = problem_class(start, goal, graph)
        start_time = time.time()
        iterations_a, _, node_a = a_star_algo(prob)
        time_a = time.time() - start_time
        cost_a = node_a.path_cost
        path_a = node_a.solution()
        
        # Run Dijkstra (Uniform Cost Search) and measure time and iterations
        start_time = time.time()
        iterations_d, _, node_d = dijkstra_algo(prob)
        time_d = time.time() - start_time
        cost_d = node_d.path_cost
        path_d = node_d.solution()
        
        results.append((start, goal, path_a, cost_a, iterations_a, time_a, path_d, cost_d, iterations_d, time_d))
    return results

# Use the predefined test_cases (e.g., in the extended environment campus_map_ext)
results = test_algorithms_performance(test_cases, astar_search, uniform_cost_search_graph, GraphProblem, campus_map_ext)

for res in results:
    start, goal, path_a, cost_a, iter_a, t_a, path_d, cost_d, iter_d, t_d = res
    print(f"From {start} to {goal}:")
    print(f"  A*  -> Path: {path_a}, Cost: {cost_a}, Iterations: {iter_a}, Time: {t_a:.4f} sec")
    print(f"  Dijk-> Path: {path_d}, Cost: {cost_d}, Iterations: {iter_d}, Time: {t_d:.4f} sec")
    print("-"*50)


# In[8]:



