import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

# Import necessary objects from your assignment module
from assignment1 import (
    campus_map_ext,
    campus_graph_data_ext,
    CampusProblemExtended,
    astar_search,
    uniform_cost_search_graph,
    final_path_colors,
    GraphProblem
)

st.title("Campus Wheelchair Navigation with Iterative Visualization")

# Sidebar inputs for start, goal, and algorithm selection
st.sidebar.header("Input Parameters")
start_point = st.sidebar.selectbox("Select Start Point", sorted(campus_map_ext.locations.keys()))
goal_point = st.sidebar.selectbox("Select Goal Point", sorted(campus_map_ext.locations.keys()))
algo_choice = st.sidebar.selectbox("Select Algorithm", ["A* Search", "Dijkstra/Uniform Cost Search"])

# When user clicks the "Find Path" button, run the algorithm once and store iterations in session_state.
if st.sidebar.button("Find Path"):
    prob = CampusProblemExtended(start_point, goal_point, campus_map_ext)
    if algo_choice == "A* Search":
        iterations, all_node_colors, node = astar_search(prob)
    else:
        iterations, all_node_colors, node = uniform_cost_search_graph(prob)

    # Store results in session_state to avoid re-running on slider change.
    st.session_state['iterations'] = iterations
    st.session_state['all_node_colors'] = all_node_colors
    st.session_state['final_path'] = node.solution()
    st.session_state['final_cost'] = node.path_cost
    st.session_state['elapsed_time'] = time.time()  # Optionally store the timestamp
    st.session_state['problem'] = prob

# If we have results stored, display the results and provide a slider for iterative visualization.
if 'all_node_colors' in st.session_state:
    st.write("### Results")
    st.write(f"**Path from {start_point} to {goal_point}:** {st.session_state['final_path']}")
    st.write(f"**Total cost:** {st.session_state['final_cost']}")
    st.write(f"**Iterations:** {st.session_state['iterations']}")
    # For example, assume each cost unit equals 10 minutes:
    estimated_time = st.session_state['final_cost'] * 10
    st.write(f"**Estimated traversal time:** {estimated_time:.1f} minutes")

    # Create a slider for selecting the iteration index
    iteration_index = st.slider("Select Iteration",
                                min_value=0,
                                max_value=len(st.session_state['all_node_colors']) - 1,
                                value=0, step=1)

    # Retrieve the node colors for the current iteration
    current_node_colors = st.session_state['all_node_colors'][iteration_index]

    # Plot the graph for the current iteration
    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(campus_graph_data_ext['graph_dict'])
    pos = campus_graph_data_ext['node_positions']
    nx.draw(G, pos=pos, with_labels=True,
            node_color=[current_node_colors[node] for node in G.nodes()],
            edge_color="gray", node_size=800, ax=ax)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=campus_graph_data_ext['edge_weights'], font_size=10, ax=ax)
    ax.set_title(f"Campus Map at Iteration {iteration_index}")
    st.pyplot(fig)
