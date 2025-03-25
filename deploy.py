import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

# Import necessary values from your assignment module
from assignment1 import (
    campus_map_ext,
    campus_graph_data_ext,
    CampusProblemExtended,
    astar_search,
    uniform_cost_search_graph,
    final_path_colors,
    GraphProblem
)

st.title("Campus Wheelchair Navigation with Iteration Visualization")

# Sidebar inputs
st.sidebar.header("Input Parameters")
start = st.sidebar.selectbox("Select Start Point", sorted(campus_map_ext.locations.keys()))
goal = st.sidebar.selectbox("Select Goal Point", sorted(campus_map_ext.locations.keys()))
algo = st.sidebar.selectbox("Select Algorithm", ["A* Search", "Dijkstra/Uniform Cost Search"])

if st.sidebar.button("Find Path"):
    start_time = time.time()
    # Use the extended problem to incorporate constraints
    prob = CampusProblemExtended(start, goal, campus_map_ext)

    if algo == "A* Search":
        iterations, all_node_colors_list, node = astar_search(prob)
    else:
        iterations, all_node_colors_list, node = uniform_cost_search_graph(prob)

    elapsed_time = time.time() - start_time
    path = node.solution()
    cost = node.path_cost
    estimated_time = cost * 10  # Example: 10 minutes per cost unit

    st.write("### Results")
    st.write(f"**Path from {start} to {goal}:** {path}")
    st.write(f"**Total cost:** {cost}")
    st.write(f"**Iterations:** {iterations}")
    st.write(f"**Estimated traversal time:** {estimated_time:.1f} minutes")
    st.write(f"**Computation time:** {elapsed_time:.4f} sec")

    st.write("#### Key Landmarks in the Environment:")
    for landmark in ["Accessible_Restroom", "Parking", "Shop", "Lift"]:
        if landmark in campus_map_ext.locations:
            st.write(f"- **{landmark}** at {campus_map_ext.locations[landmark]}")

    # Add a slider to visualize each iteration step
    iteration_index = st.slider("Select Iteration", min_value=0, max_value=len(all_node_colors_list) - 1, value=0,
                                step=1)

    # Get node colors for the current iteration
    current_node_colors = all_node_colors_list[iteration_index]

    # Plot the graph for the current iteration
    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(campus_graph_data_ext['graph_dict'])
    pos = campus_graph_data_ext['node_positions']
    nx.draw(G, pos=pos, with_labels=True,
            node_color=[current_node_colors[node] for node in G.nodes()],
            edge_color="gray", node_size=800, ax=ax)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=campus_graph_data_ext['edge_weights'],
                                 font_size=10, ax=ax)
    ax.set_title(f"Campus Map at Iteration {iteration_index}")
    st.pyplot(fig)
