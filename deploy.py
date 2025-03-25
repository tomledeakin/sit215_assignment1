import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

# Import các giá trị cần thiết từ file assignment.py
from assignment1 import (
    campus_map_ext,
    campus_graph_data_ext,
    CampusProblemExtended,
    astar_search,
    uniform_cost_search_graph,
    final_path_colors,
    GraphProblem
)

st.title("Campus Wheelchair Navigation")

# Sidebar inputs
st.sidebar.header("Input Parameters")
start = st.sidebar.selectbox("Select Start Point", sorted(campus_map_ext.locations.keys()))
goal = st.sidebar.selectbox("Select Goal Point", sorted(campus_map_ext.locations.keys()))
algo = st.sidebar.selectbox("Select Algorithm", ["A* Search", "Dijkstra/Uniform Cost Search"])

if st.sidebar.button("Find Path"):
    start_time = time.time()
    # Sử dụng lớp CampusProblemExtended để tích hợp constraints
    prob = CampusProblemExtended(start, goal, campus_map_ext)

    if algo == "A* Search":
        iterations, node_colors_list, node = astar_search(prob)
    else:
        iterations, node_colors_list, node = uniform_cost_search_graph(prob)

    elapsed_time = time.time() - start_time
    path = node.solution()
    cost = node.path_cost
    estimated_time = cost * 10  # Ước tính thời gian (ví dụ: 10 phút mỗi đơn vị chi phí)

    st.write("### Results")
    st.write(f"**Path from {start} to {goal}:** {path}")
    st.write(f"**Total cost:** {cost}")
    st.write(f"**Iterations:** {iterations}")
    st.write(f"**Estimated traversal time:** {estimated_time:.1f} minutes")
    st.write(f"**Computation time:** {elapsed_time:.4f} sec")

    st.write("#### Key landmarks in the environment:")
    for landmark in ["Accessible_Restroom", "Parking", "Shop", "Lift"]:
        if landmark in campus_map_ext.locations:
            st.write(f"- **{landmark}** at {campus_map_ext.locations[landmark]}")

    # Highlight final path on map using final_path_colors function
    final_colors = final_path_colors(campus_graph_data_ext['node_colors'], prob, path)

    # Vẽ bản đồ bằng networkx và matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(campus_graph_data_ext['graph_dict'])
    pos = campus_graph_data_ext['node_positions']
    nx.draw(G, pos=pos, with_labels=True,
            node_color=[final_colors[node] for node in G.nodes()],
            edge_color="gray", node_size=800, ax=ax)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=campus_graph_data_ext['edge_weights'], font_size=10, ax=ax)
    ax.set_title("Campus Map with Highlighted Path")
    st.pyplot(fig)
