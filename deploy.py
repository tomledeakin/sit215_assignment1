import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

# Giả sử các giá trị cần thiết được import từ assignment1.py
from assignment1 import (
    campus_map_ext,
    campus_graph_data_ext,
    CampusProblemExtended,
    astar_search,
    uniform_cost_search_graph,
    final_path_colors,
    GraphProblem
)

st.title("Campus Wheelchair Navigation with Iteration Slider")

# --- Phần chọn tham số ---
st.sidebar.header("Input Parameters")
start = st.sidebar.selectbox("Select Start Point", sorted(campus_map_ext.locations.keys()))
goal = st.sidebar.selectbox("Select Goal Point", sorted(campus_map_ext.locations.keys()))
algo_choice = st.sidebar.selectbox("Select Algorithm", ["A* Search", "Dijkstra/Uniform Cost Search"])

# --- Nút Find Path ---
if st.sidebar.button("Find Path"):
    # Tạo problem
    prob = CampusProblemExtended(start, goal, campus_map_ext)

    # Chạy A* hoặc Dijkstra, chỉ chạy một lần
    start_time = time.time()
    if algo_choice == "A* Search":
        iterations, all_node_colors_list, node = astar_search(prob)
    else:
        iterations, all_node_colors_list, node = uniform_cost_search_graph(prob)
    elapsed_time = time.time() - start_time

    path = node.solution()
    cost = node.path_cost
    estimated_time = cost * 10  # ví dụ: 10 phút mỗi đơn vị cost

    # Lưu kết quả vào session_state để không tính lại mỗi khi slider thay đổi
    st.session_state['iterations'] = iterations
    st.session_state['all_node_colors_list'] = all_node_colors_list
    st.session_state['path'] = path
    st.session_state['cost'] = cost
    st.session_state['elapsed_time'] = elapsed_time
    st.session_state['start'] = start
    st.session_state['goal'] = goal
    st.session_state['algo'] = algo_choice

# --- Hiển thị kết quả + Slider (nếu đã có kết quả trong session_state) ---
if 'all_node_colors_list' in st.session_state:
    st.write("### Results")
    st.write(f"**Algorithm:** {st.session_state['algo']}")
    st.write(f"**Path from {st.session_state['start']} to {st.session_state['goal']}:** {st.session_state['path']}")
    st.write(f"**Total cost:** {st.session_state['cost']}")
    st.write(f"**Iterations:** {st.session_state['iterations']}")
    st.write(f"**Estimated traversal time:** {st.session_state['cost'] * 10:.1f} minutes")
    st.write(f"**Computation time:** {st.session_state['elapsed_time']:.4f} sec")

    # Tạo slider để chọn bước iteration
    iteration_index = st.slider(
        "Select Iteration",
        min_value=0,
        max_value=len(st.session_state['all_node_colors_list']) - 1,
        value=0,
        step=1
    )

    # Lấy màu node tại iteration đã chọn
    current_node_colors = st.session_state['all_node_colors_list'][iteration_index]

    # Vẽ đồ thị
    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(campus_graph_data_ext['graph_dict'])
    pos = campus_graph_data_ext['node_positions']
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color=[current_node_colors[node] for node in G.nodes()],
        edge_color="gray",
        node_size=800,
        ax=ax
    )
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=campus_graph_data_ext['edge_weights'],
        font_size=10,
        ax=ax
    )
    ax.set_title(f"Campus Map - Iteration {iteration_index}")
    st.pyplot(fig)
