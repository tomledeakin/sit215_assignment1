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

st.title("Campus Wheelchair Navigation with Iteration Logic (Streamlit)")

# --- Phần chọn tham số ---
st.sidebar.header("Input Parameters")
start = st.sidebar.selectbox("Select Start Point", sorted(campus_map_ext.locations.keys()))
goal = st.sidebar.selectbox("Select Goal Point", sorted(campus_map_ext.locations.keys()))
algo_choice = st.sidebar.selectbox("Select Algorithm", ["A* Search", "Dijkstra/Uniform Cost Search"])

if 'iteration_index' not in st.session_state:
    st.session_state['iteration_index'] = 0
if 'all_node_colors_list' not in st.session_state:
    st.session_state['all_node_colors_list'] = []
if 'max_iteration' not in st.session_state:
    st.session_state['max_iteration'] = 0
if 'result_ready' not in st.session_state:
    st.session_state['result_ready'] = False

# --- Nút "Find Path" ---
if st.sidebar.button("Find Path"):
    prob = CampusProblemExtended(start, goal, campus_map_ext)

    start_time = time.time()
    if algo_choice == "A* Search":
        iterations, all_node_colors_list, node = astar_search(prob)
    else:
        iterations, all_node_colors_list, node = uniform_cost_search_graph(prob)
    elapsed_time = time.time() - start_time

    st.session_state['iteration_index'] = 0
    st.session_state['all_node_colors_list'] = all_node_colors_list
    st.session_state['max_iteration'] = len(all_node_colors_list) - 1
    st.session_state['result_ready'] = True

    st.session_state['final_path'] = node.solution()
    st.session_state['final_cost'] = node.path_cost
    st.session_state['elapsed_time'] = elapsed_time
    st.session_state['iterations'] = iterations
    st.session_state['start'] = start
    st.session_state['goal'] = goal
    st.session_state['algo_choice'] = algo_choice


# Hàm vẽ đồ thị tại iteration_index
def draw_iteration(iter_idx):
    current_node_colors = st.session_state['all_node_colors_list'][iter_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    G = nx.Graph(campus_graph_data_ext['graph_dict'])
    pos = campus_graph_data_ext['node_positions']

    nx.draw(
        G, pos=pos, with_labels=True,
        node_color=[current_node_colors[node] for node in G.nodes()],
        edge_color="green", node_size=800, ax=ax
    )
    nx.draw_networkx_edge_labels(
        G, pos=pos,
        edge_labels=campus_graph_data_ext['edge_weights'],
        font_size=10, ax=ax
    )
    ax.set_title(f"Iteration {iter_idx}")
    st.pyplot(fig)


# --- Hiển thị kết quả nếu đã có all_node_colors_list ---
if st.session_state['result_ready']:
    st.write("### Search Results")
    st.write(f"**Algorithm:** {st.session_state['algo_choice']}")
    st.write(
        f"**Path from {st.session_state['start']} to {st.session_state['goal']}:** {st.session_state['final_path']}")
    st.write(f"**Total cost:** {st.session_state['final_cost']}")
    st.write(f"**Iterations:** {st.session_state['iterations']}")
    # ví dụ: 10 phút mỗi đơn vị cost
    estimated_time = st.session_state['final_cost'] * 10
    st.write(f"**Estimated traversal time:** {estimated_time:.1f} minutes")
    st.write(f"**Computation time:** {st.session_state['elapsed_time']:.4f} sec")

    # Slider iteration
    iteration_index = st.slider(
        "Select Iteration",
        min_value=0,
        max_value=st.session_state['max_iteration'],
        value=st.session_state['iteration_index'],
        step=1
    )

    # Cập nhật iteration_index trong session_state mỗi lần slider thay đổi
    if iteration_index != st.session_state['iteration_index']:
        st.session_state['iteration_index'] = iteration_index

    # Nút "Play" để tự động lặp qua các iteration
    if st.button("Play"):
        # Mô phỏng logic visualize_callback(visualize=True)
        for i in range(st.session_state['iteration_index'], st.session_state['max_iteration'] + 1):
            st.session_state['iteration_index'] = i
            # rerun để hiển thị iteration i
            time.sleep(0.5)
            st.experimental_rerun()

    # Vẽ đồ thị cho iteration hiện tại
    draw_iteration(st.session_state['iteration_index'])
