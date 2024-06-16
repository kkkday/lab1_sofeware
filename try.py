import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
graph = nx.DiGraph()

# 添加节点和边
graph.add_edges_from({('A', 'B'): 1, ('B', 'C'): 2, ('C', 'D'): 3, ('D', 'A'): 4})

# 绘制有向图
pos = nx.spring_layout(graph)  # 指定节点布局算法
nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', node_size=500, arrowstyle='->', arrowsize=15)
plt.axis('off')  # 关闭坐标轴
plt.show()