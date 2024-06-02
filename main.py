import argparse
import re
import numpy as np
import random
import heapq
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
def queryBridgeWords(start,end):
    s_location = new_list.index(start)if start in new_list else -1
    e_location = new_list.index(end)if end in new_list else -1
    if (s_location==-1):
        print("No "+start+"in the graph!")
    elif (e_location==-1):
        print("No " + end + "in the graph!")
    else:
        paths = findbridge(s_location, e_location)
        result = []
        if (paths == []):
            print("No bridge words from \"" + str(start) + "\" to \"" + str(end) + "\"!")
        else:
            print("The bridge words from " + start + " to " + end + " are: ",end="")
            if(len(paths)!=1):
                for it in paths[:-1]:
                    print(str(new_list[it]) + ",", end="")
                print(" and " + str(new_list[paths[-1]]))
            else:
                print(str(new_list[paths[-1]]))
    return paths
def findbridge(start,end):
    word = []
    for i in range(0,size):
        each = mat[start][i]
        if (each != 0):
            if (mat[i][end]!=0):
                word.append(i)
    return word
def generateNewText(sentence):
    dict,text_list,new_size=make_dict(sentence)
    result = filter_illegal_characters(sentence).lower().split()
    num=0
    for key,value in dict.items():
        one=key[0]
        two=key[1]
        if(one in new_list)and(two in new_list):
            s_location = new_list.index(one)
            e_location = new_list.index(two)
            word_b=findbridge(s_location, e_location)
            if (word_b==[])and (num==new_size-1):
               print(sentence)
               break
            elif (word_b!=[]):
                new_word = new_list[random.choice(word_b)]
                flag = result.index(one) if one in result else -1
                while (flag != -1):
                    if result[flag + 1] != two:
                        break
                    else:
                        result.insert(flag + 1, str(new_word))
                    flag = result[flag + 1:].index(one) if one in result[flag + 1:] else -1

        num=num+1
    print(" ".join(result))

def filter_illegal_characters(text):
    pattern = r'[^\w\s]'  # 匹配非字母数字和空白字符
    filtered_text = re.sub(pattern, '', text)  # 替换非法字符为''
    return filtered_text

def make_dict(sentence):
    filtered_text = filter_illegal_characters(sentence).lower()
    wordlist = filtered_text.split()
    size = len(set(wordlist))
    new_list = list(set(wordlist))
    new_list.sort()
    # print(mat)
    # print(new_list)
    dict = {}
    n = len(wordlist)
    for i in range(0, n):
        if (i < n - 1):
            if (wordlist[i], wordlist[i + 1]) in dict:
                dict[(wordlist[i], wordlist[i + 1])] += 1
            else:
                dict[(wordlist[i], wordlist[i + 1])] = 1
    return dict,new_list,size
def graph_make(sentence):
    dict,new_list,size= make_dict(sentence)
    # print(dict)
    mat = np.zeros((size, size))
    for key, value in dict.items():
        row = new_list.index(key[0])
        colum = new_list.index(key[1])
        mat[row][colum] = value

    return new_list,mat,size,dict


def calcShortestPath(start, end):
    start = new_list.index(start)if start in new_list else -1
    end = new_list.index(end)if end in new_list else -1
    # 获取顶点的数量
    # 初始化距离列表，设置为无穷大，除了起始点到自身的距离为0
    if (start==-1):
        return [],float(-1)
    if (end==-1):
        return [],float(-2)
    distances = [float('inf')] * size
    distances[start] = 0

    # 初始化前驱节点列表，用于重建最短路径
    predecessors = [-1] * size  # 使用-1表示没有前驱节点

    # 初始化优先队列，用于选择下一个顶点
    priority_queue = [(0, start)]

    # 当队列不为空时继续执行
    while priority_queue:
        # 弹出当前距离最小的顶点
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 对当前顶点的出边进行遍历
        for neighbor, weight in enumerate(mat[current_vertex]):
            if weight > 0:  # 只有当存在边时才考虑
                # 计算通过当前顶点到达邻居的距离
                distance_through_u = current_distance + weight
                # 如果通过当前顶点到达邻居的距离更短，则更新
                if distance_through_u < distances[neighbor]:
                    distances[neighbor] = distance_through_u
                    predecessors[neighbor] = current_vertex
                    # 将邻居顶点加入优先队列
                    heapq.heappush(priority_queue, (distance_through_u, neighbor))

    # 检查是否找到了结束点的路径
    if distances[end] == float('inf'):
        return None, float('inf')  # 如果没有路径，则返回None和无穷大距离

    # 重建最短路径
    path = []
    current_vertex = end
    while current_vertex != -1:  # -1 表示没有前驱节点，即起始顶点
        path.append(current_vertex)
        current_vertex = predecessors[current_vertex]

    # 反转路径以获得从起始顶点到结束顶点的正确顺序
    path.reverse()
    return path, distances[end]
def randomWalk():
    # 获取所有节点
    all_nodes = set(sum([[k[0], k[1]] for k in dict.keys()], []))

    # 随机选择一个起始节点
    start_node = random.choice(list(all_nodes))

    # 初始化遍历状态
    visited_nodes = [start_node]
    visited_edges = []
    current_node = start_node

    while True:
        # 获取当前节点的出边
        out_edges = [(k[0], k[1]) for k in dict.keys() if k[0] == current_node]

        # 如果当前节点没有出边,遍历结束
        if not out_edges:
            break

        # 随机选择一条出边
        next_edge = random.choice(out_edges)
        next_node = next_edge[1]

        # 检查是否已经遍历过该边
        if (current_node, next_node) in visited_edges:
            break

        # 询问用户是否继续
        user_input = input(f"当前节点: {current_node}, 下一步节点: {next_node}。是否继续遍历? (y/n) ")
        if user_input.lower() != 'y':
            break

        # 更新遍历状态
        visited_nodes.append(next_node)
        visited_edges.append((current_node, next_node))
        current_node = next_node


    return visited_nodes, visited_edges
def showDirectedGraph(dict):
    # 创建有向图对象
    graph = nx.DiGraph()
    # 添加带有权值的有向边
    for edge, weight in dict.items():
        source, target = edge
        graph.add_edge(source, target, weight=weight)
    # 绘制图形
    pos = nx.fruchterman_reingold_layout(graph)  # 使用 fruchterman_reingold_layout 布局算法
    labels = nx.get_edge_attributes(graph, 'weight')

    fig, ax = plt.subplots(figsize=(size, size))  # 设置图形大小
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax)
    plt.axis('off')

    plt.savefig('directed_graph.png')  # 保存为图片文件
    plt.show()  # 显示图形

print("请输入文件路径：")
filepath=input()
wordlist = []
with open(filepath,'r',encoding='utf-8') as src:
    sentence = src.read()
new_list,mat,size,dict=graph_make(sentence)
while(True):
    print("请输入你的选项：1.展示生成的有向图。2.查询桥接词。3.根据桥接词生成新文本。4.计算两个词之间的最短路径。5.随机游走。6.输入新文件。7.退出")
    choice=input()
    if choice=='1':
        showDirectedGraph(dict);
    elif choice=='2':
        print("请输入第一个词：")
        start=input()
        print("请输入第二个词")
        end=input()
        name=queryBridgeWords(start,end)
    elif choice=='3':
        print("请输入你想查询的句子：")
        sentence=input()
        generateNewText(sentence)
    elif choice=='4':
        print("请输入第一个词：")
        start = input()
        print("请输入第二个词")
        end = input()
        if (end!=str()):
            path, distance = calcShortestPath(start, end)
            if distance == -1.0:
                print("找不到起始词：无最短路径。")
            elif distance == -2.0:
                print("找不到结束词：无最短路径。")
            elif distance == float('inf'):
                print("没有找到路径")
            else:
                lastpath = []
                for num in path:
                    lastpath.append(new_list[num])
                print("最短路径为：" + str(" ".join(lastpath)))
                print("路径长度为：" + str(distance))
        else:
            for word in new_list:
                if word!=start:
                    path, distance = calcShortestPath(start, word)
                    if distance == -1.0:
                        print("找不到起始词：无最短路径。")
                    elif distance == float('inf'):
                        print("从"+start+"到"+word+"没有找到路径")
                    else:
                        lastpath = []
                        for num in path:
                            lastpath.append(new_list[num])
                        print("从"+start+"到"+word+"最短路径为：" + str(" ".join(lastpath)))
                        print("路径长度为：" + str(distance))


    elif choice=='5':
        randomWalk()
    elif choice=='6':
        print("请输入文件路径：")
        filepath = input()
        wordlist = []
        with open(filepath, 'r',encoding="uft-8") as src:
            sentence = src.read()
        new_list, mat, size, dict = graph_make(sentence)
    elif choice=='7':
        break
    else:
        print("输入错误，请输入正确的数字选项：")








