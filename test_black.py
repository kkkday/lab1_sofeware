# test_black.py
import pytest
from main import calcShortestPath, graph_make, filter_illegal_characters, make_dict,new_list,size


@pytest.mark.parametrize("start, end, expected", [
    # 存在路径
    ("today", "happy", ([45, 48, 47, 18],3.0)),
    # 起始顶点不存在
    ("computer", "happy", ([], -1.0)),
    # 结束顶点不存在
    ("today", "computer", ([], -2.0)),
    # 无路径情况
    ("new", "today", (None, float('inf'))),
    # 相同顶点
    ("some", "some", ([36], 0)),
    # 边界值测试，第一个顶点和最后一个
    ("today", "go", ([45, 48, 40, 3, 17], 4.0)),
    # 特殊字符测试，假设文本中包含这些字符
    ("$", "homework", ([], -1.0)),
    ("" ,"" , (None,-float('inf'))),
    ("","homework",(None,float(-3)))
])
def test_calcShortestPath(start, end, expected):
     # 模拟calcShortestPath函数的调用
    path, distance = calcShortestPath(start, end)

    # 根据期望结果进行断言
    assert (path, distance) == expected