import pytest
from main import queryBridgeWords,graph_make, new_list, mat, size
def test_queryBridgeWords():
    # 测试已知的桥接词
    start_word = 'park'
    end_word = 'my'
    expected_path = [new_list.index("with")]
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path, "测试失败：未找到预期的桥接词路径"

    # 测试没有桥接词的情况
    start_word = 'we'
    end_word = 'for'
    expected_path = []
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path, "测试失败：在不需要桥接词的情况下找到了路径"

    # 测试输入词不在图中的情况
    start_word = 'computer'
    end_word = 'our'
    expected_path = None
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path ,"测试失败：未返回None"

    start_word = 'shop'
    end_word = 'computer'
    expected_path = None
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path, "测试失败：未返回None"

    start_word = 'computer'
    end_word = 'playground'
    expected_path = None
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path, "测试失败：未返回None"

    start_word = 'a'
    end_word = 'with'
    expected_path = [new_list.index("park"),new_list.index("shop")]
    result = queryBridgeWords(start_word, end_word)
    assert result == expected_path, "测试失败：未返回所有桥接词"
