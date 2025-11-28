import pytest
from src import string_utils
def test_fun1():
    assert string_utils.fun1("hello") == "HELLO"
    assert string_utils.fun1("World") == "WORLD"
    assert string_utils.fun1("") == ""
    assert string_utils.fun1("123abc") == "123ABC"

def test_fun1_invalid_input():
    with pytest.raises(ValueError):
        string_utils.fun1(123)
    with pytest.raises(ValueError):
        string_utils.fun1(None)

def test_fun2():
    assert string_utils.fun2("hello") == "olleh"
    assert string_utils.fun2("world") == "dlrow"
    assert string_utils.fun2("") == ""
    assert string_utils.fun2("a") == "a"
    assert string_utils.fun2("racecar") == "racecar"

def test_fun2_invalid_input():
    with pytest.raises(ValueError):
        string_utils.fun2(123)
    with pytest.raises(ValueError):
        string_utils.fun2([1, 2, 3])

def test_fun3():
    assert string_utils.fun3("hello", "l") == 2
    assert string_utils.fun3("hello", "o") == 1
    assert string_utils.fun3("hello", "x") == 0
    assert string_utils.fun3("", "a") == 0
    assert string_utils.fun3("aaaaaa", "a") == 6

def test_fun3_invalid_input():
    with pytest.raises(ValueError):
        string_utils.fun3("hello", 123)
    with pytest.raises(ValueError):
        string_utils.fun3("hello", "ab")  # Multiple characters
    with pytest.raises(ValueError):
        string_utils.fun3(123, "a")

def test_fun4():
    assert string_utils.fun4("hello", "world") == "hello world"
    assert string_utils.fun4("hello", "world", "-") == "hello-world"
    assert string_utils.fun4("hello", "world", "") == "helloworld"
    assert string_utils.fun4("", "", "-") == "-"
    assert string_utils.fun4("test", "123", "_") == "test_123"

def test_fun4_invalid_input():
    with pytest.raises(ValueError):
        string_utils.fun4(123, "world")
    with pytest.raises(ValueError):
        string_utils.fun4("hello", 456)
    with pytest.raises(ValueError):
        string_utils.fun4("hello", "world", 789)