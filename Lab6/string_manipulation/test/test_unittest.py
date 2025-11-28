import sys
import os
import unittest
from src import string_utils

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)




class TestStringUtils(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(string_utils.fun1("hello"), "HELLO")
        self.assertEqual(string_utils.fun1("World"), "WORLD")
        self.assertEqual(string_utils.fun1(""), "")
        self.assertEqual(string_utils.fun1("123abc"), "123ABC")

    def test_fun1_invalid_input(self):
        with self.assertRaises(ValueError):
            string_utils.fun1(123)
        with self.assertRaises(ValueError):
            string_utils.fun1(None)

    def test_fun2(self):
        self.assertEqual(string_utils.fun2("hello"), "olleh")
        self.assertEqual(string_utils.fun2("world"), "dlrow")
        self.assertEqual(string_utils.fun2(""), "")
        self.assertEqual(string_utils.fun2("a"), "a")
        self.assertEqual(string_utils.fun2("racecar"), "racecar")

    def test_fun2_invalid_input(self):
        with self.assertRaises(ValueError):
            string_utils.fun2(123)
        with self.assertRaises(ValueError):
            string_utils.fun2([1, 2, 3])

    def test_fun3(self):
        self.assertEqual(string_utils.fun3("hello", "l"), 2)
        self.assertEqual(string_utils.fun3("hello", "o"), 1)
        self.assertEqual(string_utils.fun3("hello", "x"), 0)
        self.assertEqual(string_utils.fun3("", "a"), 0)
        self.assertEqual(string_utils.fun3("aaaaaa", "a"), 6)

    def test_fun3_invalid_input(self):
        with self.assertRaises(ValueError):
            string_utils.fun3("hello", 123)
        with self.assertRaises(ValueError):
            string_utils.fun3("hello", "ab")  # Multiple characters
        with self.assertRaises(ValueError):
            string_utils.fun3(123, "a")

    def test_fun4(self):
        self.assertEqual(string_utils.fun4("hello", "world"), "hello world")
        self.assertEqual(string_utils.fun4("hello", "world", "-"), "hello-world")
        self.assertEqual(string_utils.fun4("hello", "world", ""), "helloworld")
        self.assertEqual(string_utils.fun4("", "", "-"), "-")
        self.assertEqual(string_utils.fun4("test", "123", "_"), "test_123")

    def test_fun4_invalid_input(self):
        with self.assertRaises(ValueError):
            string_utils.fun4(123, "world")
        with self.assertRaises(ValueError):
            string_utils.fun4("hello", 456)
        with self.assertRaises(ValueError):
            string_utils.fun4("hello", "world", 789)


if __name__ == '__main__':
    unittest.main()