def fun1(text):
    """
    Converts text to uppercase.
    Args:
        text (str): Input string.
    Returns:
        str: Uppercase version of the input text.
    Raises:
        ValueError: If text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    return text.upper()

def fun2(text):
    """
    Reverses a string.
    Args:
        text (str): Input string.
    Returns:
        str: Reversed version of the input text.
    Raises:
        ValueError: If text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text[::-1]

def fun3(text, char):
    """
    Counts occurrences of a character in text.
    Args:
        text (str): Input string.
        char (str): Character to count.
    Returns:
        int: Number of times char appears in text.
    Raises:
        ValueError: If inputs are not strings or char is not a single character.
    """
    if not isinstance(text, str) or not isinstance(char, str):
        raise ValueError("Both inputs must be strings.")
    if len(char) != 1:
        raise ValueError("Second argument must be a single character.")
    return text.count(char)

def fun4(text1, text2, separator=" "):
    """
    Concatenates two strings with a separator.
    Args:
        text1 (str): First string.
        text2 (str): Second string.
        separator (str): String to place between text1 and text2 (default is space).
    Returns:
        str: Concatenated string with separator.
    Raises:
        ValueError: If any input is not a string.
    """
    if not all(isinstance(x, str) for x in [text1, text2, separator]):
        raise ValueError("All inputs must be strings.")
    
    result = text1 + separator + text2
    return result


# f1_op = fun1("hello")
# f2_op = fun2("world")
# f3_op = fun3("hello", "l")
# f4_op = fun4(f1_op, f2_op, "-")