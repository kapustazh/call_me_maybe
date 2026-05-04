import math

def fn_add_numbers(a: float, b: float) -> float:
    assert isinstance(a, float)
    assert isinstance(b, float)
    return a + b

def fn_greet(name: str) -> str:
    assert isinstance(name, str)
    return f"Hello, {name}!"

def fn_multiply_numbers(a: float, b: float) -> float:
    assert isinstance(a, float)
    assert isinstance(b, float)
    return a * b

def fn_is_even(n: int) -> bool:
    assert isinstance(n, int)
    return n % 2 == 0

def fn_substitute_string_with_regex(source_string: str, regex: str, replacement: str) -> str:
    import re
    assert isinstance(source_string, str)
    assert isinstance(regex, str)
    assert isinstance(replacement, str)
    return re.sub(regex, replacement, source_string)

def fn_get_square_root(a: float) -> float:
    assert isinstance(a, float)
    return math.sqrt(a)

def fn_reverse_string(s: str) -> str:
    assert isinstance(s, str)
    return s[::-1]


exercises = {
    fn_get_square_root: [
        {
            "prompt": "What is the square root of 16?",
            "fn_args": {"a": 16.},
        },
    ],
    fn_reverse_string: [
        {
            "prompt": "Reverse the string 'hello'",
            "fn_args": {"s": "hello"},
        },
        {
            "prompt": "Reverse the string 'world'",
            "fn_args": {"s": "world"},
        },
    ],
    fn_substitute_string_with_regex: [
        {
            "prompt": "Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'",
            "fn_args": {
                "source_string": "Hello 34 I'm 233 years old",
                "regex": "\\d+",
                "replacement": "NUMBERS",
            },
        },
        {
            "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
            "fn_args": {
                "source_string": "Programming is fun",
                "regex": "[aeiouAEIOU]",
                "replacement": "*",
            },
        },
        # {
        #     "prompt": "Replace multiple consecutive spaces in 'This   has    too     many spaces' with single spaces",
        #     "fn_args": {
        #         "source_string": "This   has    too     many spaces",
        #         "regex": "\\s+",
        #         "replacement": " ",
        #     },
        # },
        {
            "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
            "fn_args": {
                "source_string": "The cat sat on the mat with another cat",
                "regex": "\\bcat\\b",
                "replacement": "dog",
            },
        },
    ],
    fn_add_numbers: [
        {
            "prompt": "What is the sum of 2 and 3?",
            "fn_args": {"a": 2., "b": 3.},
        },
        {
            "prompt": "What is the sum of 265 and 345?",
            "fn_args": {"a": 265., "b": 345.},
        },
    ],
    fn_is_even: [
        {
            "prompt": "Is 4 an even number?",
            "fn_args": {"n": 4},
        },
        {
            "prompt": "Is 7 an even number?",
            "fn_args": {"n": 7},
        },
    ],
    fn_multiply_numbers: [
        {
            "prompt": "What is the product of 3 and 5?",
            "fn_args": {"a": 3., "b": 5.},
        },
        {
            "prompt": "What is the product of 12 and 4?",
            "fn_args": {"a": 12., "b": 4.},
        },
    ],
    fn_greet: [
        {
            "prompt": "Greet shrek",
            "fn_args": {"name": "shrek"},
        },
        {
            "prompt": "Greet john",
            "fn_args": {"name": "john"},
        },
    ],
}