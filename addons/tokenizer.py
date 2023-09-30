import re

"""
simple tokenizer with vectorization being the indices of the string array.
"""

def _simple_findall(inp):
    tokens = re.findall(r'\w+', inp)
    return tokens

def tokenizer(Input_string):
    tokens = _simple_findall(Input_string)
    vectorized_tokens = [i for i in range(len(tokens))]
    return vectorized_tokens