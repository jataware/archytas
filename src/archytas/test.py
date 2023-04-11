from agent import Agent
from prompt import prompt


import pdb



def calculator(expression:str) -> str:
    """
    A simple calculator tool. Can perform basic arithmetic

    Expressions must contain exactly:
    - one left operand. Can be a float or integer
    - one operation. Can be one of + - * / ^ %
    - one right operand. Can be a float or integer

    Expressions may not contain any spaces.

    Expressions may not contain parentheses, or multiple operations. 
    If you want to do a complex calculation, you must do it in multiple steps.


    examples:
    input: 22/7
    output: 3.142857142857143

    input: 3.24^2
    output: 10.4976
    
    input: 3.24+2.5
    output: 5.74
    """

    for op in '+-*/^%':
        if op in expression:
            break
    else:
        return "Invalid expression. No operation found"
    
    if op not in '+-*/^%':
        return f"Invalid operation. Expected one of '+-*/^%', got '{op}'"
    
    _a, _b = expression.split(op)
    try:
        a = float(_a)
    except ValueError:
        return f"Invalid expression. Could not convert \"{_a}\" to float"
    try:
        b = float(_b)
    except ValueError:
        return f"Invalid expression. Could not convert \"{_b}\" to float"
    
    if op == '+':
        return str(a+b)
    elif op == '-':
        return str(a-b)
    elif op == '*':
        return str(a*b)
    elif op == '/':
        return str(a/b)
    elif op == '^':
        return str(a**b)
    elif op == '%':
        return str(a%b)
    


"""
[notes]
- GPT-3.5 doesn't follow directions. It will add extra comment text to the beginning of its response.
    
    >>> can you tell me what my age squared is?
    Sure! To calculate your age squared, I will use the calculator tool. Here's what I'm thinking:

    {
    "thought": "I need to use the calculator to find your age squared.",
    "tool": "calculator",
    "tool_input": "age^2"
    }

    Please replace "age" with your actual age.

"""

def main():
    from easyrepl import REPL

    agent = Agent(system=prompt, model='gpt-4', role='ReAct1')

    # pdb.set_trace()

    for query in REPL():
        response = agent(query)
        print(response)
        pdb.set_trace()
        1

if __name__ == '__main__':
    main()