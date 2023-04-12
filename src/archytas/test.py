from agent import Agent
from prompt import prompt
import json

from easyrepl import REPL, readl
history_file = 'chat_history.txt'

from rich import print

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
        raise ValueError(f"Invalid operation. Expected one of '+-*/^%', got '{op}'")
    
    _a, _b = expression.split(op)
    a = float(_a)
    b = float(_b)
    # try:
    # except ValueError:
    #     return f"Invalid expression. Could not convert \"{_a}\" to float"
    # try:
    # except ValueError:
    #     return f"Invalid expression. Could not convert \"{_b}\" to float"
    
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
    
def extract_action(action:dict) -> tuple[str, str, str]:
    """Verify that action has the correct keys. Otherwise, raise an error"""
    assert isinstance(action, dict), f"Action must be a json dictionary, got {type(action)}"
    assert len(action) == 3, f"Action must have exactly 3 keys, got {len(action)}"
    assert 'thought' in action, "Action is missing key 'thought'"
    assert 'tool' in action, "Action is missing key 'tool'"
    assert 'tool_input' in action, "Action is missing key 'tool_input'"

    thought = action['thought']
    tool = action['tool']
    tool_input = action['tool_input']

    return thought, tool, tool_input

class FailedTaskError(Exception): ...

def react(agent:Agent, query:str, tools:dict, max_errors:int=3, verbose:bool=True) -> str:

    # error handling. If too many errors, break the ReAct loop. Otherwise tell the agent, and continue
    errors = 0
    def err(mesg) -> str:
        nonlocal errors, max_errors, agent, verbose

        errors += 1
        if errors > max_errors:
            raise FailedTaskError(f"Too many errors during task. Last error: {mesg}")
        if verbose:
            print(f"[red]error: {mesg}[/red]")

        return agent.error(mesg)
    
    # run the initial user query
    action_str = agent.query(query)

    # ReAct loop
    while True:

        # Convert agent output to json
        try:
            action = json.loads(action_str)
        except json.JSONDecodeError:
            action_str = err(f"failed to parse action: ```{action_str}```. Action must be a valid json dictionary with keys 'thought', 'tool', and 'tool_input'.")
            continue

        # verify that action has the correct keys
        try:
            thought, tool, tool_input = extract_action(action)
        except AssertionError as e:
            action_str = err(str(e))
            continue

        # print action
        if verbose:
            print(f"thought: {thought}\ntool: {tool}\ntool_input: {tool_input}\n")

        # exit ReAct loop if agent says final_answer or fail_task
        if tool == 'final_answer':
            return tool_input
        if tool == 'fail_task':
            raise FailedTaskError(tool_input)
        
        # run tool
        try:
            tool = tools[tool]
        except KeyError:
            action_str = err(f"unknown tool \"{tool}\"")
            continue

        try:
            tool_output = tool(tool_input)
        except Exception as e:
            action_str = err(f"error running tool \"{tool}\": {e}")
            continue

        # have the agent observe the result, and get the next action
        if verbose:
            print(f"observation: {tool_output}\n")
        action_str = agent.observe(tool_output)



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

    agent = Agent(prompt=prompt, model='gpt-4')

    for query in REPL(history_file=history_file):
        if not query: continue
        try:
            answer = react(agent, query, tools={'calculator': calculator}, verbose=True)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")


if __name__ == '__main__':
    main()