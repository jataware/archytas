from archytas.auth import add_openai_auth; add_openai_auth() # handle better

from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import ask_user, datetime, timestamp, fib_n, example_tool, calculator

from rich import traceback, print; traceback.install()
from easyrepl import REPL

import pdb


def start_repl():

    # make a list of the tools to use
    # tools = [calculator, ask_user, datetime, timestamp]
    tools = [ask_user, datetime, timestamp, fib_n, example_tool, calculator]

    # create the agent
    agent = ReActAgent(tools=tools, verbose=True)

    # print the agent's prompt
    # print(agent.prompt)

    # run the REPL
    for query in REPL(history_file='chat_history.txt'):
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")




if __name__ == '__main__':
    start_repl()