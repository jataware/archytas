# from agent import Agent
# from prompt import prompt
import json
from rich import traceback, print; traceback.install()
from archytas.react import ReAct, FailedTaskError
from archytas.tools import calculator, ask_user, datetime_tool, timestamp

from easyrepl import REPL
history_file = 'chat_history.txt'


import pdb



def main():

    # agent = Agent(prompt=prompt, model='gpt-4')
    tools = {
        'calculator': calculator,
        'ask_user': ask_user,
        'datetime': datetime_tool,
        'timestamp': timestamp,
    }

    agent = ReAct(tools=tools, verbose=True)

    for query in REPL(history_file=history_file):
        if not query: continue
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")




if __name__ == '__main__':
    main()