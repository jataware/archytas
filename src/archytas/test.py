from archytas.react import ReActAgent, FailedTaskError
# from archytas.tools import calculator, ask_user, datetime, timestamp
from archytas.tools import ask_user, datetime, timestamp, fib_n, example_tool, calculator


from rich import traceback, print; traceback.install()
from easyrepl import REPL
history_file = 'chat_history.txt'


import pdb



def main():

    # tools = [calculator, ask_user, datetime, timestamp]
    tools = [ask_user, datetime, timestamp, fib_n, example_tool, calculator]

    agent = ReActAgent(tools=tools, verbose=True)

    for query in REPL(history_file=history_file):
        if not query: continue
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")




if __name__ == '__main__':
    main()