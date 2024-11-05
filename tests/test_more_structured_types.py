from typing import Dict
from archytas.tool_utils import tool, get_tool_prompt_description
from .test_structured_types import A, B, C, D, M1, M2, M3, GenericModelA, GenericModelB

def get_tool00():
    @tool(devmode=True)
    def tool00(a0: A|int, a1: A|int) -> tuple[A|int, A|int]:
        """
        Example tool

        Args:
            a0 (A|int): A or int instance
            a1 (A|int): A or int instance

        Returns:
            tuple[A|int, A|int]: tuple of A or int instances
        """
        return a0, a1
    return tool00

def get_tool0():
    @tool(devmode=True)
    def tool0(l: list[A]) -> list[A]:
        """
        Example tool

        Args:
            l (list[A]): list of A instances

        Returns:
            list[A]: list of A instances
        """
        return l
    return tool0


def get_tool1():
    @tool(devmode=True)
    def tool1(t: tuple[A, int, None]) -> tuple[A, int, None]:
        """
        Example tool

        Args:
            t (tuple[A, int, None]): tuple of A, int, and None

        Returns:
            tuple[A, int, None]: tuple of A, int, and None
        """
        return t
    return tool1

def get_tool2():
    @tool(devmode=True)
    def tool2(t: tuple[A, ...]) -> tuple[A, ...]:
        """
        Example tool

        Args:
            t (tuple[A, ...]): tuple of A instances

        Returns:
            tuple[A, ...]: tuple of A instances
        """
        return t
    return tool2


def get_tool3():
    @tool(devmode=True)
    def tool3(item: A|B) -> A|B:
        """
        Example tool

        Args:
            item (A|B): A or B instance

        Returns:
            A|B: A or B instance
        """
        return item
    return tool3


def get_tool4():
    @tool(devmode=True)
    def tool4(l: list[A|B]) -> list[A|B]:
        """
        Example tool

        Args:
            l (list[A|B]): list of A or B instances

        Returns:
            list[A|B]: list of A or B instances
        """
        return l
    return tool4


def get_tool5():
    @tool(devmode=True)
    def tool5(t: tuple[A, B, M1|M2]) -> tuple[A, B, M1|M2]:
        """
        Example tool

        Args:
            t (tuple[A, B, M1|M2]): tuple of A, B, and M1 or M2

        Returns:
            tuple[A, B, M1|M2]: tuple of A, B, and M1 or M2
        """
        return t
    return tool5


def get_tool6():
    @tool(devmode=True)
    def tool6(d: dict[A, B]) -> dict[A, B]:
        """
        Example tool

        Args:
            d (dict[A, B]): dictionary of A and B instances

        Returns:
            dict[A, B]: dictionary of A and B instances
        """
        return d
    return tool6

def get_tool7():
    @tool(devmode=True)
    def tool7(d: Dict[M1, M2]) -> Dict[M1, M2]:
        """
        Example tool

        Args:
            d (Dict[M1, M2]): dictionary of M1 and M2 instances

        Returns:
            Dict[M1, M2]: dictionary of M1 and M2 instances
        """
        return d
    return tool7


def test_tool00():
    t = get_tool00()
    get_tool_prompt_description(t)

def test_tool0():
    t = get_tool0()
    get_tool_prompt_description(t)

def test_tool1():
    t = get_tool1()
    get_tool_prompt_description(t)

def test_tool2():
    t = get_tool2()
    get_tool_prompt_description(t)

def test_tool3():
    t = get_tool3()
    get_tool_prompt_description(t)

def test_tool4():
    t = get_tool4()
    get_tool_prompt_description(t)

def test_tool5():
    t = get_tool5()
    get_tool_prompt_description(t)

def test_tool6():
    t = get_tool6()
    get_tool_prompt_description(t)

def test_tool7():
    t = get_tool7()
    get_tool_prompt_description(t)




def run_agent_example():
    from archytas.react import ReActAgent, Role
    from easyrepl import REPL

    tools = [
        get_tool00(),
        get_tool0(),
        # get_tool1(),
        # get_tool2(),
        get_tool3(),
        get_tool4(),
        # get_tool5(),
        get_tool6(),
        get_tool7(),
    ]

    agent = ReActAgent(model='gpt-4o-mini', tools=tools, verbose=True)
    print(f'prompt:\n```\n{agent.prompt}\n```')

    for query in REPL(history_file='.history'):
        response = agent.react(query)
        print(response)


if __name__ == '__main__':
    run_agent_example()