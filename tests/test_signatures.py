from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from archytas.tool_utils import tool
from typing import Union, List, Optional, Any

import pytz
from datetime import datetime

import pytest

import pdb


@dataclass
class A:
    a: int = 5
    b: list = field(default_factory=list)


@dataclass
class B:
    a: str
    b: str


@dataclass
class C:
    b: B
    a: A = field(default_factory=A)


@dataclass
class D:
    ab: A | B
    c: C | float


class M1(BaseModel):
    a: int = Field(default=1, description="description of a")
    b: int = Field(default=2, description="description of b")


class M2(BaseModel):
    a: str = Field(default="a", description="description of a")
    b: str = Field(default="b", description="description of b")


class M3(BaseModel):
    a: M1 = Field(default_factory=M1, description="description of a")
    b: M2 = Field(default_factory=M2, description="description of b")


class GenericModelA(BaseModel):
    param_a: float = Field(default=1.0, description="A general float parameter for customization")
    param_b: float = Field(default=31.0, description="A general float parameter for analysis")
    param_c: float = Field(default=25.0, description="Another general float parameter with a specific use")
    param_d: float = Field(default=4.7, description="float parameter for model input")
    param_e: float = Field(default=80.0, description="metric for generic modeling")
    count_x: int = Field(default=5, description="Integer parameter representing a count or quantity")
    rate_y: float = Field(default=70.0, description="Rate parameter for process evaluation")
    cost_adjuster_1: float = Field(default=100.0, description="Adjuster for cost or pricing analysis")
    multiplier_z: float = Field(default=1.2, description="Multiplier factor for cost or scaling")
    length_param: float = Field(default=1000.0, description="Length parameter for model calculations")
    declination_factor: Union[float, List[float]] = Field(
        default=1.0, description="Declination rate or factors for modeling")
    size_param: float = Field(default=0.16256, description="Size-related float parameter")
    count_y: int = Field(default=5, description="Secondary integer count parameter")
    coeff_param: float = Field(default=2500.0, description="Coefficient for analysis")
    import_rate: float = Field(default=0.07, description="Import rate for modeling")
    distance_metric: float = Field(default=1.0, description="Distance or proximity measurement")
    time_frame_1: int = Field(default=2, description="Time frame for process")
    buffer_time: int = Field(default=1, description="Buffer or preparation time in the process")


class GenericModelB(BaseModel):
    operator_a: float = Field(default=1.0, description="A general float operator for customization")
    operator_b: float = Field(default=31.0, description="A general float operator for analysis")
    operator_c: float = Field(default=25.0, description="Another general float operator with a specific use")
    operator_d: float = Field(default=4.7, description="float operator for model input")
    operator_e: float = Field(default=80.0, description="metric for generic modeling")
    count_x: int = Field(default=5, description="Integer parameter representing a count or quantity")
    rate_y: float = Field(default=70.0, description="Rate parameter for process evaluation")
    cost_adjuster_1: float = Field(default=100.0, description="Adjuster for cost or pricing analysis")
    multiplier_z: float = Field(default=1.2, description="Multiplier factor for cost or scaling")
    length_param: float = Field(default=1000.0, description="Length parameter for model calculations")
    declination_factor: Union[float, List[float]] = Field(
        default=1.0, description="Declination rate or factors for modeling")
    optional_param: Optional[str] = Field(default=None, description="Optional parameter for customization")
    size_param: float = Field(default=0.16256, description="Size-related float parameter")
    count_y: int = Field(default=5, description="Secondary integer count parameter")
    coeff_param: float = Field(default=2500.0, description="Coefficient for analysis")
    import_rate: float = Field(default=0.07, description="Import rate for modeling")
    distance_metric: float = Field(default=1.0, description="Distance or proximity measurement")
    time_frame_1: int = Field(default=2, description="Time frame for process")
    buffer_time: int = Field(default=1, description="Buffer or preparation time in the process")


def get_test_mytool():
    @tool(devmode=True)
    def mytool(a: list[int], b: list[float]):
        """
        Args:
            a (Any): Description of the argument `a`
            b (Any): Description of the argument `b`
        """
        print(f"a: {a}, b: {b}")

    return mytool


def get_test_tool0():
    @tool(devmode=True)
    def tool0(a: list[int], b: list[float]):
        """
        Args:
            a (list): Description of the argument `a`
            b (list): Description of the argument `b`
        """
        print(f"a: {a}, b: {b}")

    return tool0


def get_test_tool1():
    @tool(devmode=True)
    def tool1(item: A):
        """
        Args:
            item (A): Description of the argument `item`
        """
        print(f"item: {item}")

    return tool1


def get_test_tool2():
    @tool(devmode=True)
    def tool2(item: B, i: int):
        """
        Args:
            item (B): Description of the argument `item`
            i (int): Description of the argument `i`
        """
        print(f"item: {item}, i: {i}")

    return tool2


def get_test_tool3():
    @tool(devmode=True)
    def tool3(a: A, b: B):
        """
        Args:
            a (A): Description of the argument `a`
            b (B): Description of the argument `b`
        """
        print(f"a: {a}, b: {b}")

    return tool3


def get_test_tool4():
    @tool(name='apple', devmode=True)
    def tool4(item: C):
        """
        Args:
            item (C): Description of the argument `item`
        """
        print(f"item: {item}")

    return tool4


def get_test_tool5():
    @tool(devmode=True)
    def tool5(a: A, b: B, c: C, l: Optional[list[int]] = None):
        """
        Args:
            a (A): Description of the argument `a`
            b (B): Description of the argument `b`
            c (C): Description of the argument `c`
            l (list[int], optional): Description of the argument `l`. Defaults to [].
        """
        print(f"a: {a}, b: {b}, c: {c}, l: {l}")

    return tool5


def get_test_tool5a():
    @tool(devmode=True)
    def tool5a(d: D):
        """
        Args:
            d (D): Description of the argument `d`
        """
        print(f"d: {d}")

    return tool5a


def get_test_tool6():
    @tool(autosummarize=True, devmode=True)
    def tool6(item: M1):
        """
        Args:
            item (M1): Description of the argument `item`
        """
        print(f"item: {item}")

    return tool6


def get_test_tool7():
    @tool(devmode=True)
    def tool7(item: M2):
        """
        Args:
            item (M2): Description of the argument `item`
        """
        print(f"item: {item}")

    return tool7


def get_test_tool8():
    @tool(devmode=True)
    def tool8(item: M3):
        """
        Args:
            item (M3): Description of the argument `item`
        """
        print(f"item: {item}")

    return tool8


def get_test_tool9():
    @tool(devmode=True)
    def tool9(item: M3, a: A, c: C):
        """
        Args:
            item (M3): Description of the argument `item`
            a (A): Description of the argument `a`
            c (C): Description of the argument `c`
        """
        print(f"item: {item}, a: {a}, c: {c}")

    return tool9


def get_test_tool10():
    @tool(devmode=True)
    def tool10(app: GenericModelA):
        """
        Args:
            app (GenericModelA): Description of the argument `app`
        """
        print(f"app: {app}")

    return tool10


def get_test_tool11():
    @tool(devmode=True)
    def tool11(heat: GenericModelA, cool: GenericModelB):
        """
        Args:
            heat (GenericModelA): Description of the argument `heat`
            cool (GenericModelB): Description of the argument `cool`
        """
        print(f"heat: {heat}, cool: {cool}")

    return tool11


def get_test_tool12():
    @tool(devmode=True)
    def tool12(app: GenericModelA, a: A, c: C, i: int = 5, l: list | None = None):
        """
        Args:
            app (GenericModelA): Description of the argument `app`
            a (A): Description of the argument `a`
            c (C): Description of the argument `c`
            i (int): Description of the argument `i`. Defaults to 5
            l (list): Description of the argument `l`. Defaults to None
        """
        print(f"app: {app}, a: {a}, c: {c}, i: {i}, l: {l}")

    return tool12


def get_test_datetime_tool():
    @tool(name="datetime", devmode=True)
    def datetime_tool(format: str = "%Y-%m-%d %H:%M:%S %Z", timezone: str = "UTC") -> str:
        """
        Get the current date and time.

        Args:
            format (str, optional): The format to return the date and time in. Defaults to '%Y-%m-%d %H:%M:%S %Z'.
            timezone (str, optional): The timezone to return the date and time in. Defaults to 'UTC'.

        Returns:
            str: The current date and time in the specified format
        """
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime(format)

    return datetime_tool


def get_test_datetime_simple():
    @tool(devmode=True)  # (name="datetime")
    def datetime_simple() -> str:
        """
        Get the current date and time.

        Returns:
            str: The current date and time in UTC
        """
        return datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S %Z")

    return datetime_simple


def get_test_single_int():
    @tool(devmode=True)
    def single_int(i: int) -> int:
        """
        test tool that takes a single integer

        Args:
            i (int): Description of the argument `i`

        Returns:
            int: The integer you gave me
        """
        return i

    return single_int


def get_test_list_of_ints():
    @tool(devmode=True)
    def list_of_ints(l: list[int]) -> list[int]:
        """
        test tool that takes a list of integers

        Args:
            l (list): Description of the argument `l`

        Returns:
            list: The list of integers you gave me
        """
        return l

    return list_of_ints


def get_test_dict_of_ints():
    @tool(devmode=True)
    def dict_of_ints(d: dict[str, int]) -> dict[str, int]:
        """
        test tool that takes a dictionary of integers

        Args:
            d (dict): Description of the argument `d`

        Returns:
            dict[str, int]: The dictionary of integers you gave me
        """
        return d

    return dict_of_ints


def get_test_union_int_str():
    @tool(devmode=True)
    def union_int_str(x: int | str) -> str:
        """
        test tool that takes an int or a string

        Args:
            x (int | str): Description of the argument `x`

        Returns:
            str: The type of the argument you gave me
        """
        if isinstance(x, int):
            return 'you gave me an int'
        elif isinstance(x, str):
            return 'you gave me a string'
        else:
            raise ValueError('I only accept ints and strings')

    return union_int_str


def get_test_positional_only():
    @tool(devmode=True)
    def positional_only(a: int, b: B, /) -> tuple[int, B]:
        """
        test tool that has positional only arguments

        Args:
            a (int): Description of the argument `a`
            b (B): Description of the argument `b`

        Returns:
            tuple[int, B]: The arguments you gave me
        """
        return a, b

    return positional_only


def get_test_returns_union():
    import random

    @tool(devmode=True)
    def returns_union() -> int | str:
        """
        test tool that returns an int or a string

        Returns:
            int|str: The type of the return value
        """
        if random.random() < 0.5:
            return 'I am a string'
        return 1

    return returns_union


@pytest.mark.xfail(reason="`Any` not allowed in docstring type annotations")
def test_mytool():
    get_test_mytool()


def test_tool0():
    get_test_tool0()


def test_tool1():
    get_test_tool1()


def test_tool2():
    get_test_tool2()


def test_tool3():
    get_test_tool3()


def test_tool4():
    get_test_tool4()


def test_tool5():
    get_test_tool5()


@pytest.mark.xfail(reason="Union of structs not supported")
def test_tool5a():
    get_test_tool5a()


def test_tool6():
    get_test_tool6()


def test_tool7():
    get_test_tool7()


def test_tool8():
    get_test_tool8()


def test_tool9():
    get_test_tool9()


def test_tool10():
    get_test_tool10()


def test_tool11():
    get_test_tool11()


def test_tool12():
    get_test_tool12()


def test_datetime_tool():
    get_test_datetime_tool()


def test_datetime_simple():
    get_test_datetime_simple()


def test_single_int():
    get_test_single_int()


def test_list_of_ints():
    get_test_list_of_ints()


def test_dict_of_ints():
    get_test_dict_of_ints()


def test_union_int_str():
    get_test_union_int_str()


def test_positional_only():
    get_test_positional_only()


def test_returns_union():
    get_test_returns_union()


def run_agent_example():
    from archytas.react import ReActAgent, Role
    from easyrepl import REPL

    tools = [
        get_test_tool0(),
        get_test_tool1(),
        get_test_tool2(),
        get_test_tool3(),
        get_test_tool4(),
        get_test_tool5(),
        # get_test_tool5a(),
        get_test_tool6(),
        get_test_tool7(),
        get_test_tool8(),
        get_test_tool9(),
        get_test_datetime_tool(),
        get_test_datetime_simple(),
        get_test_single_int(),
        get_test_list_of_ints(),
        get_test_dict_of_ints(),
        get_test_union_int_str(),
        get_test_tool10(),
        get_test_tool11(),
        get_test_tool12(),
        get_test_positional_only(),
        get_test_returns_union(),
    ]

    agent = ReActAgent(model='gpt-4o-mini', tools=tools, verbose=True)
    print(f'prompt:\n```\n{agent.prompt}\n```')

    for query in REPL(history_file='.history'):
        response = agent.react(query)
        print(response)


if __name__ == '__main__':
    run_agent_example()