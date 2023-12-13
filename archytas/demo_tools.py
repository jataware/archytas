from archytas.tool_utils import tool


@tool()
def fib_n(n: int) -> int:
    """
    generate the nth fibonacci number

    Args:
        n (int): The index of the fibonacci number to get

    Returns:
        int: The nth fibonacci number

    Examples:
        >>> fib_n(10)
        55
        >>> fib_n(20)
        6765
    """
    n0 = 0
    n1 = 1
    for _ in range(n):
        n0, n1 = n1, n0 + n1

    return n0


@tool()
def example_tool(arg1: int, arg2: str = "", arg3: dict = None) -> int:
    """
    Simple 1 sentence description of the tool

    More detailed description of the tool. This can be multiple lines.
    Explain more what the tool does, and what it is used for.

    Args:
        arg1 (int): Description of the first argument.
        arg2 (str): Description of the second argument. Defaults to ''.
        arg3 (dict): Description of the third argument. Defaults to {}.

    Returns:
        int: Description of the return value

    Examples:
        >>> example_tool(1, 'hello', {'a': 1, 'b': 2})
        3
        >>> example_tool(2, 'world', {'a': 1, 'b': 2})
        4
    """
    return 42


@tool()
def calculator(expression: str) -> float:
    """
    A simple calculator tool. Can perform basic arithmetic

    Expressions must contain exactly:
    - one left operand. Can be a float or integer
    - one operation. Can be one of + - * / ^ %
    - one right operand. Can be a float or integer

    multiple chained operations are not currently supported.

    Expressions may not contain parentheses, or multiple operations.
    If you want to do a complex calculation, you must do it in multiple steps.

    Args:
        expression (str): A string containing a mathematical expression.

    Returns:
        float: The result of the calculation

    Examples:
        >>> calculator('22/7')
        3.142857142857143
        >>> calculator('3.24^2')
        10.4976
        >>> calculator('3.24+2.5')
        5.74
    """

    # ensure that only one operation is present
    ops = [c for c in expression if c in "+-*/^%"]
    if len(ops) > 1:
        raise ValueError(
            f"Invalid expression, too many operators. Expected exactly one of '+ - * / ^ %', found {', '.join(ops)}"
        )
    if len(ops) == 0:
        raise ValueError(
            f"Invalid expression, no operation found. Expected one of '+ - * / ^ %'"
        )

    op = ops[0]

    _a, _b = expression.split(op)
    a = float(_a)
    b = float(_b)

    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    elif op == "^":
        return a**b
    elif op == "%":
        return a % b


# TODO: still not great since have to include cls in the signature
class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


import math


# @tool()
class Math:
    def __init__(self):
        pass

    # @tool
    @staticmethod
    def sin(x: float) -> float:
        """
        Calculate the sine of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The sine of x
        """
        return math.sin(x)

    # @tool
    @staticmethod
    def cos(x: float) -> float:
        """
        Calculate the cosine of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The cosine of x
        """
        return math.cos(x)

    # @tool
    @staticmethod
    def tan(x: float) -> float:
        """
        Calculate the tangent of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The tangent of x
        """
        return math.tan(x)

    # @tool
    @classproperty  # TODO: still not great since have to include cls in the signature
    def pi(*_) -> float:
        """
        Get the value of pi

        Returns:
            float: The value of pi
        """
        return math.pi


from random import random


class Jackpot:
    """
    A simple slot machine game

    Start with 100 chips, and make bets to try and win more chips.
    """

    def __init__(
        self,
        chips: float = 100,
        win_table: list[tuple[float, float]] = [
            (0.01, 20),
            (0.02, 10),
            (0.05, 4.5),
            (0.2, 1.25),
        ],
    ):
        self._initial_chips = chips
        self.chips = chips
        self.win_table = win_table

    @tool()
    def spin(self, bet: float) -> float:
        """
        Spin the slot machine

        Args:
            bet (float): The amount to bet. Must be less than or equal to the current amount of chips in your wallet

        Returns:
            float: The amount won or lost
        """
        if bet > self.chips:
            raise ValueError(
                f"Bet must be less than or equal to the current winnings. Bet: {bet}, Winnings: {self.winnings}"
            )

        spin = random()
        total_prob = 0
        multiplier = -1
        for prob, win_multiplier in self.win_table:
            total_prob += prob
            if spin <= total_prob:
                multiplier = win_multiplier
                break

        self.chips += bet * multiplier
        return bet * multiplier

    @tool()
    def get_chips(self) -> float:
        """
        Get the current amount of chips in your wallet

        Returns:
            float: The current amount of chips in your wallet
        """
        return self.chips

    @tool()
    def reset(self):
        """
        Reset the game back to the initial number of chips
        """
        self.chips = self._initial_chips


class ModelSimulation:
    """
    Simple example of a SIR model simulation
    """

    def __init__(self, dt=0.1):
        self._default_parameters = {
            "beta": 0.002,
            "gamma": 0.1,
            "S": 990,
            "I": 10,
            "R": 0,
        }
        self.parameters = self._default_parameters.copy()
        self.dt = dt

    @tool()
    def get_model_parameters(self) -> dict:
        """
        Get the model parameters

        Returns:
            dict: The model parameters in the form {param0: value0, param1: value1, ...}

        """
        return self.parameters

    @tool()
    def set_model_parameters(self, update: dict):
        """
        Set some or all of the model parameters

        Args:
            update (dict): The parameters to update. Should be a dict of the form {param0: value0, param1: value1, ...}. Only the parameters specified will be updated.
        """
        self.parameters.update(update)

    @tool()
    def run_model(self, steps: int = 100) -> dict:
        """
        Run the model for a number of steps

        Args:
            steps (int): The number of steps to run the model for. Defaults to 100.

        Returns:
            dict: The model results in the form {param0: value0, param1: value1, ...}
        """
        S_new, I_new, R_new = (
            self.parameters["S"],
            self.parameters["I"],
            self.parameters["R"],
        )
        beta, gamma = self.parameters["beta"], self.parameters["gamma"]
        population = S_new + I_new + R_new

        for _ in range(steps):
            S_old, I_old, R_old = S_new, I_new, R_new
            dS = -beta * S_old * I_old
            dI = beta * S_old * I_old - gamma * I_old
            dR = gamma * I_old

            S_new = max(0, min(S_old + self.dt * dS, population))
            I_new = max(0, min(I_old + self.dt * dI, population))
            R_new = max(0, min(R_old + self.dt * dR, population))

            # Ensure the total population remains constant
            total_error = population - (S_new + I_new + R_new)
            R_new += total_error

        self.parameters["S"], self.parameters["I"], self.parameters["R"] = (
            S_new,
            I_new,
            R_new,
        )
        return self.parameters

    @tool()
    def reset_model(self):
        """
        Reset the model to the initial parameters
        """
        self.parameters = self._default_parameters.copy()


@tool()
def ObservablePlot(code: str):
    """
    Create an observable plot in code.

    Args:
        code (str): The Observable Plot code

    """


class PlannerTool:
    """
    A tool for helping to make long term plans. After a plan is made, the system will remind you of the plan as you execute each of the steps.
    """

    def __init__(self):
        self.current_plan = None
        self.progress = None

    @tool()
    def make_new_plan(self, draft: list[str]) -> None:
        """
        Start the plan making process.

        This starts a conversation with the system to refine and revise a plan until it is robust and well thought out.

        Args:
            draft (list[str]): An initial list of steps to include in the plan. These can be revised later.
        """
        self.current_plan = draft
        import pdb

        pdb.set_trace()
        ...

    @tool()
    def revise_single_step(self, i: int, replacement: str) -> None:
        """
        Revise a single step in the current plan

        Args:
            i (int): The index of the step to revise
            replacement (str): The new step to replace the old step with
        """
        raise NotImplementedError("TODO")

    # @tool()
    # def revise_plan(self, updates:dict[int,str]) -> None: ...

    # @tool()
    # def finalize_plan():...


from .tool_utils import AgentRef


@tool()
def pirate_subquery(query: str, agent: AgentRef) -> str:
    """
    Runs a subquery using a oneshot agent in which answers will be worded like a pirate.

    Args:
        query (str): The query to run against the agent.

    Returns:
        str: Result of the subquery in pirate vernacular.

    """
    prompt = """
    You are an pirate. Answer all questions truthfully using pirate vernacular.
    """
    return agent.oneshot(prompt=prompt, query=query)
