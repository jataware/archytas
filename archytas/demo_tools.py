from archytas.tool_utils import tool, toolset



@tool()
def fib_n(n:int) -> int:
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
def example_tool(arg1:int, arg2:str='', arg3:dict=None) -> int:
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
def calculator(expression:str) -> float:
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

    #ensure that only one operation is present
    ops = [c for c in expression if c in '+-*/^%']
    if len(ops) > 1:
        raise ValueError(f"Invalid expression, too many operators. Expected exactly one of '+ - * / ^ %', found {', '.join(ops)}")
    if len(ops) == 0:
        raise ValueError(f"Invalid expression, no operation found. Expected one of '+ - * / ^ %'")
    
    op = ops[0]

    _a, _b = expression.split(op)
    a = float(_a)
    b = float(_b)

    if op == '+':
        return a+b
    elif op == '-':
        return a-b
    elif op == '*':
        return a*b
    elif op == '/':
        return a/b
    elif op == '^':
        return a**b
    elif op == '%':
        return a%b


#TODO: still not great since have to include cls in the signature
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
    def sin(x:float) -> float:
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
    def cos(x:float) -> float:
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
    def tan(x:float) -> float:
        """
        Calculate the tangent of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The tangent of x
        """
        return math.tan(x)
    
    # @tool
    @classproperty #TODO: still not great since have to include cls in the signature
    def pi(*_) -> float:
        """
        Get the value of pi

        Returns:
            float: The value of pi
        """
        return math.pi



# @tool()
class StatefulToolExample:
    def __init__(self, i:int, s:str):
        self.i = i
        self.s = s

    # @tool()
    def inc(self) -> int:
        """
        increment the internal counter

        Returns:
            int: The new value of the internal counter
        """
        self.i += 1
        return self.i

    # @tool()
    def set_i(self, i:int):
        """
        set the internal counter

        Args:
            i (int): The new value of the internal counter
        """
        self.i = i

    # @tool()
    def set_s(self, s:str):
        """
        set the internal string

        Args:
            s (str): The new value of the internal string
        """
        self.s = s

    # @tool()
    def get_i(self) -> int:
        """
        get the internal counter
        
        Returns:
            int: The value of the internal counter
        """
        return self.i
    
    # @tool()
    def get_s(self) -> str:
        """
        get the internal string

        Returns:
            str: The value of the internal string
        """

        return self.s


from random import random
@toolset()
class Jackpot:
    """
    A simple slot machine game
    
    Start with 100 chips, and make bets to try and win more chips.
    """
    def __init__(self, chips:float=100, win_table:list[tuple[float,float]]=[(0.01, 20), (0.02, 10), (0.05, 4.5), (0.2, 1.25)]):
        self._initial_chips = chips
        self.chips = chips
        self.win_table = win_table

    @tool()
    def spin(self, bet:float) -> float:
        """
        Spin the slot machine

        Args:
            bet (float): The amount to bet. Must be less than or equal to the current amount of chips in your wallet

        Returns:
            float: The amount won or lost
        """
        if bet > self.chips:
            raise ValueError(f"Bet must be less than or equal to the current winnings. Bet: {bet}, Winnings: {self.winnings}")
        
        spin = random()
        total_prob = 0
        multiplier = -1
        for prob, win_multiplier in self.win_table:
            total_prob += prob
            if spin <= total_prob:
                multiplier = win_multiplier
                break
                
        self.chips += bet*multiplier            
        return bet*multiplier
            
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

