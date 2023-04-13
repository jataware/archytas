
def ask_user(query:str) -> str:
    """Ask the user a question. Returns the user's response"""
    return readl(prompt=f'{query} ')

from datetime import datetime
import pytz
#TODO: arguments need to be parsed. input should be a json with {format:str, timezone:str}
def datetime_tool(kwargs) -> str:
    def datetime_tool(format:str='%Y-%m-%d %H:%M:%S %Z', timezone:str='UTC') -> str:
        """Returns the current date and time in the specified format. 
        Defaults to YYYY-MM-DD HH:MM:SS format. 
        TODO: See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more information.
        TODO: list of valid timezones: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        default timezone is UTC
        """
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime(format)

    return datetime_tool(**kwargs)


def timestamp(_) -> str:
    """Returns the current unix timestamp in seconds"""
    return str(datetime.now().timestamp())

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
    