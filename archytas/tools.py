from archytas.tool_utils import tool, toolset


from easyrepl import readl
@tool()
def ask_user(query:str) -> str:
    """
    Ask the user a question and get their response. 
    
    You should ask the user a question if you do not have enough information to complete the task, and there is no suitable tool to help you.
    
    Args:
        query (str): The question to ask the user

    Returns:
        str: The user's response
    """
    return readl(prompt=f'{query} ')



from datetime import datetime
import pytz
@tool(name='datetime')
def datetime_tool(format:str='%Y-%m-%d %H:%M:%S %Z', timezone:str='UTC') -> str:
    """
    Get the current date and time. 
    
    Args:
        format (str, optional): The format to return the date and time in. Defaults to '%Y-%m-%d %H:%M:%S %Z'.
        timezone (str, optional): The timezone to return the date and time in. Defaults to 'UTC'.

    Returns:
        str: The current date and time in the specified format
    """
    # TODO: See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more information.
    # TODO: list of valid timezones: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime(format)


@tool()
def timestamp() -> float:
    """
    Returns the current unix timestamp in seconds
    
    Returns:
        float: The current unix timestamp in seconds 

    Examples:
        >>> timestamp()
        1681445698.726113
    """
    return datetime.now().timestamp()
