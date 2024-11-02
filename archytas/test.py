class A: ...

class B:
    def my_fn(self, a: int = 5) -> int:
        """
        Test Function

        Args:
            a (int, optional): Description of the argument `a`. Defaults to 5.
        
        Returns:
            int: Description of the return value.
        """
        return a

def my_fn(a: int = 5) -> int:
    """
    Test Function

    Args:
        a (int, optional): Description of the argument `a`. Defaults to 5.
    
    Returns:
        int: Description of the return value.
    """
    return a

def my_fn2(a: str) -> str: 
    """
    Test Function

    Args:
        a (str): Description of the argument `a`.
    
    Returns:
        str: Description of the return value.
    """
    return a

def my_fn1(a: A) -> A:
    """
    Test Function

    Args:
        a (A): Description of the argument `a`.

    Returns:
        A: Description of the return value.
    """
    return a