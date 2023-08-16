from typing import Any


def get_local_name(val: Any, locals: dict[str, Any]) -> str:
    """
    Determines the name of a local variable by finding a matching value in the given locals dict

    Performs a linear search over locals, so may not be performant if done frequently.

    Args:
        val (Any): The value to find the name of. This can be a module, class, function, variable, etc.
        locals (dict[str, Any]): The locals dict to search for the value in. This is usually the locals() dict from the root scope.

    Returns:
        str: The name of the local variable

    Raises:
        ValueError: If the value is not found in the locals dict
    """
    for name, obj in locals.items():
        if obj is val:
            return name
    raise ValueError(f"Value {val} not found in locals")


class InstanceMethod:
    """
    A descriptor that wraps a method so that it can be accessed from an instance of the class, but not from the class itself.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            raise TypeError(
                "This method should only be accessed from an instance of the class"
            )
        return self.func.__get__(instance, owner)
