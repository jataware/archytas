import json
from types import GenericAlias, UnionType
from typing import Any, Union, get_origin, get_args as get_type_args


def extract_json(text: str) -> dict | list:
    """
    Finds and extracts JSON from a block of text, probably a response from a LLM.
    """
    # Convert agent output to json
    lines = text.splitlines(keepends=True)
    json_block_starts = [linenum + 1 for linenum, line in enumerate(lines) if line.strip().startswith("```json")]
    if json_block_starts:
        result = []
        for block_start in json_block_starts:
            try:
                block_end =  next(linenum for linenum, line in enumerate(lines) if linenum > block_start and line.strip().startswith("```"))
                json_text = "".join(lines[block_start:block_end])
            except StopIteration:
                raise ValueError("Unable to determine bounds of triple-backtick delimited text")
            result.append(json.loads(json_text))
        if len(result) == 1:
            return result[0]
        else:
            return result
    else:
        raise ValueError("Unable to find json block")


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


def type_to_str(t: type | GenericAlias | UnionType | None) -> str:
    """
    Convert a type to a string representation
    """
    # TODO: this could be more robust, there are probably cases it doesn't cover
    if isinstance(t, type):
        return t.__name__
    elif isinstance(t, GenericAlias):
        return f"{get_origin(t).__name__}[{', '.join(type_to_str(a) for a in get_type_args(t))}]"
    elif get_origin(t) is Union:
        return ' | '.join(type_to_str(a) for a in get_type_args(t))
    elif isinstance(t, UnionType):
        return str(t)
    elif t is None:
        return "None"
    else:
        raise ValueError(f"Unsupported type {t}")
