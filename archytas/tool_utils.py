import logging
from .constants import TAB
from .agent import Agent
from .utils import type_to_str
from .structured_data_utils import (
    is_structured_type,
    get_structured_input_description,
    construct_structured_type
)
from typing import Callable, Any, ParamSpec, TypeVar, get_origin, get_args as get_type_args, Union, overload
from types import UnionType, GenericAlias
from textwrap import indent
import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback

traceback.install(show_locals=True)


logger = logging.getLogger(__name__)


# TODO: separate system tools from library tools from user tools
#      ideally system tools will have no library dependencies

# TODO: parse extra long manual page from doc string. man_page can be seen by calling man <tool_name>
# man_page:str|None=None
# wrapper._man_page = man_page


# Class/type definition for types used in dependency injection.
AgentRef = type("AgentRef", (), {})
ToolNameRef = type("ToolNameRef", (), {})
ToolFnRef = type("ToolFnRef", (), {})
LoopControllerRef = type("LoopControllerRef", (), {})
ReactContextRef = type("ReactContextRef", (), {})

INJECTION_MAPPING = {
    AgentRef: "agent",
    ToolNameRef: "tool_name",
    ToolFnRef: "raw_tool",
    LoopControllerRef: "loop_controller",
    ReactContextRef: "react_context",
}


def toolset(*args, **kwargs):
    """
    A dummy decorator for backwards compatibility.
    Provides no funcitonality.
    Any class can now contain tools without a decorator.
    """
    logger.warning(
        "Warning: The usage of the @toolset decorator is deprecated and the decorator will be removed in a future version.")

    def decorator(cls):
        return cls
    return decorator


R = TypeVar("R")
P = ParamSpec("P")
@overload
def tool(func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def tool(*, name: str | None = None, autosummarize: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def tool(func: Callable[P, R], /, *, name: str | None = None, autosummarize: bool = False) -> Callable[P, R]: ...


def tool(func=None, /, *, name: str | None = None, autosummarize: bool = False):
    """
    Decorator to convert a function into a tool for ReAct agents to use.

    Usage:
    ```
        @tool
        def my_tool(arg:type) -> type:
            '''
            Short description of the tool

            Long description of the tool

            Args:
                arg (type): Description of the argument

            Returns:
                type: Description of the return value

            Examples:
                optional description of the example
                >>> my_tool(arg)
                result
            '''
    ```
    """

    # decorator case where the decorator is used directly on the func
    # either `@tool def func()` or `tool(func, name='name', autosummarize=True)`
    if func is not None:
        return tool(name=name, autosummarize=autosummarize)(func)

    def decorator(func: Callable):
        # check that the decorator is being applied to a function
        if not inspect.isfunction(func):
            raise TypeError(
                f"tool decorator can only be applied to functions or classes. Got {func} of type {type(func)}"
            )

        # attach usage description to the wrapper function
        args_list, ret, desc, injections = get_tool_signature(func)
        arg_preprocessor = make_arg_preprocessor(args_list)

        func._name = name if name else func.__name__
        func._is_tool = True
        func.autosummarize = autosummarize

        async def run(
            args: dict | None,
            tool_context: dict[str, object] = None,
            self_ref: object = None

        ):
            """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""

            # Initialise positional and keyword argument structs
            pargs, kwargs = arg_preprocessor(args)

            if self_ref:
                pargs.insert(0, self_ref)

            # Add injections to kwargs
            for inj_name, inj_type in injections.items():
                context_key = INJECTION_MAPPING.get(inj_type, None)
                context_value = tool_context.get(context_key, None)
                if context_value:
                    kwargs[inj_name] = context_value

            if inspect.iscoroutinefunction(func):
                result = await func(*pargs, **kwargs)
            else:
                result = func(*pargs, **kwargs)

            # convert the result to a string if it is not already a string
            if not isinstance(result, str):
                result = str(result)

            return result

        # Add func as the attribute of the run method
        func.run = run

        return func

    return decorator


def is_tool(obj: Callable | type) -> bool:
    """checks if an object is a tool function, tool method, tool class, or an instance of a class tool"""
    return (
        getattr(obj, '_is_tool', False)
    )


def get_tool_prompt_description(obj: Callable | type | Any):

    # get the list of arguments
    chunks = []
    if getattr(obj, '_disabled', False):
        return ""

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        args_list, ret, desc, injections = get_tool_signature(obj)
        args_list = args_list
        ret_name, ret_type, ret_description = ret
        short_desc, long_desc, examples = desc

        ############### NAME/DESCRIPTION ###############
        chunks.append(f"{obj._name}:\n")
        if short_desc:
            chunks.append(f"{TAB}{short_desc}\n\n")
        if long_desc:
            chunks.append(f"{indent(long_desc, TAB)}\n\n")

        #################### INPUT ####################
        chunks.append(f"{TAB}_input_: ")

        if len(args_list) == 0:
            chunks.append("None")

        # all other cases have arguments wrapped in a json
        else:
            chunks.append(f"a json object with the following fields:\n{TAB}{{")
            for arg_name, arg_type, arg_desc, arg_default in args_list:
                if is_structured_type(arg_type):
                    chunks.extend(get_structured_input_description(arg_type, arg_name, arg_desc, arg_default, indent=2))
                    continue

                if not is_primitive_type(arg_type):
                    raise ValueError(f"Unsupported argument type {arg_type}")

                chunks.append(
                    f'\n{TAB}{TAB}"{arg_name}": ({type_to_str(arg_type)}{", optional" if arg_default else ""}) {arg_desc}')
            chunks.append(f"\n{TAB}}}")

        #################### OUTPUT ####################
        chunks.append("\n    _output_: ")
        if ret_type is None:
            chunks.append("None")
        else:
            chunks.append(f"({type_to_str(ret_type)}) {ret_description}")

        ############### EXAMPLES ###############
        # TODO: examples need to be parsed...
    else:
        tool_methods = [
            tool_method
            for tool_method in collect_tools_from_object(obj)
            if not getattr(tool_method, "_disabled", False)
        ]
        if tool_methods:
            ############### NAME/DESCRIPTION ###############
            if inspect.isclass(obj):
                chunks.append(f"{obj.__name__} (class):\n")
            else:
                chunks.append(f"{obj.__class__.__name__} (class instance):\n")
            docstring = inspect.getdoc(obj)
            if docstring:
                chunks.append(f"{indent(docstring, TAB)}\n\n")

            #################### METHODS ####################
            chunks.append(f"{TAB}methods:\n")
            for method in tool_methods:
                method_str = get_tool_prompt_description(method)
                chunks.append(f"{indent(method_str, TAB*2)}\n\n")

    return "".join(chunks)


def get_tool_signature(
    func: Callable,
) -> tuple[
    list[tuple[str, type, str | None, str | None]],
    tuple[str | None, type | None, str | None],
    tuple[str, str | None, list[str]],
]:
    """
    Check that the docstring and function signature match for a tool function, and return all function information.

    Args:
        func (function): The function to check and extract information from

    Returns:
        args_list: A list of tuples (name, type, description, default) for each argument
        ret: A tuple (name, type, description) for the return value
        desc: A tuple (short_description, long_description, examples) from the docstring for the function
    """
    assert inspect.isfunction(func) or inspect.ismethod(
        func), f"get_tool_signature can only be used on functions or methods. Got {func}"

    # get the function signature from the function and the docstring
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # Extract argument information from the docstring
    docstring_args = {
        arg.arg_name: (arg.type_name, arg.description, arg.default)
        for arg in docstring.params
    }

    # Extract argument information from the signature (ignore self from class methods)
    all_args = {
        k: v.annotation
        for i, (k, v) in enumerate(signature.parameters.items())
        if not (i == 0 and k == "self")
    }

    # Extract argument information from the signature (ignore self from class methods)
    injected_args = {k: v for k, v in all_args.items() if v in INJECTION_MAPPING}
    signature_args = {k: v for k, v in all_args.items() if k not in injected_args}

    # Check if the docstring argument names match the signature argument names
    if set(docstring_args.keys()) != set(signature_args.keys()):
        raise ValueError(
            f"Docstring argument names do not match function signature argument names for function '{func.__name__}'"
        )

    # Check if the docstring argument types match the signature argument types
    for arg_name, arg_type in signature_args.items():
        docstring_arg_type, _, _ = docstring_args[arg_name]
        if docstring_arg_type not in (type_to_str(arg_type), type_to_str(get_origin(arg_type))):
            raise ValueError(
                f"Docstring type '{docstring_arg_type}' does not match function signature type '{arg_type.__name__}' for argument '{arg_name}' for function '{func.__name__}'"
            )

    # Generate a list of tuples (name, type, description, default) for each argument
    # TODO: use the signature to determine if an argument is optional or not
    args_list = [
        (arg_name, signature_args[arg_name], arg_desc, arg_default)
        for arg_name, (arg_type, arg_desc, arg_default) in docstring_args.items()
    ]

    # get the return type and description (and correctly set None to empty return type)
    signature_ret_type = signature.return_annotation
    if signature_ret_type is None:
        signature_ret_type = inspect.Signature.empty

    try:
        docstring_ret_type = docstring.returns.type_name
    except AttributeError:
        docstring_ret_type = "_empty"
    if type_to_str(signature_ret_type) != docstring_ret_type:  # and signature_ret_type.__name__ != docstring_ret_type:
        logger.warning(
            f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{type_to_str(signature_ret_type)}' for function '{func.__name__}'"
        )
        # raise ValueError(
        #     f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{type_to_str(signature_ret_type)}' for function '{func.__name__}'"
        # )

    # get the return type and description
    if docstring.returns is None:
        ret = (None, None, None)
    else:
        ret = (
            docstring.returns.return_name,
            signature_ret_type,
            docstring.returns.description,
        )

    # get the docstring description and examples
    examples = [example.description for example in docstring.examples]
    desc = (docstring.short_description, docstring.long_description, examples)

    return args_list, ret, desc, injected_args


def make_arg_preprocessor(args_list: list[tuple[str, type, str | None, str | None]]) -> Callable[[dict | None], tuple[list, dict]]:
    """
    Make a preprocessor function that converts the agent's input into a tool into *pargs, **kwargs for the tool function

    Args:
        args_list: A list of tuples (name, type, description, default) for each argument

    Returns:
        preprocessor (args: Any) -> (pargs, kwargs): 
    """

    def preprocessor(args: dict | None) -> tuple[list, dict]:
        """
        Argument preprocessor function for a tool function.

        Args:
            args (dict|None): The input arguments for the tool function. None if no arguments are provided.

        Returns:
            tuple[list, dict]: The positional arguments and keyword arguments for the tool. i.e. call `func(*pargs, **kwargs)`
        """
        if not isinstance(args, dict) and args is not None:
            raise TypeError(f"_input_ must be a dictionary or None. Got {type(args)}")

        # zero argument case
        if len(args_list) == 0:
            assert args is None, f"Expected no arguments, got {args}"
            return [], {}

        # general case, arguments wrapped in json. need to determine which need to be deserialized into structured types
        # TODO: this doesn't respect if a function signature has position-only vs keyword-only arguments. need to update get_tool_signature to include that information
        pargs = []
        kwargs = {}
        for arg_name, arg_type, _, _ in args_list:

            # potentially a default arg that is already provided.
            # if not, calling the tool will fail
            if arg_name not in args:
                continue

            # primitive types can be passed directly
            if is_primitive_type(arg_type):
                kwargs[arg_name] = args[arg_name]
                continue

            # structured types are deserialized from a dict into the dataclass/pydantic model
            if is_structured_type(arg_type):
                kwargs[arg_name] = construct_structured_type(arg_type, args[arg_name])
                continue

            raise ValueError(f"Unsupported structured type {arg_type}")

        return pargs, kwargs

    return preprocessor


def is_primitive_type(arg_type: type) -> bool:
    """
    Check if a type is a primitive type

    Primitive types are `str`, `int`, `float`, `bool`, `list`, and `dict`
    Additionally list and dict may be parameterized with primitive types. e.g. `list[str]`, `dict[str, int]`
    Lastly unions are considered primitive if all of their arguments are primitive types. e.g. `str | int`
    """

    # simplest case
    if arg_type in (str, int, float, bool, list, dict):
        return True

    # list or dict parameterized with primitive types
    if get_origin(arg_type) in (list, dict):
        return all(is_primitive_type(a) for a in get_type_args(arg_type))

    # union of primitive types
    if get_origin(arg_type) is Union or isinstance(arg_type, UnionType):
        return all(is_primitive_type(a) for a in get_type_args(arg_type))

    return False


def collect_tools_from_object(obj: object):
    result = []
    for item_name, item in inspect.getmembers(obj, predicate=lambda member: inspect.ismethod(member) or inspect.isfunction(member)):
        if is_tool(item):
            result.append(item)
    return result


def make_tool_dict(tools: list[Callable | type | Any]) -> dict[str, Callable]:
    """
    Create a dictionary of tools from a list of tool functions.

    Tries to use the '_name' attribute of the @tool function, otherwise uses the function name.

    Args:
        tools (list[Callable|type|Any]): A list of tool functions, or tool classes, or instances of tool classes.

    Returns:
        dict[str, Callable]: A dictionary of tools. Class methods of class tools are included as separate functions.
    """
    tool_dict = {}
    for tool in tools:
        # If the tool is actually a class and not an instance or function, instantiate it
        if isinstance(tool, type):
            tool = tool()
        if is_tool(tool):
            name = getattr(tool, "_name", None)
            if name is None and isinstance(tool, Agent):
                name = tool.__class__.__name__
            tool_dict[name] = tool

        # add each method to the tool dictionary under the name 'class_name.method_name'
        methods = inspect.getmembers(
            tool,
            predicate=lambda member: inspect.ismethod(member) or inspect.isfunction(member)
        )
        for _, method in methods:
            if is_tool(method):
                if isinstance(tool, type):
                    cls_name = getattr(tool, "_name", None) or tool.__name__
                else:
                    cls_name = getattr(tool, "_name", None) or tool.__class__.__name__
                method_name = getattr(method, "_name", None) or getattr(method, "__name__")
                method_name = f"{cls_name}.{method_name}"
                if method_name in tool_dict and method is not tool_dict[method_name]:
                    raise ValueError(f"Tool name '{method_name}' is already in use")
                tool_dict[method_name] = method
    return tool_dict


def get_tool_names(tools: list[Callable | type | Any]):
    res = list(make_tool_dict(tools).keys())
    return res


def test():
    from archytas.tools import ask_user, datetime_tool, timestamp
    from archytas.demo_tools import fib_n, example_tool, calculator, Jackpot

    for t in [
        ask_user,
        datetime_tool,
        timestamp,
        fib_n,
        example_tool,
        calculator,
        Jackpot,
    ]:
        print(get_tool_prompt_description(t))
        print()


if __name__ == "__main__":
    test()
