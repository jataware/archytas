import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback

traceback.install(show_locals=True)
from textwrap import indent
from types import FunctionType
from typing import Callable, Any

from .agent import Agent

import logging
logger = logging.getLogger(__name__)

import pdb


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

INJECTION_MAPPING = {
    AgentRef: "agent",
    ToolNameRef: "tool_name",
    ToolFnRef: "raw_tool",
    LoopControllerRef: "loop_controller",
}

def toolset(*args, **kwargs):
    """
    A dummy decorator for backwards compatibility.
    Provides no funcitonality.
    Any class can now contain tools without a decorator.
    """
    logger.warning("Warning: The usage of the @toolset decorator is deprecated and the decorator will be removed in a future version.")
    def decorator(cls):
        return cls
    return decorator


def tool(*, name: str | None = None, autosummarize: bool = False):
    """
    Decorator to convert a function into a tool for ReAct agents to use.

    Usage:
    ```
        @tool()
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

    def decorator(func: Callable):
        # check that the decorator is being applied to a function
        if not inspect.isfunction(func):
            raise TypeError(
                f"tool decorator can only be applied to functions or classes. Got {func} of type {type(func)}"
            )

        # attach usage description to the wrapper function
        args_list, ret, desc, injections = get_tool_signature(func)

        func._name = name if name else func.__name__
        func._is_tool = True
        func.autosummarize = autosummarize

        async def run(
            args: tuple[object, dict | list | str | int | float | bool | None],
            tool_context: dict[str, object] = None,
            self_ref: object = None

        ):
            """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""
            # Initialise positional and keyword argument holders
            pargs = []
            kwargs = {}

            if self_ref:
                pargs.append(self_ref)

            # TODO: make this look at _call_type rather than isinstance to determine what to do
            #      single argument functions that take a dict vs multi-argument functions will both have a dict, but they need to be called differently func(args) vs func(**args)
            if args is None:
                pass
            elif len(args_list) == 1:
                pargs.append(args)
            elif isinstance(args, dict):
                kwargs.update(args)
            elif isinstance(args, list):
                pargs.extend(args)
            elif isinstance(args, (str, int, float, bool)):
                pargs.append(args)
            else:
                raise TypeError(
                    f"args must be a valid json object type (dict, list, str, int, float, bool, or None). Got {type(args)}"
                )

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
    return get_prompt_description(obj)


def get_prompt_description(obj: Callable | type | Any):

    # get the list of arguments
    chunks = []
    tab = "    "

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        args_list, ret, desc, injections = get_tool_signature(obj)
        args_list = args_list
        ret_name, ret_type, ret_description = ret
        short_desc, long_desc, examples = desc

        ############### NAME/DESCRIPTION ###############
        chunks.append(f"{obj._name}:\n")
        if short_desc:
            chunks.append(f"{tab}{short_desc}\n\n")
        if long_desc:
            chunks.append(f"{indent(long_desc, tab)}\n\n")

        #################### INPUT ####################
        chunks.append("    _input_: ")

        if len(args_list) == 0:
            chunks.append("None")

        # 1-argument case for simple types don't need to be wrapped in a json
        elif len(args_list) == 1 and args_list[0][1] in (str, int, float, bool):
            arg_name, arg_type, arg_desc, arg_default = args_list[0]
            chunks.append(f"({arg_type.__name__}")
            if arg_default:
                chunks.append(f", optional")
            chunks.append(f") {arg_desc}")

        # all other cases have arguments wrapped in a json
        else:
            chunks.append("a json object with the following fields:\n    {")
            for arg_name, arg_type, arg_desc, arg_default in args_list:
                chunks.append(f'\n        "{arg_name}": # ({arg_type.__name__}')
                if arg_default:
                    chunks.append(f", optional")
                chunks.append(f") {arg_desc}")
            chunks.append(f"\n{tab}}}")

        #################### OUTPUT ####################
        chunks.append("\n    _output_: ")
        if ret_type is None:
            chunks.append("None")
        else:
            chunks.append(f"({ret_type.__name__}) {ret_description}")

        ############### EXAMPLES ###############
        # TODO: examples need to be parsed...
    else:
        tool_methods = collect_tools_from_object(obj)
        if tool_methods:
            ############### NAME/DESCRIPTION ###############
            if inspect.isclass(obj):
                chunks.append(f"{obj.__name__} (class):\n")
            else:
                chunks.append(f"{obj.__class__.__name__} (class instance):\n")
            docstring = inspect.getdoc(obj)
            if docstring:
                chunks.append(f"{indent(docstring, tab)}\n\n")

            #################### METHODS ####################
            chunks.append(f"{tab}methods:\n")
            for method in tool_methods:
                method_str = get_prompt_description(method)
                chunks.append(f"{indent(method_str, tab*2)}\n\n")

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
    assert inspect.isfunction(func) or inspect.ismethod(func), f"get_tool_signature can only be used on functions or methods. Got {func}"

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
        if docstring_arg_type not in (arg_type.__name__, str(arg_type)):
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
    if signature_ret_type.__name__ != docstring_ret_type:
        raise ValueError(
            f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{signature_ret_type.__name__}' for function '{func.__name__}'"
        )

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
        methods = inspect.getmembers(tool, predicate=lambda member: inspect.ismethod(member) or inspect.isfunction(member))
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

def get_tool_names(tools:list[Callable | type | Any]):
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
