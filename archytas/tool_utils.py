import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback; traceback.install(show_locals=True)
from textwrap import indent
from typing import Callable, Any


import pdb


#TODO: separate system tools from library tools from user tools
#      ideally system tools will have no library dependencies

#TODO: parse extra long manual page from doc string. man_page can be seen by calling man <tool_name>
# man_page:str|None=None
# wrapper._man_page = man_page


def tool(*, name:str|None=None):
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
    def decorator(func:Callable):
        # check that the decorator is being applied to a function
        if not inspect.isfunction(func):
            raise TypeError(f"tool decorator can only be applied to functions or classes. Got {func} of type {type(func)}")
        
        # determine if function, or class method, and return the appropriate wrapper
        if is_class_method(func):
            return make_method_tool_wrapper(func, name)
        else:
            return make_func_tool_wrapper(func, name)

    return decorator


def toolset(*, name:str|None=None):
    """
    Decorator used to convert a class into a toolset for ReAct agents to use.

    Usage:
    ```
        @toolset()
        class MyToolset:
            '''
            description of the toolset
            '''

            def __init__(self, arg1, arg2, ...):
                # initialize the toolset, set up state, etc.
                # if has no required arguments, can pass class constructor to agent's list of tools
                # if has required arguments, must pass an instance to agent's list of tools

            @tool()
            def my_tool(self, arg:type) -> type:
                '''
                Short description of the tool method

                Long description of the tool method

                Args:
                    arg (type): Description of the argument

                Returns:
                    type: Description of the return value

                Examples:
                    optional description of the example
                    >>> my_tool(arg)
                    result
                '''

            @tool()
            def my_other_tool(self, arg:type) -> type:
                '''<tool docstring>'''
    ```
    """
    def decorator(cls:type):
        if not inspect.isclass(cls):
            raise TypeError(f"toolset decorator can only be applied to classes. Got {cls} of type {type(cls)}")

        #basically just add metadata, and return the class
        #metadata should include a list of the class's tool methods

        #get the list of @tool methods in the class
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        methods = [method for name, method in methods if is_tool(method)]

        # get the class docstring description
        docstring = inspect.getdoc(cls)

        class wrapper:
            def __init__(self, *args, **kwargs):
                # create an instance of the class
                self._instance = cls(*args, **kwargs)

                # mark this as a class tool instance
                self._is_class_tool_instance = True


        # attach the metadata to the class
        wrapper._name = name if name else cls.__name__
        wrapper._is_class_tool = True
        wrapper._docstring = docstring
        wrapper._tool_methods = methods
        wrapper._cls = cls
        
        return wrapper
    
    return decorator



def make_func_tool_wrapper(func:Callable, name:str|None=None):
    def wrapper(args:dict|list|str|int|float|bool|None):
        """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""
        
        #TODO: make this look at _call_type rather than isinstance to determine what to do
        #      single argument functions that take a dict vs mutli-argument functions will both have a dict, but they need to be called differently func(args) vs func(**args)
        if isinstance(args, dict):
            result = func(**args)
        elif isinstance(args, list):
            result = func(*args)
        elif isinstance(args, (str, int, float, bool)):
            result = func(args)
        elif args is None:
            result = func()
        else:
            raise TypeError(f"args must be a valid json object type (dict, list, str, int, float, bool, or None). Got {type(args)}")

        #convert the result to a string if it is not already a string
        if not isinstance(result, str):
            result = str(result)

        return result
    
    #attach usage description to the wrapper function
    args_list, ret, desc = get_tool_signature(func)

    wrapper._name = name if name else func.__name__
    wrapper._is_function_tool = True
    wrapper._args_list = args_list
    wrapper._ret = ret
    wrapper._desc = desc
    wrapper._func = func

    return wrapper

def make_method_tool_wrapper(func:Callable, name:str|None=None):
    def wrapper(self, args:dict|list|str|int|float|bool|None):
        """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""
        
        if isinstance(args, dict):
            result = func(self, **args)
        elif isinstance(args, list):
            result = func(self, *args)
        elif isinstance(args, (str, int, float, bool)):
            result = func(self, args)
        elif args is None:
            result = func(self)
        else:
            raise TypeError(f"args must be a valid json object type (dict, list, str, int, float, bool, or None). Got {type(args)}")

        #convert the result to a string if it is not already a string
        if not isinstance(result, str):
            result = str(result)

        return result
    
    #attach usage description to the wrapper function
    args_list, ret, desc = get_tool_signature(func)

    wrapper._name = name if name else func.__name__
    wrapper._is_method_tool = True
    wrapper._args_list = args_list
    wrapper._ret = ret
    wrapper._desc = desc
    wrapper._func = func

    return wrapper


def unwrap_tool(obj:Callable|type|Any) -> Callable|type|Any:
    """Unwrap a tool, toolset, or toolset instance"""
    assert is_tool(obj), f"unwrap can only be used on function tools, method tools, class tools, or class tool instances. Got {obj}"
    if hasattr(obj, '_instance'):
        return _unwrap_class_instance(obj._instance)
    if hasattr(obj, '_cls'):
        return _unwrap_class(obj._cls)
    if hasattr(obj, '_func'):
        return obj._func

    raise TypeError(f'unwrap could not unwrap {obj} of type {type(obj)}')

def _unwrap_class(cls:type) -> type:
    """
    unwrap a class tool (including all methods annotated with type)

    returns an identical class, as if it had never been wrapped
    """

    # Create a new class with the same name, inheriting from the original class
    unwrapped_cls = type(cls.__name__, (cls,), {})

    # Copy over the class attributes (unwrapping any wrapped methods)
    for name, attr in cls.__dict__.items():
        # Check if the attribute is a wrapped method
        if hasattr(attr, '_is_method_tool'):
            # Replace the attribute with the original unwrapped version
            setattr(unwrapped_cls, name, attr._func)
        else:
            # If not a wrapped method, copy the attribute to the new class
            assert not is_tool(attr), f"INTERNAL ERROR: Unwrapped class {cls.__name__} still has a tool {name} attached. This should not happen."
            try:
                setattr(unwrapped_cls, name, attr)
            except AttributeError as e:
                #skip not writable attributes
                if 'not writable' in str(e): ...
                #raise other errors
                else: raise

    return unwrapped_cls



def _unwrap_class_instance(instance:Any) -> Any:
    """
    unwrap a class tool instance into an instance of the underlying (unwrapped) class

    returns an identical instance, as if it was constructed from a class that had never been wrapped

    Args:
        instance (Any): An instance of a class tool. Note that this must be the _instance attribute of a class tool instance, not the class tool instance itself (e.g. you should call this with obj._instance).
    """

    # Unwrap the class of the given instance
    unwrapped_cls = _unwrap_class(instance.__class__)

    # Create a new instance of the original class without calling its constructor
    unwrapped_instance = object.__new__(unwrapped_cls)

    # Copy the instance's attributes to the new unwrapped instance
    for name, attr in instance.__dict__.items():
        setattr(unwrapped_instance, name, attr)

    return unwrapped_instance



def is_class_method(func:Callable) -> bool:
    """checks if a function is part of a class, or a standalone function"""
    assert inspect.isfunction(func), f"is_class_method can only be used on functions. Got {func}"
    return func.__qualname__ != func.__name__

def is_tool(obj:Callable|type) -> bool:
    """checks if an object is a tool function, tool method, tool class, or an instance of a class tool"""
    return hasattr(obj, '_is_function_tool') or hasattr(obj, '_is_method_tool') or hasattr(obj, '_is_class_tool') or hasattr(obj, '_is_class_tool_instance')
        

def get_tool_name(obj:Callable|type) -> str:
    """Get the name of the tool, either from the @tool _name field, or the __name__ attribute"""
    assert is_tool(obj), f"get_tool_name can only be used on decorated @tools. Got {obj}"
    return getattr(obj, '_name')

def get_tool_names(obj:Callable|type) -> list[str]:
    """
    Get the tool name, or all method names if tool is a class tool
    """
    assert is_tool(obj), f"get_tool_name can only be used on decorated @tools. Got {obj}"
    
    if hasattr(obj, '_is_class_tool') or hasattr(obj, '_is_class_tool_instance'):
        cls_name = get_tool_name(obj)

        #construct names as cls_name.method_name
        names = [f"{cls_name}.{get_tool_name(method)}" for method in obj._tool_methods]

        return names
    
    #otherwise, just return the name of the tool
    return [get_tool_name(obj)]



def get_tool_prompt_description(obj:Callable|type|Any):
    if hasattr(obj, '_is_class_tool') or hasattr(obj, '_is_class_tool_instance'):
        return get_tool_class_prompt_description(obj)
    if hasattr(obj, '_is_function_tool') or hasattr(obj, '_is_method_tool'):
        return get_tool_func_prompt_description(obj)

    raise TypeError(f"get_tool_prompt_description can only be used on @tools. Got {obj}")

def get_tool_func_prompt_description(func:Callable):
    assert is_tool(func), f"Function {func.__name__} does not have the @tool decorator attached"
    assert inspect.isfunction(func), f"get_tool_func_prompt_description can only be used on functions. Got {func}"

    #get the list of arguments
    
    args_list = func._args_list
    ret_name, ret_type, ret_description = func._ret
    short_desc, long_desc, examples = func._desc

    
    chunks = []
    tab = "    "

    ############### NAME/DESCRIPTION ###############
    chunks.append(f"{func._name}:\n")
    if short_desc:
        chunks.append(f"{tab}{short_desc}\n\n")
    if long_desc:
        chunks.append(f"{indent(long_desc, tab)}\n\n")
    
    #################### INPUT ####################
    chunks.append('    _input_: ')
    
    if len(args_list) == 0:
        chunks.append("None")
    

    elif len(args_list) == 1 and args_list[0][1] is not dict:
        arg_name, arg_type, arg_desc, arg_default = args_list[0]
        chunks.append(f"({arg_type.__name__}")
        if arg_default:
            chunks.append(f", optional")
        chunks.append(f") {arg_desc}")
    
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
    #TODO: examples need to be parsed...


    return ''.join(chunks)


def get_tool_class_prompt_description(cls:type):
    """
    Get the prompt description for a class tool, including all of its tool methods
    
    class description is as follows:
      class_name (class):
          full unmodified class docstring
          methods:
              <get_tool_func_prompt_description for each method>

    Args:
        cls (type|Any): The class tool to get the description for, or an instance of a class tool
    
    Returns:
        str: The prompt description for the class tool
    """
    assert hasattr(cls, '_is_class_tool') or hasattr(cls, '_is_class_tool_instance'), f"class or instance {cls} does not have the @tool decorator attached"


    chunks = []
    tab = "    "

    ############### NAME/DESCRIPTION ###############
    chunks.append(f"{cls._name} (class):\n")
    if cls._docstring:
        chunks.append(f"{indent(cls._docstring, tab)}\n\n")

    #################### METHODS ####################
    chunks.append(f"{tab}methods:\n")
    for method in cls._tool_methods:
        method_str = get_tool_func_prompt_description(method)
        chunks.append(f"{indent(method_str, tab*2)}\n\n")

    #strip trailing whitespace
    return ''.join(chunks).rstrip()




def get_tool_signature(func:Callable) -> tuple[list[tuple[str, type, str|None, str|None]], tuple[str|None, type|None, str|None], tuple[str, str|None, list[str]]]:
    """
    Check that the docstring and function signature match for a tool function, and return all function information.

    Args:
        func (function): The function to check and extract information from
        ignore_self (bool): If True, ignore the first argument of the function if it is named 'self'

    Returns:
        args_list: A list of tuples (name, type, description, default) for each argument
        ret: A tuple (name, type, description) for the return value
        desc: A tuple (short_description, long_description, examples) from the docstring for the function
    """
    assert inspect.isfunction(func), f"get_tool_signature can only be used on functions. Got {func}"

    # get the function signature from the function and the docstring
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # determine if self needs to be ignored
    ignore_self: bool = is_class_method(func)

    # Extract argument information from the docstring
    docstring_args = {arg.arg_name: (arg.type_name, arg.description, arg.default) for arg in docstring.params}

    # Extract argument information from the signature (ignore self from class methods)
    signature_args = {k: v.annotation for i, (k, v) in enumerate(signature.parameters.items()) if not (i == 0 and ignore_self and k == 'self')}

    # Check if the docstring argument names match the signature argument names
    if set(docstring_args.keys()) != set(signature_args.keys()):
        raise ValueError(f"Docstring argument names do not match function signature argument names for function '{func.__name__}'")

    # Check if the docstring argument types match the signature argument types
    for arg_name, arg_type in signature_args.items():
        docstring_arg_type, _, _ = docstring_args[arg_name]
        if arg_type.__name__ != docstring_arg_type:
            raise ValueError(f"Docstring type '{docstring_arg_type}' does not match function signature type '{arg_type.__name__}' for argument '{arg_name}' for function '{func.__name__}'")

    # Generate a list of tuples (name, type, description, default) for each argument
    #TODO: use the signature to determine if an argument is optional or not
    args_list = [(arg_name, signature_args[arg_name], arg_desc, arg_default) for arg_name, (arg_type, arg_desc, arg_default) in docstring_args.items()]

    # get the return type and description
    signature_ret_type = signature.return_annotation

    # # if docstring.returns:
    # if signature_ret_type == inspect.Signature.empty and docstring.returns is None:
    #     ...
    try:
        docstring_ret_type = docstring.returns.type_name
    except AttributeError:
        docstring_ret_type = '_empty'
    if signature_ret_type.__name__ != docstring_ret_type:
        raise ValueError(f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{signature_ret_type.__name__}' for function '{func.__name__}'")
    
    # get the return type and description
    if docstring.returns is None:
        ret = (None, None, None)
    else:
        ret = (docstring.returns.return_name, signature_ret_type, docstring.returns.description)

    # get the docstring description and examples
    examples = [example.description for example in docstring.examples]
    desc = (docstring.short_description, docstring.long_description, examples)
    
    return args_list, ret, desc



def make_tool_dict(tools:list[Callable|type|Any]) -> dict[str, Callable]:
    """
    Create a dictionary of tools from a list of tool functions.

    Tries to use the '_name' attribute of the @tool function, otherwise uses the function name.

    Args:
        tools (list[Callable|type|Any]): A list of tool functions, or tool classes, or instances of tool classes.

    Returns:
        dict[str, Callable]: A dictionary of tools. Class methods of class tools are included as separate functions.
    """
    #TODO: extract methods from any class tools
    tool_dict = {}
    for tool in tools:
        assert is_tool(tool), f"make_tool_dict can only be used on tools. Got {tool}"
        name = getattr(tool, '_name')
        
        
        if hasattr(tool, '_is_function_tool'):
            if name in tool_dict:
                raise ValueError(f"Tool name '{name}' is already in use")
            tool_dict[name] = tool
            continue
        
        if hasattr(tool, '_is_method_tool'):
            #TODO: not sure if methods should be allowed since they need usually need a class instance...
            raise NotImplementedError("Free-floating tool methods are not yet supported")
            pdb.set_trace()
        

        # collect methods from @toolset. handle if instance of class or class itself
        assert hasattr(tool, '_is_class_tool'), f"Tool {tool} is not a function, method, or class tool"
        if hasattr(tool, '_is_class_tool_instance'):
            instance = tool
        else:
            instance = tool()

        # add each method to the tool dictionary under the name 'class_name.method_name'
        methods = inspect.getmembers(instance._instance, predicate=inspect.ismethod)
        for _, method in methods:
            if not hasattr(method, '_name'):
                continue
            method_name = f'{name}.{method._name}'
            if method_name in tool_dict:
                raise ValueError(f"Tool name '{method_name}' is already in use")
            tool_dict[method_name] = method


    return tool_dict




def test():
    from archytas.tools import ask_user, datetime_tool, timestamp
    from archytas.demo_tools import fib_n, example_tool, calculator, Jackpot

    for t in [ask_user, datetime_tool, timestamp, fib_n, example_tool, calculator, Jackpot]:
        print(get_tool_prompt_description(t))
        print()



if __name__ == '__main__':
    test()