from types import GenericAlias, UnionType, NoneType, EllipsisType
from typing import Any, Optional, Union, List, Dict, Tuple, Iterable, get_origin, get_args, overload
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


import pdb



class NotProvided:
    """sentinel for optional parameters"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# instance of the singleton
notprovided = NotProvided()


@dataclass(frozen=True)
class NormalizedType(ABC):
    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool: ...

# Too much of a hassle to make this one a dataclass since we need to flatten nested Union_t types
class Union_t(NormalizedType):
    # Union_t can take either an Iterable[NormalizedType] or multiple NormalizedType as arguments
    @overload
    def __init__(self, types: Iterable[NormalizedType]): ...
    @overload
    def __init__(self, *types: NormalizedType): ...
    def __init__(self, types: Iterable[NormalizedType]|NormalizedType, *args: NormalizedType):

        # convert args to a single set of NormalizedType
        if isinstance(types, NormalizedType):
            _types = {types, *args}
        else:
            assert len(args) == 0, "Union_t can only take one argument if it is an Iterable[NormalizedType]"
            _types = set(types)

        # flatten any nested Union_t types into a single layer
        while any(isinstance(t, Union_t) for t in _types):
            new_types = set()
            for t in _types:
                if isinstance(t, Union_t):
                    new_types.update(t.types)
                else:
                    new_types.add(t)
            _types = new_types
        
        self.types = frozenset(_types)

    def __str__(self) -> str:
        return ' | '.join(str(t) for t in self.types)
    
    def __repr__(self) -> str:
        return f"Union_t({set(self.types)})"
    
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        if isinstance(other, Union_t):
            return all(any(t.matches(o, strict) for o in other.types) for t in self.types)
        
        #TODO: probably actually remove this case. matches are looking for docstring matches signature, 
        #      not a single type matches one of the options
        return any(t.matches(other, strict) for t in self.types)

    def __eq__(self, other):
        return isinstance(other, Union_t) and self.types == other.types
    
    def __hash__(self):
        return hash(self.types)
    
@dataclass(frozen=True)
class List_t(NormalizedType):
    element_type: NormalizedType | NotProvided = notprovided
    
    def __str__(self) -> str:
        if isinstance(self.element_type, NotProvided):
            return 'list'
        return f'list[{self.element_type}]'
    
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        # other is not a List_t
        if not isinstance(other, List_t):
            return False
        
        # non-strict means don't compare parameters e.g. List[int] can match List
        if not strict:
            return True

        # both are not parametrized
        if self.element_type is notprovided and other.element_type is notprovided:
            return True
        
        # one is parametrized and the other is not, means no match since strict=True
        if self.element_type is notprovided or other.element_type is notprovided:
            return False
        
        # just making the type checker happy. This should always be true
        assert not isinstance(self.element_type, NotProvided) and not isinstance(other.element_type, NotProvided)

        # match if parametrized types match
        return self.element_type.matches(other.element_type, strict)
    
@dataclass(frozen=True)
class Tuple_t(NormalizedType):
    component_types: tuple[NormalizedType, ...] | tuple[NormalizedType, EllipsisType] | NotProvided = notprovided

    def __str__(self) -> str:
        if isinstance(self.component_types, NotProvided):
            return 'tuple'
        return f'tuple[{", ".join(str(t) for t in self.component_types)}]'
    
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        # other is not a Tuple_t
        if not isinstance(other, Tuple_t):
            return False

        # non-strict means don't compare parameters e.g. Tuple[int, str] can match Tuple
        if not strict:
            return True

        # both are not parametrized
        if self.component_types is notprovided and other.component_types is notprovided:
            return True

        # one is parametrized and the other is not, means no match since strict=True
        if self.component_types is notprovided or other.component_types is notprovided:
            return False

        # just making the type checker happy. This should always be true
        assert not isinstance(self.component_types, NotProvided) and not isinstance(other.component_types, NotProvided)

        # match if parametrized types match
        if len(self.component_types) != len(other.component_types):
            return False

        return all(a == b for a, b in zip(self.component_types, other.component_types))

@dataclass(frozen=True)
class Dict_t(NormalizedType):
    #TODO: tbd if this is the best way to store key_type and value_type
    #      e.g. could be params: tuple[NormalizedType, NormalizedType] | NotProvided
    key_type: NormalizedType | NotProvided = notprovided
    value_type: NormalizedType | NotProvided = notprovided
    def __post_init__(self):
        # TODO: if key_type is not provided, value_type must also not be provided
        pdb.set_trace()
        ...

@dataclass(frozen=True)
class Int_t(NormalizedType):
    def __str__(self) -> str:
        return 'int'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool: 
        return isinstance(other, Int_t)

@dataclass(frozen=True)
class Float_t(NormalizedType):
    def __str__(self) -> str:
        return 'float'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool: 
        return isinstance(other, Float_t)

@dataclass(frozen=True)
class Str_t(NormalizedType):
    def __str__(self) -> str:
        return 'str'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return isinstance(other, Str_t)

@dataclass(frozen=True)
class Bool_t(NormalizedType):
    def __str__(self) -> str:
        return 'bool'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return isinstance(other, Bool_t)

@dataclass(frozen=True)
class None_t(NormalizedType):
    def __str__(self) -> str:
        return 'None'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return isinstance(other, None_t)


@dataclass(frozen=True)
class Literal_t(NormalizedType): ...

@dataclass(frozen=True)
class Dataclass_t(NormalizedType): ...

@dataclass(frozen=True)
class PydanticModel_t(NormalizedType): ...



# @dataclass(frozen=True)
# class Object_t(NormalizedType):
#     def __post_init__(self):
#         logger.warning("Object_t should not be used as a type for @tools")
#     def __str__(self) -> str:
#         return 'object'
#     def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
#         return True


def is_optional(annotation):
    """Check if the annotation is typing.Optional[T]"""
    return get_origin(annotation) is Union and type(None) in get_args(annotation)

def normalize_type(t: Any) -> NormalizedType:

    ### main error cases ###

    # Optional without a parameter doesn't make sense
    if t is Optional:
        raise ValueError("Underspecified type for tool. Optional should contain a type, e.g. Optional[int]. Note that Optional[T] is equivalent to Union[T, None]")
    
    # Object_t shouldn't be used! It's not really useful for agent instructions
    if t is object:
        # return Object_t()
        raise ValueError("Underspecified type annotation for tool. `object` does not provide enough information for the agent to create an instance of the argument to the tool.")


    # Optional[a]
    if is_optional(t):
        return Union_t(normalize_type(a) for a in get_args(t))
    
    # a | b
    if isinstance(t, UnionType):
        return Union_t(normalize_type(a) for a in t.__args__)
    # Union[a, b]
    if get_origin(t) is Union:
        return Union_t(normalize_type(a) for a in get_args(t))

    if t is str:
        return Str_t()
    if t is int:
        return Int_t()
    if t is float:
        return Float_t()
    if t is bool:
        return Bool_t()
    if t is None or t is NoneType:
        return None_t()
    
    #Tuple[a, b, c], Tuple, tuple
    if t is tuple or t is Tuple:
        return Tuple_t()
    if get_origin(t) is tuple:
        return Tuple_t(tuple(normalize_type(a) for a in get_args(t)))
    
    #List[a], List, list
    #Dict[a, b], Dict, dict


    pdb.set_trace()
    ...
    raise NotImplementedError("TODO")


#TODO: the only real way to make this safe is to manually parse the types ourselves
#      with `eval`, even if you restrict __builtins__ / etc., the user can still get access 
#      to them through the the function the docstring is attached to, e.g. `func.__globals__`
#      TBD on how necessary since generally we should be in control of the docstrings
#      and if we're not, the user would be able to execute arbitrary code anyways
# Other possible long-term solution is to just not typecheck. So long as the argnames match up
#      in the docstring vs in the signature, we should just assume it's a valid match
def evaluate_type_str(type_str: str, globals: dict[str, Any]) -> NormalizedType:
    try:
        t = eval(type_str, globals)
    except Exception as e:
        raise ValueError(f"Could not evaluate type string '{type_str}'") from e

    return normalize_type(t)

# Some debug testing
if __name__ == '__main__':
    from .test import my_fn, B
    from docstring_parser import parse as parse_docstring
    import inspect

    # func = my_fn
    func = B.my_fn

    assert func.__doc__ is not None, f"Function '{func.__name__}' has no docstring. All tools must have a docstring that matches the function signature."
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # Extract argument information from the docstring
    docstring_args: dict[str, tuple[NormalizedType, str, str]] = {
        arg.arg_name: (
            evaluate_type_str(arg.type_name or '', func.__globals__), 
            arg.description or '', 
            arg.default or ''
        )
        for arg in docstring.params
    }

    pdb.set_trace()
    ...