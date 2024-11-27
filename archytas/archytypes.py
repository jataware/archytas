from types import UnionType, NoneType, EllipsisType
from typing import Any, Optional, Union, Literal, List, Dict, Tuple, Iterable, get_origin, get_args, overload, TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import DataclassInstance  # only available to type checkers
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)





class NotProvided:
    """sentinel for optional parameters"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self):
        return 'NotProvided'
    def __str__(self):
        return 'NotProvided'
    def __eq__(self, other):
        return isinstance(other, NotProvided)
    def __hash__(self):
        return hash(NotProvided)

# instance of the singleton
notprovided = NotProvided()


@dataclass(frozen=True)
class NormalizedType(ABC):
    @abstractmethod
    def __str__(self) -> str: ...
    @classmethod
    def new(cls, value: Any) -> Any:
        """Primitive types can use .new() to create an instance of the type"""
        raise TypeError(f"Type Error: Cannot construct Type `{cls.__name__}.new()`")

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
        return f"Union_t({set(self.types)})" # wrap with set() to print it nicer
    
    def __eq__(self, other):
        if not isinstance(other, Union_t):
            return NotImplemented
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


@dataclass(frozen=True)
class Tuple_t(NormalizedType):
    component_types: tuple[NormalizedType, ...] | tuple[NormalizedType, EllipsisType] | NotProvided = notprovided

    def __str__(self) -> str:
        if isinstance(self.component_types, NotProvided):
            return 'tuple'
        return f'tuple[{", ".join(str(t) for t in self.component_types)}]'


@dataclass(frozen=True)
class Dict_t(NormalizedType):
    #TODO: tbd if this is the best way to store key_type and value_type
    #      e.g. could be params: tuple[NormalizedType, NormalizedType] | NotProvided
    key_type: NormalizedType | NotProvided = notprovided
    value_type: NormalizedType | NotProvided = notprovided

    def __str__(self) -> str:
        if isinstance(self.key_type, NotProvided) and isinstance(self.value_type, NotProvided):
            return 'dict'
        if isinstance(self.key_type, NotProvided):
            return f'dict[?, {self.value_type}]'
        if isinstance(self.value_type, NotProvided):
            return f'dict[{self.key_type}, ?]'
        return f'dict[{self.key_type}, {self.value_type}]'


@dataclass(frozen=True)
class Int_t(NormalizedType):
    def __str__(self) -> str:
        return 'int'
    @classmethod
    def new(cls, value: Any) -> int:
        if not isinstance(value, (int, float)) or value != int(value):
            raise TypeError(f"Expected an int, got {value}")
        return int(value)


@dataclass(frozen=True)
class Float_t(NormalizedType):
    def __str__(self) -> str:
        return 'float'
    @classmethod
    def new(cls, value: Any) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected a float, got {value}")
        return float(value)


@dataclass(frozen=True)
class Str_t(NormalizedType):
    def __str__(self) -> str:
        return 'str'
    @classmethod
    def new(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Expected a string, got {value}")
        return value


@dataclass(frozen=True)
class Bool_t(NormalizedType):
    def __str__(self) -> str:
        return 'bool'
    @classmethod
    def new(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"Expected a bool, got {value}")
        return bool(value)


@dataclass(frozen=True)
class None_t(NormalizedType):
    def __str__(self) -> str:
        return 'None'
    @classmethod
    def new(cls, value: Any) -> None:
        if value is not None:
            raise TypeError(f"Expected None, got {value}")
        return None


@dataclass(frozen=True)
class Literal_t(NormalizedType): ... #TODO:


@dataclass(frozen=True)
class Dataclass_t(NormalizedType):
    cls: 'type[DataclassInstance]'
    def __post_init__(self):
        # TODO: check/normalize members?
        ...
    def __str__(self) -> str:
        return self.cls.__name__


@dataclass(frozen=True)
class PydanticModel_t(NormalizedType):
    cls: 'type[BaseModel]'
    def __post_init__(self):
        # TODO: check/normalize members?
        ...
    def __str__(self) -> str:
        return self.cls.__name__


# Any should compare equal to all types
@dataclass(frozen=True)
class Any_t(NormalizedType):
    def __str__(self) -> str:
        return 'Any'
    def __eq__(self, other):
        return True
    def __req__(self, other):
        return True


def is_optional(annotation):
    """Check if the annotation is typing.Optional[T]"""
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def normalize_type(t: Any) -> NormalizedType:
    """
    Convert a type annotation to a NormalizedType
    Supports old and new style type annotations
    e.g. `List[T]` vs `list[T]`, `Optional[T]` vs `T | None`, `Union[A, B]` vs `A | B`, etc.

    The following types are supported:
    - str, int, float, bool, None
    - list, dict, tuple
    - Optional, Union, (WIP) Literal
    - dataclasses
    - pydantic models

    Args:
        t (Any): the type annotation

    Returns:
        NormalizedType: the normalized type object
    """

    #### main error cases ####

    # Optional without a parameter doesn't make sense
    if t is Optional:
        raise ValueError("Underspecified type for tool. Optional should contain a type, e.g. Optional[int]. Note that Optional[T] is equivalent to Union[T, None]")
    
    # Object_t shouldn't be used! It's not really useful for agent instructions
    if t is object:
        # return Object_t()
        raise ValueError("Underspecified type annotation for tool. `object` does not provide enough information for the agent to create an instance of the argument to the tool.")

    # Any is also not useful for the same reasons as object
    if t is Any:
        raise ValueError("Underspecified type annotation for tool. `Any` does not provide enough information for the agent to create an instance of the argument to the tool.")


    # Primitive types
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
    

    # Optional[a]
    if is_optional(t):
        return Union_t(normalize_type(a) for a in get_args(t))
    
    # a | b
    if isinstance(t, UnionType):
        return Union_t(normalize_type(a) for a in t.__args__)
    # Union[a, b]
    if get_origin(t) is Union:
        return Union_t(normalize_type(a) for a in get_args(t))

    # Literal[a, b, c]
    if get_origin(t) is Literal:
        raise NotImplementedError("Literal type annotation is not yet supported")
        # return Literal_t()
    
    #Tuple[a, b, c], Tuple, tuple
    if t is tuple or t is Tuple:
        return Tuple_t()
    if get_origin(t) is tuple:
        return Tuple_t(tuple(normalize_type(a) for a in get_args(t) if a != ...))
    
    #List[a], List, list
    if t is list or t is List:
        return List_t()
    if get_origin(t) is list:
        args = get_args(t)
        if len(args) != 1:
            raise ValueError(f"List type annotation should have exactly one argument, found {len(args)}")
        return List_t(normalize_type(args[0]))


    #Dict[a, b], Dict, dict
    if t is dict or t is Dict:
        return Dict_t()
    if get_origin(t) is dict:
        args = get_args(t)
        if len(args) != 2:
            raise ValueError(f"Dict type annotation should have exactly two arguments, found {len(args)}")
        return Dict_t(normalize_type(args[0]), normalize_type(args[1]))


    if isinstance(t, type) and is_dataclass(t): # is_dataclass also return True for dataclass instances, which we don't want to match
        #TODO: TBD if the inner types should be normalized...
        return Dataclass_t(t)

    if isinstance(t, type) and issubclass(t, BaseModel):
        #TODO: TBD if the inner types should be normalized...
        return PydanticModel_t(t)


    raise ValueError(f"Unsupported type to normalize: {t}")


#TODO: the only real way to make this safe is to manually parse the types ourselves
#      with `eval`, even if you restrict __builtins__ / etc., the user can still get access 
#      to them through the the function the docstring is attached to, e.g. `func.__globals__`
#      TBD on how necessary since generally we should be in control of the docstrings
#      and if we're not, the user would be able to execute arbitrary code anyways
# For now, the main mitigation is the devmode=False flag. Production code should always
#      have this set to False (forcing all docstring types to Any_t). When true, this
#      function will be called on the docstring's type annotations to normalize them.
def evaluate_type_str(type_str: str, globals: dict[str, Any], *, devmode: bool) -> NormalizedType:
    """
    Convert a string type annotation (i.e. from a docstring) into the actual normalized type object

    Args:
        type_str (str): the string type annotation
        globals (dict[str, Any]): the globals from the function that the annotation is attached to.
        devmode (bool): Only perform evaluation in devmode mode. If False (i.e. production), returns Any_t.

    Returns:
        NormalizedType: the normalized type object. If devmode is False, this will always be Any_t.
    """
    # Short circuit for non-devmode (e.g. production)
    if not devmode:
        return Any_t()

    # evaluate the type from the docstring in the context of the function that defined it
    try:
        t = eval(type_str, globals)
    except Exception as e:
        raise ValueError(f"Could not evaluate type string '{type_str}'") from e

    return normalize_type(t)



def is_primitive_type(arg_type: NormalizedType) -> bool:
    """
    Check if a type is a primitive type

    Primitive types are `str`, `int`, `float`, `bool`, `list`, and `dict`
    Additionally list and dict may be parameterized with primitive types. e.g. `list[str]`, `dict[str, int]`
    Lastly unions are considered primitive if all of their arguments are primitive types. e.g. `str | int`
    """

    # simplest case
    if isinstance(arg_type, (Str_t, Int_t, Float_t, Bool_t, None_t)):
        return True

    # for list, dict, tuple, and union: any inner types must be primitive
    if isinstance(arg_type, List_t):
        if isinstance(arg_type.element_type, NotProvided):
            return True
        return is_primitive_type(arg_type.element_type)

    if isinstance(arg_type, Tuple_t):
        if isinstance(arg_type.component_types, NotProvided):
            return True
        return all(is_primitive_type(t) for t in arg_type.component_types if t != ...)

    if isinstance(arg_type, Dict_t):
        if isinstance(arg_type.key_type, NotProvided) and isinstance(arg_type.value_type, NotProvided):
            return True
        if not isinstance(arg_type.key_type, NotProvided) and not is_primitive_type(arg_type.key_type):
            return False
        if not isinstance(arg_type.value_type, NotProvided) and not is_primitive_type(arg_type.value_type):
            return False
        return True

    if isinstance(arg_type, Union_t):
        return all(is_primitive_type(t) for t in arg_type.types)

    return False


def is_structured_type(arg_type: NormalizedType) -> bool:
    """Check if a type is a structured type like a dataclass or pydantic model"""
    if isinstance(arg_type, Dataclass_t) or isinstance(arg_type, PydanticModel_t):
        return True

    if isinstance(arg_type, Union_t):
        return any(is_structured_type(t) for t in arg_type.types)

    if isinstance(arg_type, List_t):
        if not isinstance(arg_type.element_type, NotProvided):
            return is_structured_type(arg_type.element_type)

    if isinstance(arg_type, Tuple_t):
        if not isinstance(arg_type.component_types, NotProvided):
            return any(is_structured_type(t) for t in arg_type.component_types if t != ...)

    if isinstance(arg_type, Dict_t):
        if not isinstance(arg_type.key_type, NotProvided):
            return is_structured_type(arg_type.key_type)
        if not isinstance(arg_type.value_type, NotProvided):
            return is_structured_type(arg_type.value_type)

    return False
