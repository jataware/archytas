from types import GenericAlias, UnionType, NoneType, EllipsisType
from typing import Any, Optional, Union, List, Dict, Tuple, Iterable, get_origin, get_args, overload
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from pydantic import BaseModel
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

        return False
        #TODO: probably actually remove this case. matches are looking for docstring matches signature, 
        #      not a single type matches one of the options
        # return any(t.matches(other, strict) for t in self.types)

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

        # strict matching of parameters
        return self.element_type == other.element_type

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

        # strict matching of parameters
        return self.component_types == other.component_types

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

    def matches(self, other, strict: bool = False) -> bool:
        # other is not a Dict_t
        if not isinstance(other, Dict_t):
            return False

        # non-strict means don't compare parameters e.g. Dict[int, str] can match Dict
        if not strict:
            return True

        # strict matching of parameters
        return self.key_type == other.key_type and self.value_type == other.value_type

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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import DataclassInstance  # only available to type checkers
@dataclass(frozen=True)
class Dataclass_t(NormalizedType):
    cls: 'type[DataclassInstance]'
    def __post_init__(self):
        # TODO: check/normalize members?
        ...
    def __str__(self) -> str:
        return self.cls.__name__
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return isinstance(other, Dataclass_t) and self.cls == other.cls

@dataclass(frozen=True)
class PydanticModel_t(NormalizedType):
    cls: 'type[BaseModel]'
    def __post_init__(self):
        # TODO: check/normalize members?
        ...
    def __str__(self) -> str:
        return self.cls.__name__
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return isinstance(other, PydanticModel_t) and self.cls == other.cls



# @dataclass(frozen=True)
# class Object_t(NormalizedType):
#     def __post_init__(self):
#         logger.warning("Object_t should not be used as a type for @tools")
#     def __str__(self) -> str:
#         return 'object'
#     def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
#         return True


@dataclass(frozen=True)
class Any_t(NormalizedType):
    def __str__(self) -> str:
        return 'Any'
    def matches(self, other: 'NormalizedType', strict: bool = False) -> bool:
        return True
    def __eq__(self, other):
        return True
    def __req__(self, other):
        return True

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


    if isinstance(t, type) and is_dataclass(t): # is_dataclass would return True for dataclass instances too, which we don't want to match
        return Dataclass_t(t)

    if isinstance(t, type) and issubclass(t, BaseModel):
        return PydanticModel_t(t)


    pdb.set_trace()
    ...
    raise NotImplementedError("TODO")


#TODO: the only real way to make this safe is to manually parse the types ourselves
#      with `eval`, even if you restrict __builtins__ / etc., the user can still get access 
#      to them through the the function the docstring is attached to, e.g. `func.__globals__`
#      TBD on how necessary since generally we should be in control of the docstrings
#      and if we're not, the user would be able to execute arbitrary code anyways
# For now, the main mitigation is the debug=False flag in the @tool decorator. Production code
#      should always have this set to False (forcing all docstring types to Any_t). When true,
#      this function will be called on the docstring's type annotations to normalize them.
def evaluate_type_str(type_str: str, globals: dict[str, Any]) -> NormalizedType:
    try:
        t = eval(type_str, globals)
    except Exception as e:
        raise ValueError(f"Could not evaluate type string '{type_str}'") from e

    return normalize_type(t)