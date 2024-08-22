from pydantic import BaseModel
from types import UnionType, GenericAlias
from typing import Any, get_origin, get_args as get_type_args, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import DataclassInstance  # only available to type checkers
from typing_extensions import TypeIs
from dataclasses import is_dataclass, asdict, _MISSING_TYPE

from .utils import type_to_str
from .constants import TAB


def verify_model_fields(cls: type[BaseModel], data: dict) -> dict:
    """
    Verify that the keys in the data dictionary are valid fields for the model class.
    Recursively verifies any nested models.
    """
    cls_fields = set(cls.__annotations__.keys())
    data_fields = set(data.keys())

    # Check if the provided data fields are all valid
    if not data_fields.issubset(cls_fields):
        raise ValueError(
            f"Invalid fields as input to {cls.__name__}: {data_fields - cls_fields}. Valid fields are: {cls_fields}")

    # Recursively verify nested models
    for key, value in data.items():
        if isinstance(value, dict) and issubclass(cls.__annotations__[key], BaseModel):
            # Recursively call verify_model_fields for nested models
            nested_cls = cls.__annotations__[key]
            verify_model_fields(nested_cls, value)

    return data


def construct_structured_type(arg_type: type, data: dict) -> 'DataclassInstance|BaseModel':
    """
    construct an instance of a structured type from a dictionary (currently only support dataclass and pydantic models)
    recursively construct nested structured types, and validate the input data against the structured type

    Args:
        arg_type (type): The structured type to construct
        data (dict): The dictionary to construct the structured type from

    Returns:
        The constructed structured type instance 
    """

    if is_dataclass(arg_type):
        return construct_dataclass(arg_type, data)

    if issubclass(arg_type, BaseModel):
        verify_model_fields(arg_type, data)
        return arg_type(**data)

    # Future support for other structured types can be added here
    raise ValueError(f"Unsupported structured type {arg_type}")


def construct_dataclass(cls: 'type[DataclassInstance]', data: dict) -> 'DataclassInstance':
    """
    Construct (potentially recursively) a dataclass instance from a dictionary.

    Args:
        dataclass_type (type[DataclassInstance]): The dataclass type to construct
        data (dict): The dictionary to construct the dataclass instance from

    Returns:
        DataclassInstance: The constructed dataclass instance
    """
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    body = {}
    for field_name, field_type in fieldtypes.items():
        if field_name in data:
            if is_structured_type(field_type):
                body[field_name] = construct_structured_type(field_type, data[field_name])
            else:
                body[field_name] = data[field_name]
    return cls(**body)


def is_structured_type(arg_type: type | UnionType | GenericAlias) -> 'TypeIs[type[BaseModel] | type[DataclassInstance]]':
    """Check if a type is a structured type like a dataclass or pydantic model"""
    if isinstance(arg_type, UnionType) or get_origin(arg_type) is Union:
        assert not any(is_structured_type(t) for t in get_type_args(arg_type)
                       ), f"Unions containing any structured types are not supported. Got {arg_type}"
        return False

    # handle if type has a generic subscript (e.g. list[str])
    arg_type = get_origin(arg_type) or arg_type
    return is_dataclass(arg_type) or issubclass(arg_type, BaseModel)


def get_structured_input_description(arg_type: type, arg_name: str, arg_desc: str, arg_default: Any | None, *, indent: int) -> list[str]:
    """
    Generate the tool description for a structured argument like a dataclass or pydantic model.

    Args:
        arg_type (type): The type of the structured argument. currently only supports dataclasses and pydantic models
        arg_name (str): The name of the argument
        arg_desc (str): The description of the argument
        arg_default (Any|None): The default value of the argument. None indicates no default value
    """

    # convert the default value to a string
    if is_dataclass(arg_default):
        arg_default = str(asdict(arg_default))  # convert dataclass to dict
    elif isinstance(arg_default, BaseModel):
        arg_default = str(arg_default.model_dump())  # convert pydantic model to dict
    else:
        arg_default = str(arg_default) if arg_default is not None else None

    if is_dataclass(arg_type):
        return get_dataclass_input_description(arg_type, arg_name, arg_desc, arg_default, indent=indent)

    if issubclass(arg_type, BaseModel):
        return get_pydantic_input_description(arg_type, arg_name, arg_desc, arg_default, indent=indent)

    raise ValueError(f"Unsupported structured type {arg_name}: {arg_type}")


def get_dataclass_input_description(arg_type: 'type[DataclassInstance]', arg_name: str, arg_desc: str, arg_default: str, *, indent: int) -> list[str]:
    """
    Build the input description for a dataclass, including all of its fields (and potentially their nested fields).

    Args:
        arg_type (type[DataclassInstance]): The dataclass type to get the input description for
        arg_name (str): The name of the argument
        arg_desc (str): The description of the argument
        arg_default (str): The default value of the argument
        indent (int): The level of indentation to use for the description

    Returns:
        list[str]: A list of strings that make up the input description for the dataclass
    """
    chunks = []
    num_fields = len(arg_type.__dataclass_fields__)
    num_required_fields = sum(1 for field in arg_type.__dataclass_fields__.values() if isinstance(
        field.default, _MISSING_TYPE) and isinstance(field.default, _MISSING_TYPE))
    num_optional_fields = num_fields - num_required_fields

    # argument name and high level description
    chunks.append(f'\n{TAB*indent}"{arg_name}":')
    if arg_desc:
        chunks.append(f" {arg_desc}.")
    if arg_default:
        chunks.append(f" Defaults to {arg_default}.")
    if num_required_fields == 0:
        chunks.append(" a json object with zero or more of the following optional fields:\n")
    elif num_optional_fields > 0:
        chunks.append(" a json object with the following fields (optional fields may be omitted):\n")
    else:
        chunks.append(" a json object with the following fields:\n")

    # opening brackets
    chunks.append(f"{TAB*(indent)}{{")

    # get the description for each field
    for field_name, field in arg_type.__dataclass_fields__.items():
        field_type = field.type
        field_desc = field.metadata.get("description", "")

        # determine the default value of the field
        if not isinstance(field.default, _MISSING_TYPE):
            field_default = field.default
        elif not isinstance(field.default_factory, _MISSING_TYPE):
            field_default = field.default_factory()
        else:
            field_default = None

        # if the field is a structured type, recursively get the input description
        if is_structured_type(field_type):
            chunks.extend(get_structured_input_description(
                field_type, field_name, field_desc, field_default, indent=indent+1))
            continue

        # add the field description to the chunks
        chunks.append(
            f'\n{TAB*(indent+1)}"{field_name}": ({type_to_str(field_type)}{", optional" if field_default is not None else ""})')
        if field_desc:
            chunks.append(f" {field_desc}.")
        if field_default is not None:
            chunks.append(f" Defaults to {field_default}.")

    # closing brackets
    chunks.append(f"\n{TAB*indent}}}")

    return chunks


def get_pydantic_input_description(arg_type: type[BaseModel], arg_name: str, arg_desc: str, arg_default: str, *, indent: int) -> list[str]:
    """
    Build the input description for a pydantic model, including all of its fields (and potentially their nested fields).

    Args:
        arg_type (type[BaseModel]): The pydantic model type to get the input description for
        arg_name (str): The name of the argument
        arg_desc (str): The description of the argument
        arg_default (str): The default value of the argument
        indent (int): The level of indentation to use for the description

    Returns:
        list[str]: A list of strings that make up the input description for the pydantic model
    """
    chunks = []
    num_fields = len(arg_type.model_fields)
    num_required_fields = sum(1 for field in arg_type.model_fields.values() if field.is_required())
    num_optional_fields = num_fields - num_required_fields

    # argument name and high level description
    chunks.append(f'\n{TAB*indent}"{arg_name}":')
    if arg_desc:
        chunks.append(f" {arg_desc}.")
    if arg_default:
        chunks.append(f" Defaults to {arg_default}.")
    if num_required_fields == 0:
        chunks.append(" a json object with zero or more of the following optional fields:\n")
    elif num_optional_fields > 0:
        chunks.append(" a json object with the following fields (optional fields may be omitted):\n")
    else:
        chunks.append(" a json object with the following fields:\n")

    # opening brackets
    chunks.append(f"{TAB*(indent)}{{")

    # get the description for each field
    for field_name, field in arg_type.model_fields.items():
        field_type = field.annotation
        field_desc = field.description

        # determine the default value of the field
        field_default = None
        if not field.is_required():
            if field.default_factory is not None:
                field_default = field.default_factory()
            else:
                field_default = field.default

        # if the field is a structured type, recursively get the input description
        if is_structured_type(field_type):
            chunks.extend(get_structured_input_description(
                field_type, field_name, field_desc, field_default, indent=indent+1))
            continue

        # add the field description to the chunks
        chunks.append(
            f'\n{TAB*(indent+1)}"{field_name}": ({type_to_str(field_type)}{", optional" if field_default is not None else ""})')
        if field_desc:
            chunks.append(f" {field_desc}.")
        if field_default is not None:
            chunks.append(f" Defaults to {field_default}.")

    # closing brackets
    chunks.append(f"\n{TAB*indent}}}")

    return chunks
