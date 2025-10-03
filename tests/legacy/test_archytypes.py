from typing import Optional, Union, Tuple, List, Dict
from archytas.tool_utils import AgentRef, LoopControllerRef, is_tool, tool, toolset
from archytas.archytypes import normalize_type, NormalizedType
import pytest

#TODO: want to set up so that different versions of python also get tested
#      3.10, 3.11, 3.12, 3.13, ...


import pdb



def test_annotation_normalization():
    from archytas.archytypes import Int_t, Float_t, Str_t, Bool_t, None_t, Union_t, List_t, Tuple_t, Dict_t, Any_t
    from types import NoneType

    # int | None
    t = Union_t(Int_t(), None_t())
    assert t == normalize_type(Optional[int])
    assert t == normalize_type(int | None)
    assert t == normalize_type(Optional[int] | None)
    assert t == normalize_type(Optional[int] | None | NoneType)
    assert t == normalize_type(Union[int, None])
    assert t == normalize_type(Union[int, Union[None, NoneType]])
    assert t == normalize_type(Union[int, Union[None, Union[None, NoneType]]])

    # int | float | str   # also testing nested unions and duplicates
    t = Union_t(Int_t(), Float_t(), Str_t())
    assert t == normalize_type(Union[int, Union[int, Union[str, Union[float, int]]]])
    assert t == normalize_type(int | float | str)
    assert t == normalize_type(int | float | str | float | str | int)


    # tuple[int, str]
    t = Tuple_t((Int_t(), Str_t()))
    assert t == normalize_type(tuple[int, str])
    assert t == normalize_type(Tuple[int, str])

    # tuple[int, ...]
    t = Tuple_t((Int_t(), ...))

    # list[int]
    t = List_t(Int_t())

    # dict[str, int]

    # Any should be equal to all types
    t = Any_t()
    assert t == normalize_type(Optional[int])
    assert normalize_type(Optional[int]) == t
    assert not (normalize_type(Optional[int]) != t)
    assert not (t != normalize_type(Optional[int]))
    assert t == normalize_type(int | float | str)
    assert t == normalize_type(int | float | str | float | str | int)
    assert t == normalize_type(Union[int, Union[int, Union[str, Union[float, int]]]])
    assert t == normalize_type(tuple[int, str])
    assert t == normalize_type(Tuple[int, str])
    assert t == normalize_type(Union[int, None])
    assert normalize_type(list[int]) == t
    assert normalize_type(dict[str, int]) == t
    #TODO:...

def test_normalized_string():
    from archytas.archytypes import Int_t, Float_t, Str_t, Bool_t, None_t, Union_t, List_t, Tuple_t, Dict_t, Any_t
    from types import NoneType

    assert str(Str_t()) == "str"
    assert str(Int_t()) == "int"
    assert str(Float_t()) == "float"
    assert str(Bool_t()) == "bool"
    assert str(None_t()) == "NoneType"
    # assert str(List_t(Str_t)) == "list[str]"
    assert str(normalize_type(list[str])) == "list[str]"
    # assert str(Union_t((str, int))) == "UnionType[str, int]"

def test_optional_signature():
    @tool
    def test1(a: Optional[int] = None) -> str:
        """
        Args:
            a (int, optional): Description of the argument `a`. Defaults to None.

        Returns:
            str: Description of the return value.
        """
        raise NotImplementedError("implementations are omitted since these are only meant to test the signature")

    @tool
    def test2(a: Optional[int]) -> Optional[int]:
        """
        Args:
            a (int, optional): Description of the argument `a`. Defaults to None.

        Returns:
            Optional[int]: Description of the return value. Can be None.
        """
        raise NotImplementedError("implementations are omitted since these are only meant to test the signature")

@pytest.mark.xfail
def test_climate_data_utility_signature():
    class ClimateDataUtilityToolset:
        """Toolset for ClimateDataUtility context"""

        @tool(devmode=True)
        async def detect_resolution(self, filepath: str, geo_columns: object, agent: AgentRef) -> str:
            """
            This function should be used to detect the resolution of a dataset.
            This can be used if the user doesn't know the resolution or if you are regridding a dataset and don't know a starting resolution.
            The resolution can further be used to make informed decisions about the scale multiplier to use for regridding.

            The dataset should have some geographical data in it in the form of a latitude and longitude column.

            Args:
                filepath (str): The filepath to the dataset to open.
                geo_columns (object): The names of the geographical columns in the dataset. This is an optional argument for this tool.
                    This is an object with the keys 'lat_column' and 'lon_column'.
                    The 'lat_column' key should have the name of the latitude column and the 'lon_column' key should have the name of the longitude column.

            Returns:
                str: Returned description of the resolution of the dataset.

            You should show the user the result after this function runs.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")



        @tool(devmode=True)
        async def regrid_dataset(
            self,
            dataset: str,
            target_resolution: tuple,
            agent: AgentRef,
            loop: LoopControllerRef,
            aggregation: Optional[str] = "interp_or_mean",
        ) -> str:
            """
            This tool should be used to show the user code to regrid a netcdf dataset with detectable geo-resolution.

            If a user asks to regrid a dataset, use this tool to return them code to regrid the dataset.

            If you are given a netcdf dataset, use this tool instead of any other regridding tool.

            If you are asked about what is needed to regrid a dataset, please provide information about the arguments of this tool.

            Args:
                dataset (str): The name of the dataset instantiated in the jupyter notebook.
                target_resolution (tuple): The target resolution to regrid to, e.g. (0.5, 0.5). This is in degrees longitude and latitude.
                aggregation (Optional): The aggregation function to be used in the regridding. The options are as follows:
                    'conserve'
                    'min'
                    'max'
                    'mean'
                    'median'
                    'mode'
                    'interp_or_mean'
                    'nearest_or_mode'

            Returns:
                str: Status of whether or not the dataset has been persisted to the HMI server.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")


        @tool(devmode=True)
        async def get_netcdf_plot(
            self,
            dataset_variable_name: str,
            agent: AgentRef,
            loop: LoopControllerRef,
            plot_variable_name: Optional[str] = None,
            lat_col: Optional[str] = "lat",
            lon_col: Optional[str] = "lon",
            time_slice_index: Optional[int] = 1,
        ) -> str:
            """
            This function should be used to get a plot of a netcdf dataset.

            This function should also be used to preview any netcdf dataset.

            If the user asks to plot or preview a dataset, use this tool to return plotting code to them.

            You should also ask if the user wants to specify the optional arguments by telling them what each argument does.

            Args:
                dataset_variable_name (str): The name of the dataset instantiated in the jupyter notebook.
                plot_variable_name (Optional): The name of the variable to plot. Defaults to None.
                    If None is provided, the first variable in the dataset will be plotted.
                lat_col (Optional): The name of the latitude column. Defaults to 'lat'.
                lon_col (Optional): The name of the longitude column. Defaults to 'lon'.
                time_slice_index (Optional): The index of the time slice to visualize. Defaults to 1.

            Returns:
                str: The code used to plot the netcdf.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")




def test_fixed_climate_data_utility_signature():
    """
    Differences between the two annotations:
    - annotate `geo_columns` as `dict` instead of `object`
    - in docstring for `regrid_dataset`, change `aggregation (Optional)` to `aggregation (str, optional)`. Note that Optional is case-sensitive.
    - other instances of `(Optional)` in docstring replaced with `(<type>, optional)`
        - note that these could also be replaced with `(Optional[<type>])`

    """
    class ClimateDataUtilityToolset:
        """Toolset for ClimateDataUtility context"""

        @tool(devmode=True)
        async def detect_resolution(self, filepath: str, geo_columns: dict, agent: AgentRef) -> str:
            """
            This function should be used to detect the resolution of a dataset.
            This can be used if the user doesn't know the resolution or if you are regridding a dataset and don't know a starting resolution.
            The resolution can further be used to make informed decisions about the scale multiplier to use for regridding.

            The dataset should have some geographical data in it in the form of a latitude and longitude column.

            Args:
                filepath (str): The filepath to the dataset to open.
                geo_columns (dict): The names of the geographical columns in the dataset. This is an optional argument for this tool.
                    This is an object with the keys 'lat_column' and 'lon_column'.
                    The 'lat_column' key should have the name of the latitude column and the 'lon_column' key should have the name of the longitude column.

            Returns:
                str: Returned description of the resolution of the dataset.

            You should show the user the result after this function runs.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")



        @tool(devmode=True)
        async def regrid_dataset(
            self,
            dataset: str,
            target_resolution: tuple,
            agent: AgentRef,
            loop: LoopControllerRef,
            aggregation: Optional[str] = "interp_or_mean",
        ) -> str:
            """
            This tool should be used to show the user code to regrid a netcdf dataset with detectable geo-resolution.

            If a user asks to regrid a dataset, use this tool to return them code to regrid the dataset.

            If you are given a netcdf dataset, use this tool instead of any other regridding tool.

            If you are asked about what is needed to regrid a dataset, please provide information about the arguments of this tool.

            Args:
                dataset (str): The name of the dataset instantiated in the jupyter notebook.
                target_resolution (tuple): The target resolution to regrid to, e.g. (0.5, 0.5). This is in degrees longitude and latitude.
                aggregation (str, optional): The aggregation function to be used in the regridding. The options are as follows:
                    'conserve'
                    'min'
                    'max'
                    'mean'
                    'median'
                    'mode'
                    'interp_or_mean'
                    'nearest_or_mode'

            Returns:
                str: Status of whether or not the dataset has been persisted to the HMI server.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")


        @tool(devmode=True)
        async def get_netcdf_plot(
            self,
            dataset_variable_name: str,
            agent: AgentRef,
            loop: LoopControllerRef,
            plot_variable_name: Optional[str] = None,
            lat_col: Optional[str] = "lat",
            lon_col: Optional[str] = "lon",
            time_slice_index: Optional[int] = 1,
        ) -> str:
            """
            This function should be used to get a plot of a netcdf dataset.

            This function should also be used to preview any netcdf dataset.

            If the user asks to plot or preview a dataset, use this tool to return plotting code to them.

            You should also ask if the user wants to specify the optional arguments by telling them what each argument does.

            Args:
                dataset_variable_name (str): The name of the dataset instantiated in the jupyter notebook.
                plot_variable_name (str, optional): The name of the variable to plot. Defaults to None.
                    If None is provided, the first variable in the dataset will be plotted.
                lat_col (str, optional): The name of the latitude column. Defaults to 'lat'.
                lon_col (str, optional): The name of the longitude column. Defaults to 'lon'.
                time_slice_index (int, optional): The index of the time slice to visualize. Defaults to 1.

            Returns:
                str: The code used to plot the netcdf.
            """
            raise NotImplementedError("implementations are omitted since these are only meant to test the signature")
