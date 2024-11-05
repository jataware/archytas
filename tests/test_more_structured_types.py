from archytas.tool_utils import tool, get_tool_prompt_description
from .test_structured_types import A, B, C, D, M1, M2, M3, GenericModelA, GenericModelB


def get_list_of_structs():
    @tool(devmode=True)
    def tool0(l: list[A]) -> list[A]:
        """
        Example tool

        Args:
            l (list[A]): list of A instances

        Returns:
            list[A]: list of A instances
        """
        return l
    
    return tool0


def test_list_of_structs():
    t = get_list_of_structs()
    get_tool_prompt_description(t)