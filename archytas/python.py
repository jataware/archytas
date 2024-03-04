import sys
from io import StringIO
from types import ModuleType
from typing import Any
from archytas.utils import get_local_name

from archytas.language_environment import LanguageEnvironment


class Python(LanguageEnvironment):
    def add_locals(
        self, new_locals: list[type | ModuleType | Any], outer_locals: dict[str, Any]
    ):
        for local in new_locals:
            if isinstance(local, type):
                update = {local.__name__: local}
            elif isinstance(local, ModuleType):
                name = get_local_name(local, outer_locals)
                update = {name: local}
            else:
                # TODO: not sure if best way to handle other types...
                name = get_local_name(local, outer_locals)
                update = {name: local}

            self._added_locals.update(update)
            self.locals.update(update)

    def execute(self, script:str):
        exec(script, self.locals)


def main():
    import numpy as np

    class MyClass:
        def __init__(self, value):
            self.value = value

    p = Python()

    # prog0
    out, err = p.run_script(
        """
print('Hello world!')
    """
    )
    print(f"out: {out}")
    print(f"err: {err}")

    # prog1
    out, err = p.run_script(
        """
raise Exception('Hello world!')
    """
    )
    print(f"out: {out}")
    print(f"err: {err}")

    # prog2
    p.add_locals([np], locals())
    out, err = p.run_script(
        """
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
    """
    )
    print(f"out: {out}")
    print(f"err: {err}")
    print(f'a: {p.locals["a"]}')
    print(f'b: {p.locals["b"]}')

    p.add_locals([MyClass], locals())

    out, err = p.run_script(
        """
result = np.dot(a, b)
my_obj = MyClass(np.array([1, 2, 3]))
    """
    )
    print(f"out: {out}")
    print(f"err: {err}")
    print(f'result: {p.locals["result"]}')
    print(f'my_obj: {p.locals["my_obj"]}')


if __name__ == "__main__":
    main()
