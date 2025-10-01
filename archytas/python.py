import sys
import traceback
from io import StringIO
from types import ModuleType
from typing import Any
from archytas.utils import get_local_name

import pdb


class Python:
    def __init__(self):
        self.locals: dict[str, Any] = {}
        self._added_locals: dict[
            str, Any
        ] = {}  # keep track of any modules/classes/functions/etc. added to locals
        self.imports: list[str] = []  # keep track of any imports added
        self.all_scripts: list[str] = []  # keep track of all scripts run

    def update_locals(self, new_locals: dict):
        self.locals.update(new_locals)

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

    def run_script(self, script: str):
        exception_info = None

        # capture any stdout/stderr from the script
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        # save the script text
        self.all_scripts.append(script)

        # run the script
        try:
            exec(script, self.locals)
        except Exception as e:
            exception_info = {
                'type'      : type(e).__name__,
                'message'   : str(e),
                'traceback' : traceback.format_exc()
            }

        # restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return {
            "stdout"    : captured_stdout.getvalue(),
            "stderr"    : captured_stderr.getvalue(),
            "exception" : exception_info
        }

    # def add_imports(self, imports:list[str]):
    #     #TODO: this should import them into the locals dict by calling exec() on the import statements
    #     #      also handle splitting up multiple imports in a single string
    #     #TODO: this could also be a more generic add_code() method
    #     self.imports.extend(imports)


def main():
    import numpy as np

    class MyClass:
        def __init__(self, value):
            self.value = value

    p = Python()

    # prog0
    print('----- prog0 -----')
    out = p.run_script(
        """
print('Hello world!')
    """
    )
    print(f"out: {out}")

    # prog1
    print('----- prog1 -----')
    out = p.run_script(
        """
raise Exception('Hello world!')
    """
    )
    print(f"out: {out}")

    # prog2
    p.add_locals([np], locals())
    print('----- prog2 -----')
    out = p.run_script(
        """
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
    """
    )
    print(f"out: {out}")
    print(f'a: {p.locals["a"]}')
    print(f'b: {p.locals["b"]}')

    p.add_locals([MyClass], locals())

    print('----- prog3 -----')
    out = p.run_script(
        """
result = np.dot(a, b)
my_obj = MyClass(np.array([1, 2, 3]))
    """
    )
    print(f"out: {out}")
    print(f'result: {p.locals["result"]}')
    print(f'my_obj: {p.locals["my_obj"]}')


if __name__ == "__main__":
    main()
