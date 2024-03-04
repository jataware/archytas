import sys
from io import StringIO
from types import ModuleType
from typing import Any
try: 
    from juliacall import Main as jl
except ModuleNotFoundError:
    is_module_active = False
else: 
    is_module_active = True

class Julia:
    def __init__(self):
        if not is_module_active:
            raise ModuleNotFoundError("Optional dependency `juliacall` is not installed")
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
            sys.stderr.write(str(e))

        # restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return captured_stdout.getvalue(), captured_stderr.getvalue()

    # def add_imports(self, imports:list[str]):
    #     #TODO: this should import them into the locals dict by calling exec() on the import statements
    #     #      also handle splitting up multiple imports in a single string
    #     #TODO: this could also be a more generic add_code() method
    #     self.imports.extend(imports)
