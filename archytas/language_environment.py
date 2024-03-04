import sys
from io import StringIO
from typing import Any
from abc import ABC, abstractmethod


class LanguageEnvironment(ABC):
    def __init__(self):
        self.locals: dict[str, Any] = {}
        self.imports: list[str] = []
        self.all_scripts: list[str] = []

    @abstractmethod
    def add_locals(self, new_locals):
        pass

    @abstractmethod
    def execute(self): 
        pass

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
            self.execute(script)
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