import sys
from io import StringIO
from typing import Any
from tempfile import NamedTemporaryFile
import subprocess

from tool_utils import tool

tool_docs = lambda language: (
    f"Runs {language} n code in a {language} environment.\n\n"
    "The environment is persistent between runs, so any variables created will be available in subsequent runs."
    "The only visible effects of this tool are from output to stdout/stderr. "
    "If you want to view a result, you MUST print it.\n\n"
    f"Args:\n\tcode (str): The {language} code to run"
    "Returns:\n\tstr"
)

class LanguageEnvironment:
    language = "Python"
    
    def __new__(cls, _name, _bases, methods):
        def run(self, code: str) -> str:
            return self.run_script(code)

        language = cls.language
        run.__doc__ = tool_docs(language)
        methods["run"] = tool(run)
        return super().__new__(cls)

    def __init__(self, executable: str | None = None, response_char_limit=1000):
        language = self.__class__.language
        default_executable = f"{language.lower()} %s"
        self.executable = executable if executable is None else default_executable
        self.locals: dict[str, Any] = {}
        self.imports: list[str] = []
        self.history: list[str] = []
        self.response_char_limit = response_char_limit

    def get_standalone_tools(self):
        def run(code: str) -> str:
            return self.run_script(code)

        language = self.__class__.language
        run.__doc__ = tool_docs(language)
        return [tool(run)]


    def run_script(self, script: str):
        self.history.append(script)
        with NamedTemporaryFile() as file:
            file.write(script)
            response = subprocess.run(self.executable % file.name, shell=True, capture_output=True)

        stdout, stderr = response.stdout.decode(), response.stderr.decode()
        def prep_for_llm(output, name):
            if len(output) > self.response_char_limit:
                output = output[-(self.response_char_limit+1):-1]
            return f"{name}t: \n`````{output}\n`````\n"

        return prep_for_llm(stdout, "stdout") + prep_for_llm(stderr, "stderr")

    def imports(self, imports:list[str]):
        """
        import pkg_resources

        for dist in pkg_resources.working_set:
            print(dist.project_name, dist.version)
        """
