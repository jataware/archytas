from typing import Any
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
import subprocess

from archytas.tool_utils import tool

tool_docs = lambda language: (
    f"Runs {language} n code in a {language} environment.\n\n"
    "The environment is persistent between runs, so any variables created will be available in subsequent runs."
    "The only visible effects of this tool are from output to stdout/stderr. "
    "If you want to view a result, you MUST print it.\n\n"
    f"Args:\n\tcode (str): The {language} code to run\n"
    "Returns:\n\tstr: potentially truncated stdout and stderr"
)

python_scripts = {
    "dependency_list": (
        "import pkg_resources\n"
        "print([dist.project_name for dist in pkg_resources.working_set])"
    ),
}


@dataclass
class Dependency:
    name: str
    alias: str | None = None
    exports: list[str] | None = None


@dataclass
class Local:
    name: str
    type_name: str
    serialization: str
    

class LanguageEnvironment:
    language = "Python"
    scripts = python_scripts
    
    @property
    def dependency_header(self):
        imports = ""
        for dep in self.dependencies:
            imports += f"import {dep.name}"
            if dep.alias is not None:
                imports += f" {dep.alias}"
            elif dep.exports is not None and len(dep.exports) > 0:
                imports += f""
            imports += "\n"
        return imports

    @property
    def locals_header(self):
        locals = "import dill\n"
        for local in self.locals:
            local += f"{local.name} = dill.loads({local.serialization})\n"
        return locals        

    def __new__(cls):
        env_class = super().__new__(cls)
        def run(self, code: str) -> str:
            return self.run_script(code)

        language = cls.language
        run.__doc__ = tool_docs(language)
        env_class.run = tool()(run)
        return env_class

    def __init__(self, executable: str | None = None, response_char_limit=5000):
        language = self.__class__.language
        default_executable = f"{language.lower()} %s"
        self.executable = executable if executable is None else default_executable
        self.locals: list[Local] = []
        self.dependencies: list[Dependency] = []
        self.history: list[str] = []
        self.response_char_limit = response_char_limit

    @tool()
    def dependency_list(self) -> str:
        """
        List all dependencies that are currently installed in the environment.
        All of these packages are usable dependencies but they might still need
        to be imported into the environment.

        Returns:
            str: Names of all the packages currently installed 
        """
        return self.run_action(self.__class__.scripts["dependency_list"])


    # def get_standalone_tools(self):
    #     def run(code: str) -> str:
    #         return self.run_script(code)

    #     language = self.__class__.language
    #     run.__doc__ = tool_docs(language)
    #     return [tool(run)]

    def run_action(self, script):    
        with NamedTemporaryFile() as file:
            file.write(script)
            try:
                response = subprocess.run(self.executable % file.name, shell=True, capture_output=True).stdout.decode()
            except subprocess.CalledProcessError:
                return "ENCOUNTERED ERROR RUNNING ACTION"
            return response

    def run_script(self, script: str):
        self.history.append(script)
        full_script = self.import_header + self.locals_header + script
        with NamedTemporaryFile() as file:
            file.write(full_script)
            response = subprocess.run(self.executable % file.name, shell=True, capture_output=True)

        stdout, stderr = response.stdout.decode(), response.stderr.decode()
        def prep_for_llm(output, name):
            if len(output) > self.response_char_limit:
                output = output[:self.response_char_limit/2] + "...\n[TRUNCATED]\n..." + output[:-self.response_char_limit/2]
            return f"{name}t: \n`````{output}\n`````\n"

        return prep_for_llm(stdout, "stdout") + prep_for_llm(stderr, "stderr")

