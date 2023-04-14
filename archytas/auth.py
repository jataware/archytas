import toml
import os


def add_openai_auth():
    if "OPENAI_API_KEY" not in os.environ:
        local_filehandle = '.openai.toml'
        home_filehandle = os.path.expanduser('~/.openai.toml')
        if os.path.exists(local_filehandle): 
            conf = toml.load(local_filehandle)
        elif os.path.exists(home_filehandle):
            conf = toml.load(home_filehandle)
        else:
            raise Exception("No OpenAI Key Given")
        os.environ['OPENAI_API_KEY'] = conf.get('openai_key', None)
