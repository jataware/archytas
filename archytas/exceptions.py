
class AuthenticationError(Exception):
    pass


class ModelError(Exception):
    pass


class ExecutionError(Exception):
    pass


class ContextWindowExceededError(ExecutionError):
    def __init__(self, *args, sent=None, maximum=None):
        super().__init__(*args)
        if isinstance(sent, str):
            self.sent = int(sent)
        if isinstance(maximum, str):
            self.maximum = int(maximum)
