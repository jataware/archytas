
class AuthenticationError(Exception):
    pass


class ModelError(Exception):
    pass


class ExecutionError(Exception):
    pass


class ContextWindowExceededError(ExecutionError):
    pass
