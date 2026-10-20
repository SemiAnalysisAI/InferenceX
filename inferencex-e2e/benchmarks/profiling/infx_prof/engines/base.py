"""What an engine adapter declares."""


class Engine:
    name = None
    caller_roots = ()
    launcher_modules = ()
    launcher_packages = ()
    version_packages = ()
    env_prefixes = ()
    window_control = "file"

    def install(self, registry):
        raise NotImplementedError
