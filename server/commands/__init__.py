import importlib
import pkgutil
from pathlib import Path
from .base import Command

_registry: dict[str, Command] = {}

def _discover_commands():
    """Auto-discover and register all Command subclasses."""
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name == "base":
            continue
        module = importlib.import_module(f".{module_name}", package=__name__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, Command) and 
                attr is not Command and
                attr.name):
                _registry[attr.name] = attr()

_discover_commands()

def get_all_commands() -> dict[str, Command]:
    return _registry

def get_tools() -> list[dict]:
    return [cmd.to_tool() for cmd in _registry.values()]

def execute(name: str, **kwargs) -> str:
    if name not in _registry:
        return f"Unknown command: {name}"
    return _registry[name].execute(**kwargs)

def inject_pi_client(pi_client):
    """Inject pi_client into hardware commands that need it."""
    for cmd in _registry.values():
        if hasattr(cmd, 'pi_client') or cmd.name in ('set_volume', 'get_volume'):
            cmd.pi_client = pi_client
