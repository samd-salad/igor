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

def execute(command_name: str, _ctx=None, **kwargs) -> str:
    """Execute a command by name with optional room context.

    _ctx (InteractionContext) is injected by the orchestrator for room-aware
    commands. Commands that need it pull _ctx from kwargs; simple commands
    ignore it.
    """
    if command_name not in _registry:
        return f"Unknown command: {command_name}"
    # Inject _ctx for commands that accept it
    cmd = _registry[command_name]
    import inspect
    sig = inspect.signature(cmd.execute)
    if '_ctx' in sig.parameters:
        kwargs['_ctx'] = _ctx
    return cmd.execute(**kwargs)

def inject_dependencies(registry=None, room_state_mgr=None, pi_client=None):
    """Inject shared dependencies into commands that need them.

    Args:
        registry: ClientRegistry for dynamic client lookup.
        room_state_mgr: RoomStateManager for per-room state.
        pi_client: Legacy PiCallbackClient (fallback for hardware commands).
    """
    for cmd in _registry.values():
        if registry is not None:
            cmd._registry = registry
        if room_state_mgr is not None:
            cmd._room_state_mgr = room_state_mgr
        if pi_client is not None and (hasattr(cmd, 'pi_client') or cmd.name in ('set_volume', 'get_volume', 'label_wakeword')):
            cmd.pi_client = pi_client
