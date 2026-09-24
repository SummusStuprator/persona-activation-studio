"""Read-only application resources and writable installation state."""
from pathlib import Path
import os
import sys
import studio_assets

CODE_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = CODE_ROOT if (CODE_ROOT / 'studio.toml.example').is_file() else Path(studio_assets.__file__).resolve().parent
SOURCE_INSTALL = ASSET_ROOT == CODE_ROOT

def data_root():
    explicit = os.environ.get('STUDIO_HOME')
    if explicit:
        return Path(explicit).expanduser().resolve()
    if SOURCE_INSTALL:
        return CODE_ROOT
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    return base / 'persona-activation-studio'

def config_path():
    explicit = os.environ.get('STUDIO_CONFIG')
    return Path(explicit).expanduser().resolve() if explicit else data_root() / 'studio.toml'
