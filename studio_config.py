from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, sys, tomllib

from studio_paths import CODE_ROOT as REPO_ROOT, ASSET_ROOT, data_root, config_path
CONFIG_PATH = config_path()

@dataclass(frozen=True)
class StudioPaths:
    root: Path
    workspace: Path
    project: Path
    exports: Path
    account_db: Path
    datasets: Path
    models: Path
    staging: Path
    anchors: Path
    hf_cache: Path

def _resolve(base: Path, value: str | Path) -> Path:
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base/p).resolve()

def load_config() -> dict:
    data={}
    path = config_path()
    if path.exists():
        with path.open("rb") as f:data=tomllib.load(f)
    raw=data.get("paths",{})
    workspace=_resolve(data_root(),raw.get("workspace","workspace"))
    project=_resolve(workspace,raw.get("project","persona-project"))
    return {
        "paths":{
            "workspace":workspace,
            "project":project,
            "exports":_resolve(project,raw.get("exports","x_exports")),
            "account_db":_resolve(project,raw.get("account_db","x_accounts.db")),
            "datasets":_resolve(workspace,raw.get("datasets","datasets")),
            "models":_resolve(workspace,raw.get("models","models/persona")),
            "staging":_resolve(workspace,raw.get("staging","models/staging")),
            "anchors":_resolve(workspace,raw.get("anchors","anchors")),
        },
        "runtime":data.get("runtime",{}),
        "python":data.get("python",{}).get("executable",sys.executable),
    }

def paths() -> StudioPaths:
    c=load_config();p=c["paths"]
    hf=Path(os.environ.get("HF_HOME",Path.home()/".cache"/"huggingface")).expanduser().resolve()
    return StudioPaths(root=REPO_ROOT,workspace=p["workspace"],project=p["project"],exports=p["exports"],
        account_db=p["account_db"],datasets=p["datasets"],models=p["models"],staging=p["staging"],
        anchors=p["anchors"],hf_cache=hf)

def ensure_workspace() -> StudioPaths:
    p=paths()
    for d in (p.workspace,p.project,p.exports,p.datasets,p.models,p.staging,p.anchors):
        d.mkdir(parents=True,exist_ok=True)
    return p
