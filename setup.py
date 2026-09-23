"""Include runtime resources in wheels without duplicating source assets."""
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
RESOURCE_PATTERNS = (
    'README.md', 'LICENSE', 'THIRD_PARTY.md', 'studio.toml.example', 'pyproject.toml',
    'CONTRIBUTING.md',
    'docs/*.md', 'scripts/*.ps1', 'scripts/*.sh', 'scripts/*.py', 'native/*.cpp',
    'native/CMakeLists.txt', 'paper/README.md', 'paper/LICENSE',
    'paper/datasets/*.json', 'tests/*.py', 'examples/**/*',
    'constraints/*.txt', 'CHANGELOG.md', 'SECURITY.md', 'CITATION.cff',
)


class ResourceBuild(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / 'studio_assets'
        for pattern in RESOURCE_PATTERNS:
            for source in ROOT.glob(pattern):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


setup(cmdclass={'build_py': ResourceBuild})
