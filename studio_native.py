"""Build the pinned llama.cpp bridge for source and wheel installations."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
from studio_paths import ASSET_ROOT, data_root

COMMIT = 'c0bc8591e8815c63cb01dd3f051a8b0df02501c9'
UPSTREAM = 'https://github.com/ggml-org/llama.cpp.git'


def run(command, **kwargs):
    print('+', subprocess.list2cmdline(list(map(str, command))), flush=True)
    result=subprocess.run(list(map(str, command)), **kwargs)
    if result.returncode:
        raise RuntimeError(f'Build command exited {result.returncode}: {command[0]}')
    return result


def git(vendor, *args, check=True):
    result = subprocess.run(['git', '-C', str(vendor), *args],
                            capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result


def build(cuda=False, jobs=2, architectures='native', cuda_root=None):
    if jobs < 1:
        raise ValueError('Build jobs must be positive.')
    for program in ('git', 'cmake'):
        if not shutil.which(program):
            raise FileNotFoundError(program + ' is not on PATH.')
    import re
    output=subprocess.check_output(['cmake','--version'],text=True)
    version=tuple(map(int,re.search(r'(\d+)\.(\d+)\.(\d+)',output).groups()))
    if version < (3,24,0):
        raise RuntimeError('CMake 3.24+ is required; found '+'.'.join(map(str,version)))
    vendor = data_root() / '.vendor/llama.cpp'
    if not (vendor / '.git').is_dir():
        vendor.mkdir(parents=True, exist_ok=True)
        git(vendor, 'init')
        git(vendor, 'remote', 'add', 'origin', UPSTREAM)
    if git(vendor, 'status', '--porcelain').stdout.strip():
        raise RuntimeError('The llama.cpp checkout contains edits. Commit or move them before building.')
    if git(vendor, 'cat-file', '-e', COMMIT + '^{commit}', check=False).returncode:
        git(vendor, 'fetch', '--depth', '1', 'origin', COMMIT)
    if git(vendor, 'rev-parse', 'HEAD', check=False).stdout.strip() != COMMIT:
        git(vendor, 'checkout', '--detach', COMMIT)
    folder = data_root() / 'native' / ('build-cuda' if cuda else 'build-cpu')
    command = ['cmake', '-S', ASSET_ROOT / 'native', '-B', folder,
               '-DLLAMA_CPP_DIR=' + str(vendor), '-DGGML_BACKEND_DL=ON',
               '-DGGML_NATIVE=OFF', '-DCMAKE_BUILD_TYPE=Release',
               '-DGGML_CUDA=' + ('ON' if cuda else 'OFF')]
    targets = ['activation_bridge', 'ggml-cpu']
    if cuda:
        if not cuda_root:
            nvcc = shutil.which('nvcc')
            if not nvcc:
                raise FileNotFoundError('CUDA compiler nvcc is not on PATH.')
            cuda_root = str(Path(nvcc).resolve().parent.parent)
        if os.name == 'nt':
            command += ['-T', 'cuda=' + cuda_root]
        command += ['-DCMAKE_CUDA_ARCHITECTURES=' + architectures]
        targets.append('ggml-cuda')
    run(command)
    run(['cmake', '--build', folder, '--config', 'Release', '--target',
         *targets, '--parallel', str(jobs)])
    from native_deploy import deploy
    deploy(folder, cuda=cuda, build_info={'llama_commit':COMMIT,'cuda':cuda,'architectures':architectures})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--architectures', default='native')
    parser.add_argument('--cuda-root')
    args = parser.parse_args()
    build(args.cuda, args.jobs, args.architectures, args.cuda_root)


if __name__ == '__main__':
    main()
