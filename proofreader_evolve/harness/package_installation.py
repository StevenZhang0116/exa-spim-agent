"""Install agent-requested Python dependencies in the driver's existing environment."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from packaging.requirements import Requirement


def installed_packages():
    return dict(sorted((dist.metadata['Name'], dist.version)
                       for dist in importlib.metadata.distributions() if dist.metadata['Name']))


def install_packages(gen_dir, packages):
    """Add named dependencies with pip, preserving already installed versions.

    This host tool can access the package index; model workers remain offline.
    Requirements are package names/specifiers, not pip options or local code paths.
    """
    if not isinstance(packages, list) or not packages:
        raise ValueError('Supply a nonempty list of package requirements, e.g. ["catboost>=1.2"]')
    requirements = []
    for value in packages:
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each package requirement must be a nonempty string')
        requirement = Requirement(value)
        if requirement.url:
            raise ValueError('Use package names and version specifiers from the package index, not URLs or paths')
        requirements.append(str(requirement))

    gen_dir = Path(gen_dir)
    installs = gen_dir / 'package_installs'
    directory = installs / f'install{len(list(installs.glob("install*"))) + 1:03d}'
    directory.mkdir(parents=True)
    before = installed_packages()
    constraints = directory / 'constraints.txt'
    constraints.write_text(''.join(f'{name}==={version}\n' for name, version in before.items()))
    (directory / 'request.json').write_text(json.dumps({'packages': requirements, 'python': sys.executable}, indent=2))
    command = [sys.executable, '-I', '-m', 'pip', '--isolated', 'install',
               '--disable-pip-version-check', '--no-input', '--prefix', sys.prefix,
               '--constraint', str(constraints.resolve()), *requirements]
    log_path = directory / 'pip.log'
    error = None
    returncode = None
    with log_path.open('wb') as log:
        try:
            process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                returncode = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                error = 'Package installation exceeded 600 seconds'
        except OSError as exc:
            error = f'{type(exc).__name__}: {exc}'
    importlib.invalidate_caches()
    after = installed_packages()
    environment_path = gen_dir / 'model_environment.json'
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    environment['packages'] = after
    environment_path.write_text(json.dumps(environment, indent=2))
    with log_path.open('rb') as log:
        log.seek(max(0, log_path.stat().st_size - 8000))
        output = log.read().decode(errors='replace')
    result = {'status': 'installed' if returncode == 0 and error is None else 'error',
              'packages': requirements, 'python': sys.executable, 'returncode': returncode,
              'changed_packages': {name: version for name, version in after.items() if before.get(name) != version},
              'output': output, 'log_file': str(log_path),
              'note': 'Existing package versions are held fixed. No evaluation units charged.'}
    if error:
        result['error'] = error
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    return result
