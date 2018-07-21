import sys
py2 = sys.version_info < (3,)


def pytest_ignore_collect(path, config):
    return path.basename == 'test_benchmark.py' and (py2 or config.option.verbose < 0)
