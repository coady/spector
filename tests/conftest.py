from importlib import metadata


def pytest_report_header(config):
    return 'numpy-' + metadata.version('numpy')
