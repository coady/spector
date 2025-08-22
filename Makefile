all: spector/*.pyx
	uv sync --no-install-project
	uv run --no-project cythonize $?

check: spector/*.pyx
	uv sync --no-install-project
	uv run --no-project cythonize -aX linetrace=True $?
	uv run python setup.py build_ext -i --define CYTHON_TRACE_NOGIL
	uv run pytest -s --cov

bench: all
	uv run python setup.py build_ext -i
	uv run pytest --codspeed

lint: all
	uv run ruff check .
	uv run ruff format --check .
	uv run cython-lint spector/*.pyx --ignore E501
	uv run mypy -p spector

html: all
	uv run python setup.py build_ext -i
	uv run --with spector mkdocs build
