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
	uvx ruff check
	uvx ruff format --check
	uv run cython-lint spector/*.pyx --ignore E501
	uvx ty check spector

html: all
	uv run python setup.py build_ext -i
	uv run --group docs mkdocs build
