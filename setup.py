from setuptools import Extension, setup

ext_module = Extension(
    "spector.vector",
    ["spector/vector.cpp"],
    define_macros=[("Py_LIMITED_API", "0x030B0000")],
    py_limited_api=True,
)
setup(ext_modules=[ext_module])
