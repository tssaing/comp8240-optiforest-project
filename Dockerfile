FROM python:3.8-slim

WORKDIR /app

# The pinned versions come from the authors' README and must not move.
RUN pip install --no-cache-dir numpy==1.20.1 scikit-learn==0.22.1 \
    pandas==1.4.1 scipy

# ECOD baseline, for the comparison study.
#
# pyod and the packages below are installed with --no-deps so that pip cannot
# pull a newer numpy or scikit-learn over the pins above and silently
# invalidate every reproduction result.
#
# Everything here is pinned to a version contemporary with numpy 1.20:
#   numba 0.54.1 is the last release that supports numpy 1.20. Later versions
#   refuse to import with "Numba needs NumPy 1.22 or greater".
#   llvmlite 0.37.0 is the matching build for numba 0.54.
#   statsmodels 0.12.2 and patsy 0.5.2 are from the same period.
RUN pip install --no-cache-dir --no-deps pyod==0.9.8 && \
    pip install --no-cache-dir --no-deps \
        combo==0.1.3 statsmodels==0.12.2 numba==0.54.1 llvmlite==0.37.0 \
        patsy==0.5.2

# pyod.models.ecod imports matplotlib for an optional plotting helper, so it is
# needed even though no plots are drawn here.
#
# Pillow is pinned first because current releases use numpy.typing.NDArray,
# which was added in numpy 1.21 and does not exist in 1.20. Installing
# matplotlib afterwards then finds Pillow already satisfied and leaves it alone.
RUN pip install --no-cache-dir "Pillow==9.5.0"
RUN pip install --no-cache-dir matplotlib==3.5.3

# Fail the build rather than ship an image that is quietly broken. The first
# check catches a moved pin, the second catches pyod failing to import.
RUN python -c "import numpy, sklearn; \
    assert numpy.__version__ == '1.20.1', 'numpy moved to ' + numpy.__version__; \
    assert sklearn.__version__ == '0.22.1', 'sklearn moved to ' + sklearn.__version__; \
    print('pins intact:', numpy.__version__, sklearn.__version__)"

RUN python -c "from pyod.models.ecod import ECOD; print('pyod ECOD imports')"

COPY . .

CMD ["python", "demo.py", "--dataset=ad", "--threshold=403", "--branch=0"]
