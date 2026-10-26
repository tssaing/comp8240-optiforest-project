FROM python:3.8-slim

WORKDIR /app

# Versions pinned in the authors' README.
RUN pip install --no-cache-dir numpy==1.20.1 scikit-learn==0.22.1 \
    pandas==1.4.1 scipy

# ECOD baseline. --no-deps stops pip replacing the pinned numpy and scikit-learn.
# numba 0.54.1 is the last release that works with numpy 1.20.
RUN pip install --no-cache-dir --no-deps pyod==0.9.8 && \
    pip install --no-cache-dir --no-deps \
        combo==0.1.3 statsmodels==0.12.2 numba==0.54.1 llvmlite==0.37.0 \
        patsy==0.5.2

# pyod needs matplotlib. Pillow is pinned first so it works with numpy 1.20.
RUN pip install --no-cache-dir "Pillow==9.5.0"
RUN pip install --no-cache-dir matplotlib==3.5.3

# Fail the build if a pin moved or pyod cannot be imported.
RUN python -c "import numpy, sklearn; \
    assert numpy.__version__ == '1.20.1', 'numpy moved to ' + numpy.__version__; \
    assert sklearn.__version__ == '0.22.1', 'sklearn moved to ' + sklearn.__version__; \
    print('pins intact:', numpy.__version__, sklearn.__version__)"

RUN python -c "from pyod.models.ecod import ECOD; print('pyod ECOD imports')"

COPY . .

CMD ["python", "demo.py", "--dataset=ad", "--threshold=403", "--branch=0"]
