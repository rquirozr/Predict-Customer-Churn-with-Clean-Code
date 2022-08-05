# conftest.py
import pytest

def pytest_configure():
    pytest.df = None
    pytest.df_encoded = None
    pytest.X_train = None
    pytest.X_test = None
    pytest.y_train = None
    pytest.y_test = None