'''
This module performs unit testings for churn_library.py

Author: Ricardo Quiroz
Date: Jun 28, 2022
'''

import logging
import pytest
import churn_library as cls
import constants as c

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data(c.PTH_DATA)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    pytest.df = df
    return df


def test_eda():
    '''
    test perform eda function
    '''
    df = pytest.df
    cls.perform_eda(df)
    for image_name, pth in [
        ("churn_distribution", c.PTH_EDA_CHURN),
        ("customer_age_distribution", c.PTH_EDA_CUSTOMER_AGE),
        ("marital_status_distribution", c.PTH_EDA_MARITAL_STATUS),
        ("total_trans_ct_histogram", c.PTH_EDA_TOTAL_TRANS_CT),
            ("correlation_heatmap", c.PTH_EDA_CORRELATION), ]:
        try:
            with open(pth, 'r', encoding="utf-8"):
                logging.info(
                    "Testing perform_eda png: SUCCESS for image %s", image_name)
        except FileNotFoundError as err:
            logging.error(
                "Testing perform_eda png: Image %s missing", image_name)
            raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = pytest.df
    try:
        df_encoded = cls.encoder_helper(df, c.CATEGORY_LST)
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: DataFrame doesn't appear to have rows or columns")
        raise err
    try:
        for category in c.CATEGORY_LST:
            assert f'{category}_Churn' in df_encoded.columns
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: DataFrame doesn't have the necessary encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")

    pytest.df_encoded = df_encoded
    return df_encoded


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_encoded = pytest.df_encoded
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_encoded)
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Train and Test split done wrong")
        raise err

    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test
    return X_train, X_test, y_train, y_test


def test_train_models():
    '''
    test train_models
    '''
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test
    cls.train_models(X_train, X_test, y_train, y_test)
    for pth in [
            c.PTH_RESULTS_ROC_CURVE,
            c.PTH_MODELS_RFC,
            c.PTH_MODELS_LOGISTIC,
            c.PTH_RESULTS_EXPLAINER,
            c.PTH_RESULTS_CLASSIFICATION_REPORT_TRAIN,
            c.PTH_RESULTS_CLASSIFICATION_REPORT_TEST, ]:
        try:
            with open(pth, 'r', encoding="utf-8"):
                logging.info(
                    "Testing test_train_models png: SUCCESS for image %s", pth)
        except FileNotFoundError as err:
            logging.error(
                "Testing test_train_models png: Image %s missing", pth)
            raise err


if __name__ == '__main__':
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
