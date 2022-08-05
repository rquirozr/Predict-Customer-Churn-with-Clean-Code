# constants.py

PTH_DATA = 'data/bank_data.csv'

PTH_EDA_CHURN = './images/eda/churn_distribution.png'
PTH_EDA_CUSTOMER_AGE = './images/eda/customer_age_distribution.png'
PTH_EDA_MARITAL_STATUS = './images/eda/marital_status_distribution.png'
PTH_EDA_TOTAL_TRANS_CT = './images/eda/total_trans_ct_histogram.png'
PTH_EDA_CORRELATION = './images/eda/correlation_heatmap.png'

PTH_MODELS_RFC = './models/rfc_model.pkl'
PTH_MODELS_LOGISTIC = './models/logistic_model.pkl'

PTH_RESULTS_ROC_CURVE = './images/results/plot_roc_curve.png'
PTH_RESULTS_EXPLAINER = './images/results/explainer.png'
PTH_RESULTS_CLASSIFICATION_REPORT_TRAIN = './images/results/classification_report_train.png'
PTH_RESULTS_CLASSIFICATION_REPORT_TEST = './images/results/classification_report_test.png'

CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]