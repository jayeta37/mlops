from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

classification_pipeline = Pipeline(
    [
        ('Mean Imputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('Mode Imputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('Domain Processing', pp.DomainProcessing(variables_to_modify=config.FEATURES_TO_MODIFY, variables_to_add=config.FEATURES_TO_ADD)),
        ('Drop Columns', pp.DropColumns(columns_to_drop=config.DROP_FEATURES)),
        ('Label Encoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Log Transformation', pp.LogTransform(columns_to_log=config.LOG_FEATURES)),
        ('Min Max Scale', MinMaxScaler()),
        ("Logistic Classifier", LogisticRegression(random_state=0))
    ]
)