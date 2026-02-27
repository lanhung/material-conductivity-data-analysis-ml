from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def build_feature_pipeline():
    """
    Build a Scikit-Learn preprocessing pipeline.
    Note: Column names here have been updated to match the output of MaterialDataProcessor (ETL).
    """
    # 1. Numeric features (column names from material_data_processor.py)
    numeric_features = [
        'total_dopant_fraction',
        'average_dopant_radius',   # formerly avg_dopant_radius
        'average_dopant_valence',  # formerly avg_dopant_valence
        'number_of_dopants',       # formerly num_dopants
        'maximum_sintering_temperature', # formerly max_sinter_temp
        'total_sintering_duration'       # formerly total_sinter_time
    ]

    # 2. Categorical features
    categorical_features = [
        'synthesis_method',
        'primary_dopant_element'   # formerly primary_dopant
    ]

    # 3. Text features
    text_feature = ['material_source_and_purity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features),

            # Text pipeline
            ('text', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('flatten', FunctionTransformer(lambda x: x.squeeze(), validate=False)),
                ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
                ('svd', TruncatedSVD(n_components=16))
            ]), text_feature)
        ]
    )
    return preprocessor
