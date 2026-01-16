from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def build_feature_pipeline():
    """
    构建 Scikit-Learn 预处理流水线。
    注意：这里的列名已更新为匹配 MaterialDataProcessor (ETL) 的输出。
    """
    # 1. 数值特征 (列名来自 material_data_processor.py)
    numeric_features = [
        'total_dopant_fraction',
        'average_dopant_radius',   # 原名为 avg_dopant_radius
        'average_dopant_valence',  # 原名为 avg_dopant_valence
        'number_of_dopants',       # 原名为 num_dopants
        'maximum_sintering_temperature', # 原名为 max_sinter_temp
        'total_sintering_duration'       # 原名为 total_sinter_time
    ]

    # 2. 分类特征
    categorical_features = [
        'synthesis_method',
        'primary_dopant_element'   # 原名为 primary_dopant
    ]

    # 3. 文本特征
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