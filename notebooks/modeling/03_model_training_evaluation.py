import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import os

sys.path.append('../../code/src')
from config import RANDOM_SEED, DATA_PATHS, MODEL_CONFIG

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.metrics import geometric_mean_score

import joblib

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Found unknown categories.*during transform')


class DataLoader:

    @staticmethod
    def load_X_data(path: str = None) -> pd.DataFrame:
        if path is None:
            path = DATA_PATHS['processed_X']
        
        try:
            data = pd.read_csv(path)
            print(f"  Lectura exitosa de X: {len(data)} filas")
            return data
        except FileNotFoundError:
            print(f"ERROR: No se encontró archivo en: {path}")
            raise
        except Exception as e:
            print(f"ERROR inesperado: {e}")
            raise
    
    @staticmethod
    def load_y_data(path: str = None) -> pd.DataFrame:
        if path is None:
            path = DATA_PATHS['processed_y']
        
        try:
            data = pd.read_csv(path)
            print(f"  Lectura exitosa de y: {len(data)} filas")
            return data
        except FileNotFoundError:
            print(f"ERROR: No se encontró archivo en: {path}")
            raise
        except Exception as e:
            print(f"ERROR inesperado: {e}")
            raise


class PreprocessingPipelineBuilder:

    def __init__(self):
        self.numericas = ['laufzeit', 'hoehe', 'alter']
        self.nominales = [
            'laufkont', 'moral', 'verw', 'sparkont', 'famges',
            'buerge', 'weitkred', 'wohn', 'pers', 'telef', 'gastarb'
        ]
        self.ordinales = [
            'beszeit', 'rate', 'wohnzeit', 'verm', 'bishkred', 'beruf'
        ]
        self.pipeline = None
    
    def build_pipeline(self) -> ColumnTransformer:
        numericas_pipe = Pipeline(steps=[
            ('impMediana', SimpleImputer(strategy='median')),
            ('escalaNum', MinMaxScaler(feature_range=(1, 2)))
        ])
        
        nominales_pipe = Pipeline(steps=[
            ('impModa', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        ordinales_pipe = Pipeline(steps=[
            ('impOrd', SimpleImputer(strategy='most_frequent')),
            ('ordtrasnf', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('numpipe', numericas_pipe, self.numericas),
                ('nominals', nominales_pipe, self.nominales),
                ('ordinales', ordinales_pipe, self.ordinales)
            ],
            remainder='passthrough'
        )
        
        return self.pipeline


class ModelEvaluator:

    def __init__(self, cv_folds=5, cv_repeats=3, random_state=None):
        self.cv_folds = cv_folds
        self.cv_repeats = cv_repeats
        self.random_state = random_state if random_state is not None else RANDOM_SEED
        self.scoring_metrics = {
            'miaccuracy': 'accuracy',
            'miprecision': 'precision',
            'mirecall': 'recall',
            'mifi': 'f1',
            'miauc': 'roc_auc',
            'miprauc': 'average_precision',
            'migmean': make_scorer(geometric_mean_score)
        }
    
    def evaluate_model(self, model, pipeline, X, y, model_name):
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_folds,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline,
                X,
                np.ravel(y),
                scoring=self.scoring_metrics,
                cv=cv,
                return_train_score=True
            )
        
        print(f'>> {model_name}')
        for j, k in enumerate(list(scores.keys())):
            if j > 1:
                print(f'\t {k} {np.nanmean(scores[k]):.4f} ({np.nanstd(scores[k]):.3f})')
        
        return scores


class ModelPipelineBuilder:

    def __init__(self, preprocessor, oversampling_method, model):
        self.preprocessor = preprocessor
        self.oversampling_method = oversampling_method
        self.model = model
    
    def build_pipeline(self) -> ImbPipeline:
        pipeline = ImbPipeline(steps=[
            ('preprocesamiento', self.preprocessor),
            ('sub_sobre_muestreo', self.oversampling_method),
            ('model', self.model)
        ])
        return pipeline


class ModelConfigurator:

    @staticmethod
    def create_logistic_regression(random_state=None):
        return LogisticRegression(
            penalty='l2',
            solver='newton-cg',
            max_iter=1000,
            C=1,
            random_state=random_state or RANDOM_SEED
        )
    
    @staticmethod
    def create_decision_tree(random_state=None):
        return DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=20,
            random_state=random_state or RANDOM_SEED
        )
    
    @staticmethod
    def create_random_forest(random_state=None):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_split=50,
            random_state=random_state or RANDOM_SEED
        )
    
    @staticmethod
    def create_svm(random_state=None):
        return SVC(
            kernel='rbf',
            C=10,
            gamma='auto',
            random_state=random_state or RANDOM_SEED
        )
    
    @staticmethod
    def create_xgboost(random_state=None):
        return XGBClassifier(
            booster='gbtree',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            random_state=random_state or RANDOM_SEED,
            n_jobs=-1
        )


class OversamplingConfigurator:

    @staticmethod
    def create_smote(random_state=None):
        return SMOTE(random_state=random_state or RANDOM_SEED)
    
    @staticmethod
    def create_borderline_smote(random_state=None, k_neighbors=5, m_neighbors=10):
        return BorderlineSMOTE(
            random_state=random_state or RANDOM_SEED,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors
        )


class ModelSaver:

    @staticmethod
    def save_model(model, output_dir='../../models', filename='best_model.joblib'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  Directorio '{output_dir}/' creado")
        
        output_file = os.path.join(output_dir, filename)
        joblib.dump(model, output_file)
        print(f"  Modelo guardado en: {output_file}")


class ModelTrainer:

    def __init__(self):
        self.preprocessor_builder = PreprocessingPipelineBuilder()
        self.evaluator = ModelEvaluator()
        self.model_configurator = ModelConfigurator()
        self.oversampling_configurator = OversamplingConfigurator()
        self.model_saver = ModelSaver()
        
        self.X_traintest = None
        self.y_traintest = None
        self.preprocessor = None
        self.models_results = {}
    
    def load_data(self):
        print("\n" + "=" * 60)
        print("PASO 1: Carga de Datos")
        print("=" * 60)
        
        self.X_traintest = DataLoader.load_X_data()
        self.y_traintest = DataLoader.load_y_data()
        
        print(f"\nShape X: {self.X_traintest.shape}")
        print(f"Shape y: {self.y_traintest.shape}")
        
        return self.X_traintest, self.y_traintest
    
    def build_preprocessing_pipeline(self):
        print("\n" + "=" * 60)
        print("PASO 2: Construcción de Pipeline de Preprocesamiento")
        print("=" * 60)
        
        self.preprocessor = self.preprocessor_builder.build_pipeline()
        print("  Pipeline de preprocesamiento construido")
        print(f"  Variables numéricas: {len(self.preprocessor_builder.numericas)}")
        print(f"  Variables nominales: {len(self.preprocessor_builder.nominales)}")
        print(f"  Variables ordinales: {len(self.preprocessor_builder.ordinales)}")
    
    def train_logistic_regression(self):
        print("\n" + "=" * 60)
        print("PASO 3: Entrenamiento - Regresión Logística")
        print("=" * 60)
        
        model = self.model_configurator.create_logistic_regression()
        metodo_uo = self.oversampling_configurator.create_borderline_smote()
        
        pipeline = ModelPipelineBuilder(self.preprocessor, metodo_uo, model).build_pipeline()
        scores = self.evaluator.evaluate_model(
            model, pipeline, self.X_traintest, self.y_traintest, "Regresión_Logística"
        )
        
        self.models_results['logistic_regression'] = {
            'model': model,
            'pipeline': pipeline,
            'scores': scores,
            'oversampling': metodo_uo
        }
        
        return model, pipeline, scores
    
    def train_decision_tree(self):
        print("\n" + "=" * 60)
        print("PASO 4: Entrenamiento - Decision Tree")
        print("=" * 60)
        
        model = self.model_configurator.create_decision_tree()
        metodo_uo = self.oversampling_configurator.create_smote()
        
        pipeline = ModelPipelineBuilder(self.preprocessor, metodo_uo, model).build_pipeline()
        scores = self.evaluator.evaluate_model(
            model, pipeline, self.X_traintest, self.y_traintest, "DecisionTree-DT"
        )
        
        self.models_results['decision_tree'] = {
            'model': model,
            'pipeline': pipeline,
            'scores': scores,
            'oversampling': metodo_uo
        }
        
        return model, pipeline, scores
    
    def train_random_forest(self):
        print("\n" + "=" * 60)
        print("PASO 5: Entrenamiento - Random Forest")
        print("=" * 60)
        
        model = self.model_configurator.create_random_forest()
        metodo_uo = self.oversampling_configurator.create_smote()
        
        pipeline = ModelPipelineBuilder(self.preprocessor, metodo_uo, model).build_pipeline()
        scores = self.evaluator.evaluate_model(
            model, pipeline, self.X_traintest, self.y_traintest, "Random Forest"
        )
        
        self.models_results['random_forest'] = {
            'model': model,
            'pipeline': pipeline,
            'scores': scores,
            'oversampling': metodo_uo
        }
        
        return model, pipeline, scores
    
    def train_svm(self):
        print("\n" + "=" * 60)
        print("PASO 6: Entrenamiento - Support Vector Machine")
        print("=" * 60)
        
        model = self.model_configurator.create_svm()
        metodo_uo = self.oversampling_configurator.create_smote()
        
        pipeline = ModelPipelineBuilder(self.preprocessor, metodo_uo, model).build_pipeline()
        scores = self.evaluator.evaluate_model(
            model, pipeline, self.X_traintest, self.y_traintest, "Support Vector Machine"
        )
        
        self.models_results['svm'] = {
            'model': model,
            'pipeline': pipeline,
            'scores': scores,
            'oversampling': metodo_uo
        }
        
        return model, pipeline, scores
    
    def train_xgboost(self):
        print("\n" + "=" * 60)
        print("PASO 7: Entrenamiento - XGBoost")
        print("=" * 60)
        
        model = self.model_configurator.create_xgboost()
        metodo_uo = self.oversampling_configurator.create_smote()
        
        pipeline = ModelPipelineBuilder(self.preprocessor, metodo_uo, model).build_pipeline()
        scores = self.evaluator.evaluate_model(
            model, pipeline, self.X_traintest, self.y_traintest, "XGBoosting"
        )
        
        self.models_results['xgboost'] = {
            'model': model,
            'pipeline': pipeline,
            'scores': scores,
            'oversampling': metodo_uo
        }
        
        return model, pipeline, scores
    
    def save_best_model(self):
        print("\n" + "=" * 60)
        print("PASO 8: Guardado del Mejor Modelo")
        print("=" * 60)
        
        best_model_name = 'logistic_regression'
        best_model = self.models_results[best_model_name]['model']
        
        print(f"  Modelo seleccionado: Regresión Logística")
        print("  Razón: Mejor rendimiento en ROC-AUC/PR-AUC y alta interpretabilidad")
        
        self.model_saver.save_model(best_model)
    
    def train_all_models(self):
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("Dataset: German Credit")
        print("=" * 60)
        
        self.load_data()
        self.build_preprocessing_pipeline()
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        self.save_best_model()
        
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return self.models_results


def main():

    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    print(f"\n  Modelos entrenados: {len(results)}")
    print(f"  Seed aleatoria: {RANDOM_SEED}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()

