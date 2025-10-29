"""
Modeling module for MLOps project.

Este módulo contiene la funcionalidad para entrenar y realizar predicciones con modelos.
"""

from mlops_project.modeling.predict import load_model, predict
from mlops_project.modeling.train import evaluate_model, train_model

__all__ = ["train_model", "evaluate_model", "predict", "load_model"]
