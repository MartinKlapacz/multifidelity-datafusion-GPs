import pytest
from src.MFDataFusion import MultifidelityDataFusion
import src.data.exampleCurves2D as ex2D
import src.data.exampleCurves1D as ex1D
import numpy as np
from unittest.mock import patch 
import matplotlib.pyplot as plt


def test_adaptation_improves_mse():
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = ex2D.get_curve1(num_hf=5, num_lf=80)
    model = MultifidelityDataFusion(
        name='model',
        input_dim=2,
        tau=.001,
        num_derivatives=2,
        f_high=f_high,
        f_low=f_low,
        use_composite_kernel=True,
    )

    model.fit(X_train_hf)
    mse_before = model.get_mse(X_test, y_test)
    model.adapt(5)
    mse_after = model.get_mse(X_test, y_test)
    assert mse_after < mse_before

# def test_model_with_adapted_input_better_than_model_with_random_input():
#     X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = ex1D.get_curve4(num_hf=5, num_lf=80)
#     model1 = MultifidelityDataFusion(
#         name='model',
#         input_dim=1,
#         tau=.001,
#         num_derivatives=2,
#         f_high=f_high,
#         f_low=f_low,
#         use_composite_kernel=True,
#     )
#     model1.fit(X_train_hf)
#     model1.adapt(15)
#     mse1 = model1.get_mse(X_test, y_test)

#     X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = ex1D.get_curve4(num_hf=20, num_lf=80)
#     model2 = MultifidelityDataFusion(
#         name='model',
#         input_dim=1,
#         tau=.001,
#         num_derivatives=2,
#         f_high=f_high,
#         f_low=f_low,
#         use_composite_kernel=True,
#     )
#     model2.fit(X_train_hf)
#     mse2 = model2.get_mse(X_test, y_test)
#     assert mse1 < mse2