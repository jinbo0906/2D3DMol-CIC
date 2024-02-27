from typing import Callable

import torch
import torch.nn as nn
import numpy as np


def get_loss_func(run_conf) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    # Nested dictionary of the form {dataset_type: {loss_function: loss_function callable}}
    supported_loss_functions = {
        "regression": {
            "mse": nn.MSELoss(reduction="none"),
            "bounded_mse": bounded_mse_loss,
            "mve": normal_mve,
            "evidential": evidential_loss,
        }
    }

    dataset_type = run_conf['train_conf']['dataset_type']
    # Error if no loss function supported
    if dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{dataset_type}" not supported.')

    # Return loss function if it is represented in the supported_loss_functions dictionary
    loss_function_choices = supported_loss_functions.get(dataset_type, dict())
    loss_function = loss_function_choices.get(run_conf['train_conf']['loss_function'])

    if loss_function is not None:
        return loss_function

    else:
        raise ValueError(
            f'Loss function "{loss_function}" not supported with dataset type {dataset_type}. \
            Available options for that dataset type are {loss_function_choices.keys()}.'
        )


def normal_mve(pred_values, targets):
    """
    Use the negative log likelihood function of a normal distribution as a loss function used for making
    simultaneous predictions of the mean and error distribution variance simultaneously.

    :param pred_values: Combined predictions of means and variances of shape(data, tasks*2).
                        Means are first in dimension 1, followed by variances.
    :return: A tensor loss value.
    """
    # Unpack combined prediction values
    pred_means, pred_var = torch.split(pred_values, pred_values.shape[1] // 2, dim=1)

    return torch.log(2 * np.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (
        2 * pred_var
    )


def bounded_mse_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    less_than_target: torch.tensor,
    greater_than_target: torch.tensor,
) -> torch.tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param less_than_target: A tensor with boolean values indicating whether the target is a less-than inequality.
    :param greater_than_target: A tensor with boolean values indicating whether the target is a greater-than inequality.
    :return: A tensor containing loss values of shape(batch_size, tasks).
    """
    predictions = torch.where(
        torch.logical_and(predictions < targets, less_than_target), targets, predictions
    )

    predictions = torch.where(
        torch.logical_and(predictions > targets, greater_than_target),
        targets,
        predictions,
    )

    return nn.functional.mse_loss(predictions, targets, reduction="none")


def evidential_loss(pred_values, targets, lam=0, epsilon=1e-8):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :param pred_values: Combined prediction values for mu, v, alpha, and beta parameters in shape(data, tasks*4).
                        Order in dimension 1 is mu, v, alpha, beta.
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Unpack combined prediction values
    mu, v, alpha, beta = torch.split(pred_values, pred_values.shape[1] // 4, dim=1)

    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss
