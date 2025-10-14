import torch.nn as nn


def conf_predict(triples, probabilities, model):
    """The goal of evaluate task is to predict the confidence of triples.

    """
    pred_score = model(triples)
    mae_loss = nn.L1Loss()
    mae = mae_loss(pred_score, probabilities)

    mse_loss = nn.MSELoss()
    mse = mse_loss(pred_score, probabilities)

    return mse, mae


def print_results(mse_value, mae_value):
    header = "Confidence Prediction Results"
    separator = "=" * 48

    print(header)
    print(separator)
    print(f"| {'Metric':<35} | {'Value':<10} |")
    print(separator)
    print(f"| {'Mean Squared Error (MSE)':<35} | {mse_value:<10.5f} |")
    print(f"| {'Mean Absolute Error (MAE)':<35} | {mae_value:<10.5f} |")
    print(separator)
