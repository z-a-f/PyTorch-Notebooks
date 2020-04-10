import torch

def running_loss_accuracy(loss, y_hat, y_true, total_data, running_vals):
    """Computes the running loss and accuracy.

    Args:
        loss: Torch loss function (evaluated)
        y_hat, y_true: Predicted and true target values
        total_data: Total amount of data being used in the current run
        running_vals: Tuple consisting of
            - Running loss
            - Running accuracy

    Returns:
        running_loss
        running_accuracy
    """
    current_loss, current_accuracy = running_vals
    fraction_of_data = len(y_true) / total_data
    current_loss += loss.item() * fraction_of_data
    y_predict = (y_hat >= 0.5).to(torch.float)
    current_accuracy += (y_predict == y_true).to(torch.float).mean().item() * fraction_of_data
    return current_loss, current_accuracy
