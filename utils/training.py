# Support functions

def evaluate_batch(model, loss_fn, x_data, y_data):
    """Runs a single batch evaluation of the model and returns the loss.

    Args:
        model: PyTorch model
        loss_fn: Torch loss function
        x_data, y_data: data to be evaluated

    Returns:
        y_hat: Target prediction
        loss: PyTorch loss function (evaluated)
    """
    y_hat = model(x_data)  # Forward pass
    loss = loss_fn(y_hat, y_data)  # Loss
    return y_hat, loss

def train_batch(model, loss_fn, optimizer, x_data, y_data):
    """Runs a single batch training iteration on a model amd returns the loss.

    Args:
        model: Torch model
        loss_fn: Torch loss function
        optimizer: Optimizer function
        x_data, y_data: data to be evaluated

    Returns:
        y_hat: Target prediction
        loss: Torch loss function (evaluated)
    """
    optimizer.zero_grad()  # Reset the optimizer
    y_hat, loss = evaluate_batch(model, loss_fn, x_data, y_data)  # Forward and Loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update pass
    return y_hat, loss

def run_epoch(model, data_iter, loss_fn, optimizer=None, performance_fn=None,
              preprocess_fn=None, mode='eval', device=None):
    """Runs a single epoch over all batches.

    Args:
        model: PyTorch model
        data_iter: PyTorch data iterator
        loss_fn: Loss function
        optimizer: Optimizer to use for training
        performance_fn: Function to evaluate the performance.
            This function should accept the following arguments:
            loss, y_hat, y_true, total_data, running_vals
        preprocess_fn: Function to preprocess the data.
            This function should accept the following arguments:
            x_batch, y_batch
        mode: current running mode. Could be either 'train' or 'eval'
        device: Optional device to send the data to

    Returns:
        running_loss
        running_accuracy
    """
    total_data = len(data_iter.data())
    running_loss = 0.0
    running_accuracy = 0.0
    if performance_fn is None:
        raise ValueError('Need the performance function!')
    if mode == 'train' and optimizer is None:
        raise ValueError('Need optimizer for training!')
    for batch_example in data_iter:
        x_batch = batch_example.text.pin_memory()
        y_batch = batch_example.label.pin_memory()
        if preprocess_fn is not None:
            x_batch, y_batch = preprocess_fn(x_batch, y_batch)
        if device is not None:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
        if mode == 'train':
            model.train()
            y_hat, loss = train_batch(model, loss_fn, optimizer, x_batch, y_batch)
        else:
            model.eval()
            y_hat, loss = evaluate_batch(model, loss_fn, x_batch, y_batch)
        running_loss, running_accuracy = performance_fn(loss, y_hat, y_batch, total_data,
                                                        (running_loss, running_accuracy))
    return running_loss, running_accuracy
