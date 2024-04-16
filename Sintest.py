import numpy as np
import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[3
                                                    ], [4]]))

    def forward(self, x):
        return self.linear(x)

# Custom loss function
def custom_loss(y_pred, y_true):
    # Apply sine to each prediction element-wise, then sum the outputs
    sin_y_pred = torch.sin(y_pred)
    sum_sin_y_pred = torch.sum(sin_y_pred, dim=1, keepdim=True)  # Ensure dimensions match y_true
    return torch.mean((y_true - sum_sin_y_pred) ** 2)

def compute_hessian_and_eigenvalues(model, data, target):
    """
    Compute the Hessian matrix and its eigenvalues for the weights of a neural network model.

    :param model: The neural network model.
    :param data: Input data (X).
    :param target: Target data (Y).
    :return: Hessian matrix and its eigenvalues.
    """
    # Forward pass
    output = model(data)
    # Compute loss
    loss = torch.mean((target - torch.sum(torch.sin(output), dim=1, keepdim=True)) ** 2)

    # First-order gradients (w.r.t weights)
    first_order_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Flatten the first-order gradients
    grads_flatten = torch.cat([g.contiguous().view(-1) for g in first_order_grads])

    # Hessian computation
    hessian = []
    for grad in grads_flatten:
        # Compute second-order gradients (w.r.t each element in the first-order gradients)
        second_order_grads = torch.autograd.grad(grad, model.parameters(), retain_graph=True)

        # Flatten and collect the second-order gradients
        hessian_row = torch.cat([g.contiguous().view(-1) for g in second_order_grads])
        hessian.append(hessian_row)

    # Stack to form the Hessian matrix
    hessian_matrix = torch.stack(hessian)

    # Compute eigenvalues
    eigenvalues, _ = torch.linalg.eig(hessian_matrix)

    return hessian_matrix, eigenvalues

# Note: To use this function, you'll need to provide your neural network model, the input data (X), and the target data (Y).


def check_local_minimum(eigenvalues):
    # Check if all eigenvalues have a positive real part
    if all(eig.real > 0 for eig in eigenvalues):
        print("This is a local minimum.")
    else:
        print("This is not a local minimum.")

     

# Initialize the model and optimizer
model = SimpleLinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Data for training
inputs = torch.tensor([[np.pi/4], [np.pi/2],[3*np.pi/4],[np.pi]], dtype=torch.float32)

targets = torch.tensor([[2.23], [0.71],[0.16],[1.0]], dtype=torch.float32)

# Number of epochs to train
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Forward pass: Compute predicted y by passing x to the model
    outputs = model(inputs)

    # Compute loss
    loss = custom_loss(outputs, targets)

    # Backward pass: Compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

    # Log weights
    print(f"Epoch {epoch+1}/{num_epochs}, Weights: {model.linear.weight.data.numpy()}")

# This setup will print the weight matrices after each epoch, showing how they are updated.

hessian_matrix_central, eigenvalues_central = compute_hessian_and_eigenvalues(model, inputs, targets)

print(eigenvalues_central)
check_local_minimum(eigenvalues_central)
