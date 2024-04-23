import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleLinearModel(nn.Module):
    def __init__(self, seed=None):
        super(SimpleLinearModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        else: 
            torch.manual_seed(6) 
        self.linear = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            # randomize the initial omega
            self.linear.weight.copy_(torch.randn(2, 1))
            #self.linear.weight.copy_(torch.tensor([[3], [4]]))

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

def train_model(seed, inputs, targets, num_epochs=100, lr= 0.05, print_every=10):
    # Initialize the model and optimizer with the given seed
    model = SimpleLinearModel(seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Dictionary to store the loss history
    loss_history = []


    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = custom_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Record the loss
        loss_history.append(loss.item())

        # Print every 'print_every' epochs
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Weights: {model.linear.weight.data.numpy()}")

        if epoch + 1 == num_epochs:
            hessian_matrix_central, eigenvalues_central = compute_hessian_and_eigenvalues(model, inputs, targets)
            print(eigenvalues_central)
            check_local_minimum(eigenvalues_central)
        

    return loss_history, model.linear.weight.data.numpy()


# Data for training
inputs = torch.tensor([[np.pi/4], [np.pi/2],[3*np.pi/4],[np.pi]], dtype=torch.float32)

targets = torch.tensor([[2.23], [0.71],[0.16],[1.0]], dtype=torch.float32)

loss_histories = {}
weights = {}

# Train the model for different seeds
for i in range(20,40):  # Adjust the range for more seeds
    print(f"Training with seed {i}")
    loss_history, weight = train_model(seed=i, inputs=inputs, targets=targets, num_epochs=300, lr=0.05, print_every=10)
    loss_histories[i] = loss_history
    weights[i] = weight

# Example of how to access and print the loss history for a specific seed
#print(len(loss_histories[5]))  # Length of loss history for seed 5
#print(loss_histories[5])  # Loss history for seed 5

plt.figure(figsize=(10, 5))
for seed, history in loss_histories.items():
    plt.plot(history, label=f'Seed {seed}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History by Seed')

# Place the legend on the left side outside of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

plt.show()


#hessian_matrix_central, eigenvalues_central = compute_hessian_and_eigenvalues(model, inputs, targets)

#print(eigenvalues_central)
#check_local_minimum(eigenvalues_central)
