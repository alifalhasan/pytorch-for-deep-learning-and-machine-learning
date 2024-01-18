import torch
from torch import nn
import matplotlib.pyplot as plt



# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"



# Create some data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias



# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]



# Plot the data
def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

plot_prediction(X_train, y_train, X_test, y_test)



# Building a PyTorch Linear Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() # Constructor
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, # this implies number of inputs
                                      out_features=1)  # this implies number of outputs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
        


# Set the manual seed
torch.manual_seed(42)
model = LinearRegressionModel()
print(model.state_dict())



# Set the model to use the target device
model.to(device)



# Training
## Setup Loss Function
loss_fn = nn.L1Loss()

## Setup Optimizer
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

## Training Loop
torch.manual_seed(42)

epochs = 200

# Put data on the target device(device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model.train()

    # Forwared Pass
    y_pred = model(X_train)

    # Calculate Loss
    loss = loss_fn(y_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Perform backpropagation
    loss.backward()

    # Optimizer step
    optimizer.step()



    ## Testing
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch%10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")




# Making and evaluating predictions
model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
plot_prediction(predictions = y_pred.cpu())



# Saving and Loading trained model
from pathlib import Path

## Create model's directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

## Create model save path
MODEL_NAME = "1-pytorch-workflow.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

## Save the model's state dict
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)



# Load a pytorch model
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.to(device)