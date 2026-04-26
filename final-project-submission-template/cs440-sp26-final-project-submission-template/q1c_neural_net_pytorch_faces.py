"""Template for a 3 layer feed forward neural network for binary face
classification, implemented with PyTorch.

This is Part 1(c) on the face task. You are expected to use `torch.nn`,
autograd, and `torch.optim`.

Required public API (fixed for auto grading):
  * class `PyTorchNeuralNetworkFaces` (a `torch.nn.Module` subclass)
    with a `forward` method.
  * class `PyTorchFacesClassifier` wrapper with `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1c_neural_net_pytorch_faces.py <training_percent>
"""

import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this file. Install with `pip install torch`."
    ) from exc

from util_faces import load_faces, flatten_images


class PyTorchNeuralNetworkFaces(nn.Module):
    """Three layer MLP: 4200 to hidden1 to hidden2 to 2 (face, not face)."""

    def __init__(self, input_size: int = 70 * 60,
                 hidden1_size: int = 128,
                 hidden2_size: int = 64,
                 output_size: int = 2):
        """Construct `nn.Linear` and activation modules for each layer."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning raw logits of shape (N, 2)."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PyTorchFacesClassifier:
    """Thin wrapper that drives training and prediction for the module."""

    def __init__(
        self,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Build the module, the loss, and the optimiser."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchNeuralNetworkFaces(
            input_size=70*60,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=2
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Fit the PyTorch model on the provided training data."""
        # Convert numpy to tensors
        X = flatten_images(training_images)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(training_labels).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, image: np.ndarray) -> int:
        """Predict 0 or 1 for a single 70x60 image."""
        # Flatten image
        x = image.reshape(1, -1)
        # Convert to tensor
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            pred = torch.argmax(outputs, dim=1).item()
        
        return pred

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # Flatten all images
        X = flatten_images(images)
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(labels).to(self.device)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)
        
        # Calculate accuracy
        correct = (preds == y_tensor).sum().item()
        return correct / len(labels)


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the PyTorch NN on faces."""
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        clf = PyTorchFacesClassifier()
        start = time.time()
        clf.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = clf.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== PyTorch NN | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
