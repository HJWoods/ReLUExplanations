import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from ExplanationEngine import ExplanationEngine, train_model, save_model, load_model

def load_mnist_data(batch_size=64):
    """Load MNIST dataset with preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_mnist_model(input_size=784, hidden_size=128, num_classes=10):
    """Create a ReLU network for MNIST classification."""
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )
    return model

if __name__ == "__main__":
    FORCE_RETRAIN = True  # Set to True to retrain even if model exists
    
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    print("Creating MNIST model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_mnist_model(input_size=784, hidden_size=32, num_classes=10)
    model = model.to(device)
    
    model_file = 'mnist_model.pth'
    if FORCE_RETRAIN:
        print("Force retrain enabled - training new model...")
        model = train_model(model, train_loader, test_loader, epochs=3)
        save_model(model, model_file)
    elif not load_model(model, model_file):
        print("Training new model...")
        model = train_model(model, train_loader, test_loader, epochs=3)
        save_model(model, model_file)
    else:
        print("Using pre-trained model.")
    
    engine = ExplanationEngine(model)
    
    print("\nGetting a test example...")
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    
    x = images[0].view(-1).numpy()  # Flatten to 1D array
    actual_label = labels[0].item()
    
    # Get model prediction
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(x_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    print(f"Test image - Actual: {actual_label}, Predicted: {predicted_label}")
    
    # Generate explanations
    print(f"\nGenerating explanations for class {predicted_label}...")
    why_res = engine.why(x)
    
    # Try to find why it's not classified as a different class
    counterfactual_class = (predicted_label + 1) % 10
    print(f"\nFinding why it's not classified as class {counterfactual_class}...")
    why_not_res = engine.why_not(x, counterfactual_cls=counterfactual_class, max_visited=60000)
    
    print("\n" + "="*50)
    print("WHY NOT EXPLANATION:")
    print(engine.format_why_not_explanation(why_not_res))
    print("\n" + "="*50)
    
    print("\nFirst few constraints from WHY explanation:")
    A = why_res.get("A")
    b = why_res.get("b")
    if A is not None and b is not None:
        for i in range(min(5, len(A))):
            print(f"Constraint {i+1}: {np.array2string(A[i][:10], precision=2)}... <= {b[i]:.4f}")
        if len(A) > 5:
            print(f"... and {len(A) - 5} more constraints")

