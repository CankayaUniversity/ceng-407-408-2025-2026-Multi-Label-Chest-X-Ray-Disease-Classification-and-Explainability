from torch import optim
import torch
import torch.nn as nn
from models.full_model import DenseNetCBAM


def verify_training_readiness(num_classes=14, batch_size=4, epochs=1):
    print("--- Starting Model Training Verification ---")

    model = DenseNetCBAM(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_target = torch.randint(0, num_classes, (batch_size,))

    print(f"Model instantiated successfully.")
    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Dummy Target Shape: {dummy_target.shape}")

    try:
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            output = model(dummy_input)

            loss = criterion(output, dummy_target)

            loss.backward()

            optimizer.step()

            print(f"\nEpoch {epoch + 1}/{epochs} successfully completed.")
            print(f"Loss value: {loss.item():.4f}")

            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item()

            if total_grad_norm > 0:
                print(f"Total gradient norm: {total_grad_norm:.4f}")
                print(
                    "Model is structurally ready for training (Forward, Backward, and Optimization steps confirmed).")
            else:
                print("Warning: Gradients appear to be zero. Check for potential issues.")

        print("\n--- Verification Complete: Model passed all structural tests. ---")

    except Exception as e:
        print("\nThe model failed the training verification step.")
        print(f"Error Details: {e}")
        print("This indicates a dimension mismatch or connection error that the dummy forward pass did not catch.")


if __name__ == '__main__':
    verify_training_readiness()