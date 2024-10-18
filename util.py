import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_train_test_val_loader(X, Y, batch_size=32):
    dataset = TensorDataset(X, Y)
    num_samples = len(dataset)
    train_size = int(0.7*num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


def train_model(model, train_loader, val_loader, model_id):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.unsqueeze(1).float()
            batch_Y = batch_Y.unsqueeze(1).float()
            output = model(batch_X)
            loss = criterion(output, batch_Y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_X = val_X.unsqueeze(1).float()
                val_Y = val_Y.unsqueeze(1).float()
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_Y).item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_id}.pth')  # Save model weights
            print("Model saved!")
    
        
    print("training complete")


def test_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            
            if batch_X.dim() == 2:
                batch_X = batch_X.unsqueeze(1)
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_Y)
            total_loss += loss.item()
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(batch_Y.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)

    # Calculate error statistics
    errors = all_predictions - all_targets
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Print results
    print(f'Average Loss: {total_loss / len(test_loader):.6f}')
    print(f'Mean Absolute Error (MAE): {mae:.6f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')
    print(f'R-squared (R2): {r2:.6f}')
    print(f'Mean Error: {mean_error:.6f}')
    print(f'Standard Deviation of Error: {std_error:.6f}')
    print(f'Min Target: {np.min(all_targets):.6f}, Max Target: {np.max(all_targets):.6f}')
    print(f'Min Prediction: {np.min(all_predictions):.6f}, Max Prediction: {np.max(all_predictions):.6f}')
    
    # Print histogram of errors
    print("\nError Distribution:")
    hist, bin_edges = np.histogram(errors, bins=10)
    for i in range(len(hist)):
        print(f'{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {hist[i]}')

    return all_targets, all_predictions