# base_model.py
# MazeBaseModel

import torch.nn as nn
import torch
from torch import optim
import logging
import torch.optim.lr_scheduler as lr_scheduler
import csv
import os

OUTPUT = "output/"
TRAINING_PROGRESS_HTML = "training_progress.html"
TRAINING_PROGRESS_PNG = "training_progress.png"
LOSS_FILE = os.path.join(OUTPUT, "loss_data.csv")
os.makedirs(OUTPUT, exist_ok=True)


class MazeBaseModel(nn.Module):
    """
    Base model for solving mazes using PyTorch.
    Subclasses should define their specific architectures by implementing the forward method.
    """

    def __init__(self):
        super(MazeBaseModel, self).__init__()
        self.model_name = "MazeBaseModel"  # Define the model name

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_model(self, dataloader, val_loder, num_epochs=20, learning_rate=0.001, weight_decay=0.001,
                    device='cpu', tensorboard_writer=None):
        """
        Generic training loop using CrossEntropyLoss and Adam optimizer.
    
        Parameters:
            dataloader (DataLoader): The data loader providing input data and labels for training.
            num_epochs (int): The number of epochs for training the model.
            learning_rate (float): The learning rate for the optimizer.
            training_samples (int): Maximum number of training samples handled per epoch.
            weight_decay (float): Weight decay (L2 penalty) for regularization in the optimizer.
            device (str): The device (e.g., 'cpu' or 'cuda') used for training.
            tensorboard_writer: Optional TensorBoard writer for logging.
    
        Returns:
            self: The trained model or its final loss after training.
        """
        self.to(device)
        # Set up early stopping patience to monitor overfitting
        patience = 5
        trigger_times = 0

        # Define optimizer, learning rate scheduler, and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience)
        criterion = nn.CrossEntropyLoss()


        train_losses = []  # List to store the epoch losses for training

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.train()  # Put the model in training mode
            # Loop through batches from the data loader
            for iteration, (local_context, target_action, steps_number) in enumerate(dataloader):
                target_action = target_action.to(device)
                # Convert local_context to PyTorch tensor and ensure it's at least 2D
                local_context = torch.tensor(local_context, dtype=torch.float32, device=device)

                # If local_context is 1D, convert it to 2D (batch_size, num_features)
                if local_context.dim() == 1:
                    local_context = local_context.unsqueeze(0)  # Convert shape (num_features,) → (1, num_features)

                # Convert steps_number to PyTorch tensor and ensure it's at least 2D
                steps_number = torch.tensor(steps_number, dtype=torch.float32, device=device)

                # If steps_number is 0D (a single scalar), make it 1D
                if steps_number.dim() == 0:
                    steps_number = steps_number.unsqueeze(0)  # Convert scalar to (1,)

                # Make sure steps_number is (batch_size, 1)
                steps_number = steps_number.unsqueeze(1)  # Convert shape (1,) → (1, 1)

                # Ensure both tensors are now 2D before concatenation
                assert local_context.dim() == 2, f"local_context has wrong shape: {local_context.shape}"
                assert steps_number.dim() == 2, f"steps_number has wrong shape: {steps_number.shape}"

                # Concatenate along the feature dimension
                inputs = torch.cat((local_context, steps_number), dim=1)

                # Add sequence dimension for RNN input
                inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, sequence_length=1, num_features)

                assert inputs.shape[-1] == 5, f"Expected input features to be 5, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, f"Expected target labels to be 1-dimensional, but got {target_action.dim()} dimensions"

                # Forward pass
                outputs = self.forward(inputs)  # Pass input through the model
                loss = criterion(outputs, target_action)  # Compute cross-entropy loss

                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagate loss to calculate gradients
                optimizer.step()  # Update model weights

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)

            #Moitoring
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            scheduler.step(epoch_loss)
            with open(LOSS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name, epoch + 1, epoch_loss])

            train_losses.append(epoch_loss)

            self.eval()  # Put model in evaluation mode for validation
            val_loss_sum = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loder:
                    local_context, target_action, steps_number = batch
                    # Convert target_action to a tensor and force it to be 1D (batch dimension)
                    if isinstance(target_action, (list, tuple)):
                        # If it's already a list of integers (when batch_size > 1)
                        target_action = torch.tensor(target_action, dtype=torch.long, device=device)
                    else:
                        # For a single sample (batch_size == 1), wrap it in a list to get shape (1,)
                        target_action = torch.tensor([target_action], dtype=torch.long, device=device)
                    # Ensure local_context is a PyTorch tensor and at least 2D
                    if not isinstance(local_context, torch.Tensor):
                        local_context = torch.as_tensor(local_context, dtype=torch.float32, device=device)
                    if local_context.ndim == 1:
                        local_context = local_context.unsqueeze(0)  # (features,) -> (1, features)

                    # Ensure steps_number is a PyTorch tensor and reshape to (batch_size, 1)
                    if not isinstance(steps_number, torch.Tensor):
                        steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)
                    if steps_number.ndim == 0:
                        steps_number = steps_number.unsqueeze(0)  # scalar -> (1,)
                    if steps_number.ndim == 1:
                        steps_number = steps_number.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)

                    # Now both tensors should be 2D.
                    # For instance: local_context -> (batch_size, num_features) and steps_number -> (batch_size, 1)
                    inputs = torch.cat((local_context, steps_number), dim=1)
                    # Add the sequence dimension for RNN input (resulting in shape: (batch_size, sequence_length=1, num_features+1))
                    inputs = inputs.unsqueeze(1)

                    # Forward pass
                    outputs = self.forward(inputs)  # Pass through mode

                    loss = criterion(outputs, target_action)  # Define your loss_function as per your training setup
                    val_loss_sum += loss.item()

                    num_batches += 1

            # Compute average validation loss
            val_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/epoch", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Loss/Validation", val_loss, epoch)

                # Log weight updates and gradient norms
                for name, param in self.named_parameters():
                    tensorboard_writer.add_histogram(f"Weights/{name}", param, epoch)
                    if param.grad is not None:
                        tensorboard_writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

            if epoch_loss < min(train_losses):
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info(f"Early Stopping Triggered at Epoch {epoch + 1}")
                    break

        if tensorboard_writer:
            tensorboard_writer.close()

        last_loss = train_losses[-1] if train_losses else None
        logging.info(f"Training Complete for {self._get_name()}. Final Loss: {last_loss}")
        return self, last_loss
