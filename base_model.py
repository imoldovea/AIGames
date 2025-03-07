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

    def train_model(self, dataloader, valloder, num_epochs=20, learning_rate=0.001, training_samples=100, weight_decay=0.001,
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

        self.train()  # Put the model in training mode
        train_losses = []  # List to store the epoch losses for training

        for epoch in range(num_epochs):
            running_loss = 0.0

            # Loop through batches from the data loader
            for iteration, (local_context, target_action, steps_number) in enumerate(dataloader):
                target_action = target_action.to(device)
                local_context = torch.tensor(local_context).to(device).float()  # Load batch and send to device
                steps_number = steps_number.to(device).unsqueeze(
                    1).float()  # Ensure step number shape is [batch_size, 1]
                inputs = torch.cat((local_context, steps_number), dim=1)  # Concatenate context and steps
                inputs = inputs.unsqueeze(1)  # Add channel dimension for compatibility with 1D conv layers (if used)

                assert inputs.shape[-1] == 5, f"Expected input features to be 5, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, f"Expected target labels to be 1-dimensional, but got {target_action.dim()} dimensions"

                # Forward pass
                outputs = self.forward(inputs)  # Pass input through the model
                loss = criterion(outputs, target_action)  # Compute cross-entropy loss

                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagate loss to calculate gradients
                optimizer.step()  # Update model weights

                running_loss += loss.item() * inputs.size(0)

                if iteration + 1 >= training_samples:
                    logging.info(f"Training samples limit: Epoch {epoch + 1}/{num_epochs}, Iteration {iteration + 1}/{training_samples}, Loss: {loss.item():.4f}")
                    break


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

            # No gradients are needed for validation
            with torch.no_grad():
                for data, labels in valloder:  #
                    data, labels = data.to(device), labels.to(device)  # Move data to GPU/CPU as appropriate

                    # Forward pass
                    outputs = self(data)

                    # Compute loss
                    loss = criterion(outputs, labels)  # Define your loss_function as per your training setup
                    val_loss_sum += loss.item()

                    num_batches += 1

            # Compute average validation loss
            val_loss = val_loss_sum / num_batches

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
