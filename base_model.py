# base_model.py
# MazeBaseModel

import torch.nn as nn
import torch
from torch import optim
import logging
import torch.optim.lr_scheduler as lr_scheduler
import csv
import os
from torch.utils.data import DataLoader


OUTPUT = "output/"
TRAINING_PROGRESS_HTML = "training_progress.html"
TRAINING_PROGRESS_PNG = "training_progress.png"
LOSS_FILE = os.path.join(OUTPUT, "loss_data.csv")

os.makedirs(OUTPUT, exist_ok=True)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

class MazeBaseModel(nn.Module):
    """
    Base model for solving mazes using PyTorch.
    Subclasses should define their specific architectures by implementing the forward method.
    """

    def __init__(self):
        super(MazeBaseModel, self).__init__()
        self.model_name = "MazeBaseModel"  # Define the model name
        self.last_loss = 1

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_model(self, dataloader, val_loader, num_epochs=20, learning_rate=0.001, weight_decay=0.001,
                    device='cpu', tensorboard_writer=None):
        """
        Generic training loop using CrossEntropyLoss and Adam optimizer.
    
        Parameters:
            dataloader (DataLoader): The data loader providing input data and labels for training.
            val_loader (ValDataLoader): The data loader providing input data and labels for validation.
            num_epochs (int): The number of epochs for training the model.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for regularization in the optimizer.
            device (str): The device (e.g., 'cpu' or 'cuda') used for training.
            tensorboard_writer: Optional TensorBoard writer for logging.
    
        Returns:
            self: The trained model or its final loss after training.
        """
        self.to(device)
        # Set up early stopping patience to monitor overfitting
        patience = 5
        loss_trigger_times = 0
        validation_loss_trigger_times = 0

        # Define optimizer, learning rate scheduler, and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience)
        criterion = nn.CrossEntropyLoss()

        train_losses = {"train": [], "validation": []}  # Dictionary to store both training and validation losses

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.train()  # Put the model in training mode
            # Loop through batches from the data loader
            for iteration, (local_context, target_action, steps_number) in enumerate(dataloader):
                target_action = target_action.to(device).long()
                # Convert local_context to PyTorch tensor and ensure it's at least 2D
                local_context = torch.as_tensor(local_context, dtype=torch.float32, device=device)

                # If local_context is 1D, convert it to 2D (batch_size, num_features)
                if local_context.dim() == 1:
                    local_context = local_context.unsqueeze(0)  # Convert shape (num_features,) → (1, num_features)

                # Convert steps_number to PyTorch tensor and ensure it's at least 2D
                steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)

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

            # After finishing the training epoch and recording epoch_loss, do validation:
            validation_loss=self._validate_model(val_loader = val_loader,criterion =  criterion, device = device, epoch = epoch)
            train_losses["train"].append(epoch_loss)
            train_losses["validation"].append(validation_loss)
            #Stop is no improvement on loss function

            STOP_ON_LOSS = False
            if epoch_loss < min(train_losses["train"]) or STOP_ON_LOSS:
                loss_trigger_times = 0
            else:
                loss_trigger_times += 1
                if loss_trigger_times >= patience:
                    logging.info(f"Early Stopping Loss Triggered at Epoch {epoch + 1}")
                    break

            #Stop if no improvement on validation loss.
            STOP_ON_VALIDATION_LOSS = True
            if validation_loss < min(train_losses["validation"]) or STOP_ON_VALIDATION_LOSS:
                validation_loss_trigger_times = 0
            else:
                validation_loss_trigger_times += 1
                if validation_loss_trigger_times >= patience:
                    logging.info(f"Early Stopping Validation Loss Triggered at Epoch {epoch + 1}")
                    break

            # Moitoring
            self._monitor_training(epoch, num_epochs, epoch_loss, scheduler, validation_loss, tensorboard_writer)

        self.last_loss = train_losses["train"][-1] if train_losses["train"] else None
        logging.info(f"Training Complete for {self._get_name()}. Final Loss: {self.last_loss}")
        return self

    def _monitor_training(self, epoch, num_epochs, epoch_loss, scheduler, validation_loss, tensorboard_writer):
        """
        Logs training information and updates tensorboard/histograms for monitoring.

        Parameters:
            epoch (int): Current epoch.
            num_epochs (int): Total number of epochs.
            epoch_loss (float): Training loss for the current epoch.
            scheduler: Learning rate scheduler.
            validation_loss (float): Validation loss for the current epoch.
            tensorboard_writer: Optional TensorBoard writer for logging.
        """
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)
        with open(LOSS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.model_name, epoch + 1, epoch_loss, validation_loss])

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Loss/epoch", epoch_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Validation", validation_loss, epoch)
            for name, param in self.named_parameters():
                tensorboard_writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    tensorboard_writer.add_histogram(f"Gradients/{name}", param.grad, epoch)


    def _validate_model(self, val_loader, criterion, device, epoch):
            """
            Runs the validation phase and logs metrics.

            Parameters:
                val_loader: Validation DataLoader.
                criterion: Loss function.
                device: Device to use (e.g. 'cpu' or 'cuda').
                epoch: Current epoch (for logging purposes).

            Returns:
                Average validation loss.
            """
            self.eval()  # Set model to evaluation mode
            val_loss_sum = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    local_context, target_action, steps_number = batch

                    # Convert target_action to a tensor ensuring it has a batch dimension
                    if isinstance(target_action, (list, tuple)):
                        target_action = torch.tensor(target_action, dtype=torch.long).to(device)
                    else:
                        target_action = torch.tensor([target_action], dtype=torch.long, device=device)

                    # Ensure local_context is a tensor and at least 2D
                    if not isinstance(local_context, torch.Tensor):
                        local_context = torch.as_tensor(local_context, dtype=torch.float32, device=device)
                    if local_context.ndim == 1:
                        local_context = local_context.unsqueeze(0)

                    # Ensure steps_number is a tensor with shape (batch_size, 1)
                    if not isinstance(steps_number, torch.Tensor):
                        steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)
                    if steps_number.ndim == 0:
                        steps_number = steps_number.unsqueeze(0)
                    if steps_number.ndim == 1:
                        steps_number = steps_number.unsqueeze(1)

                    # Concatenate local_context and steps_number along the feature dimension
                    inputs = torch.cat((local_context, steps_number), dim=1)
                    # Add a sequence dimension for RNN input: (batch_size, sequence_length, num_features)
                    inputs = inputs.unsqueeze(1)

                    # Forward pass
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, target_action)
                    val_loss_sum += loss.item()
                    num_batches += 1

            # After the validation loop
            average_val_loss = val_loss_sum / num_batches if num_batches > 0 else float('inf')

            return average_val_loss
