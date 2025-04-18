# base_model.py
# MazeBaseModel

import cProfile
import csv
import io
import logging
import os
import pstats
import time
from configparser import ConfigParser
from functools import wraps
from typing import Optional, Callable, Any, TypeVar  # *new* Import required typing classes

import psutil
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

T = TypeVar('T', bound=Callable[..., Any])  # *new* Define T as a type variable for use in type annotations

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

TRAINING_PROGRESS_HTML = "training_progress.html"
TRAINING_PROGRESS_PNG = "training_progress.png"
LOSS_FILE = os.path.join(OUTPUT, "loss_data.csv")

class MazeBaseModel(nn.Module):
    """
    Base model for solving mazes using PyTorch.
    Subclasses should define their specific architectures by implementing the forward method.
    """

    def __init__(self):
        super(MazeBaseModel, self).__init__()
        self.model_name = "MazeBaseModel"  # Define the model name
        self.last_loss = 1

        # Set up early stopping patience to monitor overfitting
        self.patience = config.getint("DEFAULT", "patience", fallback=5)

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_weights(self):
        weights = []
        for name, param in self.named_parameters():
            # Consider only parameters with gradients and at least 2 dimensions (e.g., weight matrices)
            if param.requires_grad and param.ndim >= 2:
                weights.append((name, param.detach().cpu().numpy()))
        return weights

    def _compute_resource_usage(self):
        """
        Computes system resource usage (CPU load, GPU load, and RAM usage).

        Returns:
            cpu_load (float): CPU usage percentage.
            gpu_load (float): GPU memory utilization (if applicable).
            ram_usage (float): RAM usage in GB.
        """

        cpu_load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        gpu_load = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(
            0) if torch.cuda.is_available() else 0.0
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)
        return round(cpu_load, 0), round(gpu_load, 0), round(ram_usage, 0)

    def profile_method(output_file: Optional[str] = None) -> Callable[[T], T]:
        """Decorator for profiling a method"""

        def decorator(func: T) -> T:
            @wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                profiler = cProfile.Profile()
                profiler.enable()

                result = func(self, *args, **kwargs)

                profiler.disable()

                # Print stats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(30)  # Print top 30 time-consuming functions
                print(s.getvalue())

                # Optionally save to file
                if output_file:
                    ps.dump_stats(output_file)
                    print(f"Profile data saved to {output_file}")

                return result

            return wrapper

        return decorator

    @profile_method(output_file=f"{OUTPUT}train_model_profile.prof")
    def train_model(self, dataloader, num_epochs=20, learning_rate=0.0001, weight_decay=0.001,
                    device='cpu', tensorboard_writer=None, val_loader=None):
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
        logging.debug(f"Training {self.model_name} for {num_epochs} epochs on {device}...")

        # Record the start time
        start_time = time.time()

        # Check if we ar ein development mode.
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training with reduced data set.")
            num_epochs = 2

        self.to(device)

        improvement_threshold = config.getfloat("DEFAULT", "improvement_threshold", fallback=0.01)
        logging.info(
            f"Early stopping patience set to {self.patience} epochs. Training will stop after {self.patience} "
            f"if no improvement over {improvement_threshold}"
            f" epochs without improvement on training loss or validation loss."
        )

        # Define optimizer, learning rate scheduler, and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_factor = config.getfloat("DEFAULT", "lr_factor", fallback=0.7)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=self.patience)
        criterion = nn.CrossEntropyLoss()

        train_losses = {"train": [], "validation": []}  # Dictionary to store both training and validation losses
        train_accuracies = {"train": [], "validation": []}  # Dictionary to store accuracies

        # Set up a counter and best loss value for early stopping
        best_validation_loss = float("inf")
        early_stopping_counter = 0

        # Number of steps to accumulate for calculating average time per step
        # This helps provide a more stable time measurement
        ccumulation_steps = 2

        for epoch in range(num_epochs):
            # network performance monitoring
            running_loss: float = 0.0
            correct = 0
            total = 0

            cpu_loads = []
            gpu_loads = []
            ram_usages = []
            avg_cpu = 0
            avg_gpu = 0
            avg_ram = 0

            self.train()  # Put the model in training mode
            # Loop through batches from the data loader
            desc = f"Epoch {epoch + 1} Training Progress"
            iterator = tqdm(dataloader, desc=desc, leave=True)

            start_time = time.time()
            for iteration, batch in enumerate(iterator):
                logging.debug(f"Training batch {iteration + 1} of {len(dataloader)}")
                local_context, relative_position, target_action, steps_number = batch

                target_action = target_action.to(device).long()
                # Convert local_context to PyTorch tensor and ensure it's at least 2D
                local_context = torch.as_tensor(local_context, dtype=torch.float32, device=device)
                relative_position = torch.as_tensor(relative_position, dtype=torch.float32, device=device)

                # If local_context is 1D, convert it to 2D (batch_size, num_features)
                local_context = local_context.view(-1, local_context.size(-1))
                if relative_position.dim() == 1:
                    relative_position = relative_position.unsqueeze(0)

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

                # Concatenate features: local_context (4 values) + relative_position (2 values) + steps_number (1 value)
                inputs = torch.cat((local_context, relative_position, steps_number), dim=1)
                inputs.to(device)
                # Add sequence dimension for RNN input
                inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, sequence_length=1, num_features)

                assert inputs.shape[-1] == 7, f"Expected input features to be 7, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, (f"Expected target labels to be 1-dimensional, but got "
                                                  f"{target_action.dim()} dimensions")

                # Compute resource usage during the batch processing
                cpu_load, gpu_load, ram_usage = self._compute_resource_usage()

                # Forward pass
                outputs = self.forward(inputs)  # Pass input through the model
                outputs.to(device)
                loss = criterion(outputs, target_action)  # Compute cross-entropy loss

                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagate loss to calculate gradient
                optimizer.step()  # Update model weights

                # Call resource usage after processing the batch
                cpu_load, gpu_load, ram_usage = self._compute_resource_usage()
                cpu_loads.append(cpu_load)
                gpu_loads.append(gpu_load)
                ram_usages.append(ram_usage)
                avg_cpu = sum(cpu_loads) / len(cpu_loads)
                avg_gpu = sum(gpu_loads) / len(gpu_loads)
                avg_ram = sum(ram_usages) / len(ram_usages)

                # calculate loss
                running_loss += loss.item() * inputs.size(0)
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target_action.size(0)
                correct += (predicted == target_action).sum().item()

            # Calculate average time per step in milliseconds for this epoch
            # This helps users understand the processing delay
            time_per_step = int(((time.time() - start_time) / ccumulation_steps) * 1000)
            #Back Epoch loop. Compute loss and accuracy
            epoch_loss = running_loss / int(len(dataloader.dataset))
            training_accuracy = correct / total if total > 0 else 0.0

            # After finishing the training epoch and recording epoch_loss, do validation:
            # validation_loss=self._validate_model(val_loader = val_loader,criterion =  criterion, device = device, epoch = epoch)
            validation_loss, validation_accuracy = self._validate_model(val_loader=val_loader,
                                                                        criterion=criterion,
                                                                        device=device,
                                                                        epoch=epoch)

            train_losses["train"].append(epoch_loss)
            train_losses["validation"].append(validation_loss)
            train_accuracies["train"].append(training_accuracy)
            train_accuracies["validation"].append(validation_accuracy)

            # Monitoring
            self._monitor_training(
                epoch=epoch,
                num_epochs=num_epochs,
                epoch_loss=epoch_loss,
                scheduler=scheduler,
                validation_loss=validation_loss,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                time_per_step=time_per_step,
                cpu_load=avg_cpu,
                gpu_load=avg_gpu,
                ram_usage=avg_ram,
                tensorboard_writer=tensorboard_writer,
            )
            # Reset the accumulators for the next set of iterations
            cpu_loads.clear()
            gpu_loads.clear()
            ram_usages.clear()

            # Stop is no improvement on loss function
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate

            if validation_loss < best_validation_loss * (1 - improvement_threshold):
                best_validation_loss = validation_loss
                early_stopping_counter = 0  # Reset if improvement is achieved
            else:
                early_stopping_counter += 1

            logging.info(
                f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f} | Validation Loss = {validation_loss:.4f} |"
                f" Training Accuracy = {training_accuracy:.4f} | Validation Accuracy = {validation_accuracy:.4f} |"
                f" Learning Rate = {current_lr:.6f} | "
                f"Early Stopping Counter = {early_stopping_counter}")

            # Log performance metrics to explain processing time and resource usage
            logging.info(
                f"Performance: Time per step = {time_per_step} ms | CPU Load = {avg_cpu}% | "
                f"GPU Load = {avg_gpu}% | RAM Usage = {avg_ram} GB")

            # Trigger early stopping if no improvement within set patience and the learning rate is sufficiently low.
            if early_stopping_counter >= self.patience and current_lr < improvement_threshold:
                logging.info(
                    f"Early Stopping triggered at Epoch {epoch + 1} due to lack of validation loss improvement")
                break

        # After training is complete, record the end time and compute the duration
        training_duration = time.time() - start_time

        # Log the training duration. This uses the logging module, so make sure your logging is configured as desired.
        logging.info(f"Training time for model {self.model_name}: {training_duration:.2f} seconds")


        self.last_loss = train_losses["train"][-1] if train_losses["train"] else None
        logging.info(f"Training Complete for {self._get_name()}. Final Loss: {self.last_loss}")

        return self

    def _monitor_training(self, epoch, num_epochs, epoch_loss, scheduler, validation_loss=0,
                          training_accuracy=0, validation_accuracy=0,
                          time_per_step=0, cpu_load=0, gpu_load=0, ram_usage=0, tensorboard_writer=None):
        """
        Logs training information and updates tensorboard/histograms for monitoring.

        Parameters:
            epoch (int): Current epoch.
            num_epochs (int): Total number of epochs.
            epoch_loss (float): Training loss for the current epoch.
            scheduler: Learning rate scheduler.
            validation_loss (float): Validation loss for the current epoch.
            training_accuracy (float): Training accuracy for the current epoch.
            validation_accuracy (float): Validation accuracy for the current epoch.
            time_per_step (int): Average processing time per step in milliseconds.
            cpu_load (float): Average CPU utilization percentage during training.
            gpu_load (float): Average GPU memory utilization percentage (if available).
            ram_usage (float): Average RAM usage in GB during training.
            tensorboard_writer: Optional TensorBoard writer for logging.
        """
        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss:.4f}")
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training mazes will be loaded from the development folder.")
            num_epochs = 2

        scheduler.step(epoch_loss)
        with open(LOSS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.model_name, epoch + 1, epoch_loss, validation_loss,
                             training_accuracy, validation_accuracy, time.time(), time_per_step, cpu_load, gpu_load,
                             ram_usage])

        if config.getboolean("MONITORING", "tensorboard", fallback=False):
            # Log to tensorboard
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Loss/validation", validation_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/train", training_accuracy, epoch)
                tensorboard_writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)

                # Log weights and gradients
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
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                local_context, relative_position, target_action, steps_number = batch

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
                # Convert relative_position to a tensor if needed
                if not isinstance(relative_position, torch.Tensor):
                    relative_position = torch.as_tensor(relative_position, dtype=torch.float32, device=device)
                if relative_position.ndim == 1:
                    relative_position = relative_position.unsqueeze(0)

                # Ensure steps_number is a tensor with shape (batch_size, 1)
                if not isinstance(steps_number, torch.Tensor):
                    steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)
                if steps_number.ndim == 0:
                    steps_number = steps_number.unsqueeze(0)
                if steps_number.ndim == 1:
                    steps_number = steps_number.unsqueeze(1)

                # Concatenate features: local_context (4 values) + relative_position (2 values) + steps_number (1 value)
                inputs = torch.cat((local_context, relative_position, steps_number), dim=1)
                # Add a sequence dimension for RNN input: (batch_size, sequence_length, num_features)
                inputs = inputs.unsqueeze(1)

                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, target_action)
                val_loss_sum += loss.item()
                num_batches += 1

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target_action.size(0)
                correct += (predicted == target_action).sum().item()

        # After the validation loop
        average_val_loss = val_loss_sum / num_batches if num_batches > 0 else float('inf')
        validation_accuracy = correct / total if total > 0 else 0.0

        return average_val_loss, validation_accuracy
