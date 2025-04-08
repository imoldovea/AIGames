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

import psutil
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        self.lr_scheduler = None

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

    def profile_method(output_file=None):
        """Decorator for profiling a method"""

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
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
    def train_model(self, dataloader, val_loader, num_epochs=20, learning_rate=0.0001, weight_decay=0.001,
                    momentum=0.9, optimizer_type='adam', scheduler_type='plateau', scheduler_params={},
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
            exit_weight: Weight for the exit prediction loss (default: 1.0)

        Returns:
            self: The trained model or its final loss after training.
        """
        logging.debug(f"Training {self.model_name} for {num_epochs} epochs on {device}...")

        # Record the start time
        start_time = time.time()

        # Check if we ar ein development mode.
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training mazes will be loaded from the development folder.")
            num_epochs

        exit_weight = config.getfloat("DEFAULT", "exit_weight", fallback=1.0)
        self.to(device)

        # Set up early stopping patience to monitor overfitting
        patience = config.getint("DEFAULT", "patience", fallback=5)
        improvement_threshold = config.getfloat("DEFAULT", "improvement_threshold", fallback=0.01)
        logging.info(
            f"Early stopping patience set to {patience} epochs. Training will stop after {patience} "
            f"if no improvement over {improvement_threshold}"
            f" epochs without improvement on training loss or validation loss."
        )

        # Define optimizer, learning rate scheduler, and loss function
        # Setup optimizer based on type
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum,
                                        weight_decay=weight_decay)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                            momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Setup learning rate scheduler
        lr_factor = config.getfloat("DEFAULT", "lr_factor", fallback=0.7)
        if scheduler_type.lower() == 'plateau':
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor,
                                                               patience=patience)
        elif scheduler_type.lower() == 'step':
            step_size = int(scheduler_params.get('step_size', num_epochs // 3))
            gamma = float(scheduler_params.get('gamma', 0.1))
            self.lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'cosine':
            t_max = scheduler_params.get('t_max', num_epochs)
            eta_min = float(scheduler_params.get('eta_min', 0))
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else:
            self.lr_scheduler = None

        # Loss functions
        action_criterion = nn.CrossEntropyLoss()
        exit_criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits for exit prediction

        criterion = nn.CrossEntropyLoss()

        train_losses = {"train": [], "validation": []}  # Dictionary to store both training and validation losses
        train_accuracies = {"train": [], "validation": []}  # Dictionary to store accuracies

        # Set up a counter and best loss value for early stopping
        best_validation_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            # network performance monitoring
            running_loss = 0.0
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
                local_context, relative_position, target_action, steps_number, exit_target = batch

                # Move input data to device
                local_context = local_context.to(device)
                relative_position = relative_position.to(device)

                # Forward pass
                outputs = self.forward(local_context, relative_position)

                target_action = target_action.to(device).long()
                exit_target = exit_target.to(device).float()  # Float for BCE loss

                # Convert local_context to PyTorch tensor and ensure it's at least 2D
                local_context = torch.as_tensor(local_context, dtype=torch.float32, device=device)
                relative_position = torch.as_tensor(relative_position, dtype=torch.float32, device=device)

                # If local_context is 1D, convert it to 2D (batch_size, num_features)
                local_context = local_context.view(-1, local_context.size(-1))
                if relative_position.dim() == 1:
                    relative_position = relative_position.unsqueeze(0)
                # If steps_number is 0D (a single scalar), make it 1D
                if steps_number.dim() == 0:
                    steps_number = steps_number.unsqueeze(0)  # Convert scalar to (1,)

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

                # Convert steps_number to a tensor if needed
                if not isinstance(steps_number, torch.Tensor):
                    steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)
                # Ensure steps_number has the correct dimensions
                if steps_number.ndim == 0:
                    steps_number = steps_number.unsqueeze(0)  # Add batch dimension
                if steps_number.ndim == 1:
                    steps_number = steps_number.unsqueeze(1)  # Add feature dimension

                # Concatenate features: local_context (4 values) + relative_position (2 values) + steps_number (1 value)
                inputs = torch.cat((local_context, relative_position, steps_number), dim=1)
                inputs = inputs.to(device)

                # Add sequence dimension for RNN input
                inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, sequence_length=1, num_features)

                assert inputs.shape[-1] == 7, f"Expected input features to be 7, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, (f"Expected target labels to be 1-dimensional, but got "
                                                  f"{target_action.dim()} dimensions")

                # Compute resource usage during the batch processing
                cpu_load, gpu_load, ram_usage = self._compute_resource_usage()

                # Forward pass
                # Return two outputs from forward
                # Ensure outputs are on the correct device
                action_logits, exit_logits = outputs
                exit_logits = exit_logits.to(device)

                action_loss = action_criterion(action_logits, target_action)
                # Calculate exit prediction loss if target data is available
                if exit_target is not None:
                    exit_loss = exit_criterion(exit_logits, exit_target)
                    # Combined loss (you might want to weight these differently)
                    total_loss = action_loss + exit_weight * exit_loss
                else:
                    exit_loss = 0  # Added default value
                    total_loss = action_loss

                # Use total_loss for both backpropagation and running loss calculation
                running_loss += total_loss.item() * inputs.size(0)

                optimizer.zero_grad()  # Clear previous gradients
                total_loss.backward()  # Backpropagate loss to calculate gradient
                optimizer.step()  # Update model weights

                # Call resource usage after processing the batch
                cpu_load, gpu_load, ram_usage = self._compute_resource_usage()
                cpu_loads.append(cpu_load)
                gpu_loads.append(gpu_load)
                ram_usages.append(ram_usage)
                avg_cpu = sum(cpu_loads) / len(cpu_loads)
                avg_gpu = sum(gpu_loads) / len(gpu_loads)
                avg_ram = sum(ram_usages) / len(ram_usages)

                #calculate loss
                running_loss += total_loss.item() * inputs.size(0)
                # Calculate training accuracy
                _, predicted = torch.max(action_logits.data, 1)
                total += target_action.size(0)
                correct += (predicted == target_action).sum().item()

            time_per_step = (time.time() - start_time) / len(iterator)
            #Back Epoch loop. Compute loss and accuracy
            epoch_loss = running_loss / len(dataloader.dataset)
            training_accuracy = correct / total if total > 0 else 0.0

            # After finishing the training epoch and recording epoch_loss, do validation:
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
                scheduler=self.lr_scheduler,
                validation_loss=validation_loss,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                time_per_step=time_per_step,
                cpu_load=avg_cpu,
                gpu_load=avg_gpu,
                ram_usage=avg_ram,
                tensorboard_writer=tensorboard_writer,
            )

            # Add this code to step the scheduler
            if scheduler_type.lower() in ['step', 'cosine']:
                self.lr_scheduler.step()
            elif scheduler_type.lower() == 'plateau':
                self.lr_scheduler.step(validation_loss)  # For ReduceLROnPlateau, you need to pass the validation loss

            # Reset the accumulators for the next set of iterations
            cpu_loads.clear()
            gpu_loads.clear()
            ram_usages.clear()

            # With this code that handles different scheduler types:
            if scheduler_type.lower() == 'plateau':
                # For ReduceLROnPlateau, get the learning rate from the optimizer
                current_lr = optimizer.param_groups[0]['lr']
            else:
                # For other schedulers that have get_last_lr() method
                current_lr = self.lr_scheduler.get_last_lr()[0]

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

            # Trigger early stopping if no improvement within set patience and the learning rate is sufficiently low.
            if early_stopping_counter >= patience and current_lr < improvement_threshold:
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
    action_correct = 0
    action_total = 0
    exit_correct = 0
    exit_total = 0
    has_exit_prediction = hasattr(self, 'exit_predictor')

    # Check if we have separate criteria for action and exit prediction
    if isinstance(criterion, tuple) and len(criterion) == 2:
        action_criterion, exit_criterion = criterion
    else:
        action_criterion = criterion
        exit_criterion = nn.BCEWithLogitsLoss()  # Default exit criterion

    with torch.no_grad():
        for batch in val_loader:
            local_context, relative_position, target_action, steps_number = batch[:4]
            target_exit = batch[4] if len(batch) > 4 else None

            # Convert target_action to a tensor ensuring it has a batch dimension
            if isinstance(target_action, (list, tuple)):
                target_action = torch.tensor(target_action, dtype=torch.long).to(device)
            else:
                target_action = torch.tensor([target_action], dtype=torch.long, device=device)

            # Process target_exit if provided
            if target_exit is not None:
                if isinstance(target_exit, (list, tuple)):
                    target_exit = torch.tensor(target_exit, dtype=torch.float32).to(device)
                else:
                    target_exit = torch.tensor([target_exit], dtype=torch.float32, device=device)

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

            # Convert steps_number to a tensor if needed
            if not isinstance(steps_number, torch.Tensor):
                steps_number = torch.as_tensor(steps_number, dtype=torch.float32, device=device)
            # Ensure steps_number has the correct dimensions
            if steps_number.ndim == 0:
                steps_number = steps_number.unsqueeze(0)  # Add batch dimension
            if steps_number.ndim == 1:
                steps_number = steps_number.unsqueeze(1)  # Add feature dimension

            # Concatenate features: local_context (4 values) + relative_position (2 values) + steps_number (1 value)
            inputs = torch.cat((local_context, relative_position, steps_number), dim=1)
            # Add a sequence dimension for RNN input: (batch_size, sequence_length, num_features)
            inputs = inputs.unsqueeze(1)

            # Forward pass - handle both cases (with and without exit_predictor)
            outputs = self.forward(inputs)

            if has_exit_prediction:
                # Model returns a tuple of (action_logits, exit_logits)
                action_logits, exit_logits = outputs

                # Handle different types of criterion
                if isinstance(criterion, tuple) and len(criterion) == 2:
                    action_loss = action_criterion(action_logits, target_action)

                    if target_exit is not None:
                        exit_loss = exit_criterion(exit_logits, target_exit)
                        loss = action_loss + exit_loss
                    else:
                        loss = action_loss
                else:
                    # Single criterion for action prediction only
                    action_loss = criterion(action_logits, target_action)
                    loss = action_loss
            else:
                # Legacy model with single output
                action_logits = outputs
                loss = criterion(action_logits, target_action)

            val_loss_sum += loss.item()
            num_batches += 1

            # Calculate accuracy - use action_logits for consistency
            if has_exit_prediction:
                _, predicted_actions = torch.max(action_logits.data, 1)
            else:
                _, predicted_actions = torch.max(outputs.data, 1)

            action_total += target_action.size(0)
            action_correct += (predicted_actions == target_action).sum().item()

            # Calculate exit prediction accuracy if target data is available
            if has_exit_prediction and target_exit is not None:
                predicted_exits = (torch.sigmoid(exit_logits) > 0.5).float()
                exit_total += target_exit.size(0)
                exit_correct += (predicted_exits == target_exit).sum().item()

    # After the validation loop
    average_val_loss = val_loss_sum / num_batches if num_batches > 0 else float('inf')
    action_accuracy = action_correct / action_total if action_total > 0 else 0.0

    # Only return exit accuracy if we have exit data
    if exit_total > 0:
        exit_accuracy = exit_correct / exit_total
        return average_val_loss, action_accuracy, exit_accuracy
    else:
        return average_val_loss, action_accuracy, None
