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
        logging.debug(f"Training {self.model_name} for {num_epochs} epochs on {device}...")

        start_time = time.time()
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training with reduced data set.")
            num_epochs = 2

        self.to(device)
        improvement_threshold = config.getfloat("DEFAULT", "improvement_threshold", fallback=0.01)
        logging.info(
            f"Early stopping patience set to {self.patience} epochs. Training will stop after {self.patience} "
            f"if no improvement over {improvement_threshold} epochs."
        )

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_factor = config.getfloat("DEFAULT", "lr_factor", fallback=0.7)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=self.patience)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        train_losses = {"train": [], "validation": []}
        train_accuracies = {"train": [], "validation": []}
        best_validation_loss = float("inf")
        early_stopping_counter = 0
        accumulation_steps = 2

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            self.train()
            desc = f"Epoch {epoch + 1} Training Progress"
            iterator = tqdm(dataloader, desc=desc, leave=True)

            epoch_start = time.time()
            for iteration, batch in enumerate(iterator):
                # Assume `batch` is a tuple: (inputs, target_actions)
                #   inputs   : shape (batch_size, seq_len, input_size)
                #   target_actions : shape (batch_size, seq_len)
                inputs, target_actions = batch
                inputs = inputs.to(device)
                target_actions = target_actions.to(device)

                # Forward pass produces outputs of shape [batch_size, seq_len, output_size]
                outputs = self.forward(inputs)

                # Validate shapes (optional assertions)
                assert outputs.ndim == 3, f"Expected outputs to be 3D, got {outputs.shape}"
                assert inputs.ndim == 3, f"Expected inputs to be 3D, got {inputs.shape}"

                # Flatten the sequence dimension so that CrossEntropyLoss can be applied
                batch_size, seq_len, output_size = outputs.shape
                outputs_flat = outputs.contiguous().view(batch_size * seq_len, output_size)
                targets_flat = target_actions.contiguous().view(batch_size * seq_len)

                loss = criterion(outputs_flat, targets_flat)

                # ✅ Check for invalid loss values before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"⚠️ Invalid loss encountered at epoch {epoch + 1}, batch {iteration + 1}")
                    raise ValueError(f"Invalid loss encountered {epoch + 1}, batch {iteration + 1}")

                # update progress bar with loss
                iterator.set_postfix(loss=loss.item())

                optimizer.zero_grad()
                loss.backward()
                # ✅ Add this line for gradient clipping (right after backward)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # ✅ Per-batch logging every 10 iterations
                if iteration % 10 == 0:
                    grad_norm = sum((p.grad.norm().item() ** 2 for p in self.parameters() if p.grad is not None)) ** 0.5
                    logging.debug(
                        f"Epoch {epoch + 1} | Batch {iteration + 1} | Loss: {loss.item():.4f} | GradNorm: {grad_norm:.4f}")
                    if tensorboard_writer:
                        step = epoch * len(dataloader) + iteration
                        tensorboard_writer.add_scalar("BatchLoss/train", loss.item(), step)
                        tensorboard_writer.add_scalar("GradientNorm/train", grad_norm, step)

                optimizer.step()

                running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs_flat, 1)
                total += targets_flat.size(0)
                correct += (predicted == targets_flat).sum().item()

            epoch_loss = running_loss / len(dataloader.dataset)
            training_accuracy = correct / total if total > 0 else 0.0

            # Validate model (using val_loader with similar reshaping steps)...
            validation_loss, validation_accuracy = self._validate_model(val_loader, criterion, device, epoch)

            train_losses["train"].append(epoch_loss)
            train_losses["validation"].append(validation_loss)
            train_accuracies["train"].append(training_accuracy)
            train_accuracies["validation"].append(validation_accuracy)

            self._monitor_training(
                epoch=epoch,
                num_epochs=num_epochs,
                epoch_loss=epoch_loss,
                scheduler=scheduler,
                validation_loss=validation_loss,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                time_per_step=int(((time.time() - epoch_start) / accumulation_steps) * 1000),
                tensorboard_writer=tensorboard_writer,
            )

            current_lr = scheduler.get_last_lr()[0]
            if validation_loss < best_validation_loss * (1 - improvement_threshold):
                best_validation_loss = validation_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            logging.info(
                f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f} | Validation Loss = {validation_loss:.4f} |"
                f" Training Accuracy = {training_accuracy:.4f} | Validation Accuracy = {validation_accuracy:.4f} |"
                f" Learning Rate = {current_lr:.6f} | "
                f"Early Stopping Counter = {early_stopping_counter}"
            )

            if early_stopping_counter >= self.patience and current_lr < improvement_threshold:
                logging.info(f"Early Stopping triggered at Epoch {epoch + 1}")
                break

        training_duration = time.time() - start_time
        logging.info(f"Training time for model {self.model_name}: {training_duration:.2f} seconds")
        self.last_loss = train_losses["train"][-1] if train_losses["train"] else None
        logging.info(f"Training Complete for {self.model_name}. Final Loss: {self.last_loss}")
        return self

    def _monitor_training(self, epoch, num_epochs, epoch_loss, scheduler, validation_loss=0,
                          training_accuracy=0, validation_accuracy=0,
                          time_per_step=0, tensorboard_writer=None):
        # Retrieve actual resource usage
        cpu_load, gpu_load, ram_usage = self._compute_resource_usage()

        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss:.4f}")

        # Update learning rate and log resource usage
        scheduler.step(epoch_loss)
        with open(LOSS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            write_header = not os.path.exists(LOSS_FILE) or os.path.getsize(LOSS_FILE) == 0
            with open(LOSS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "model_name", "epoch", "train_loss", "val_loss",
                        "train_acc", "val_acc", "timestamp", "time_per_step",
                        "cpu_load", "gpu_load", "ram_usage"
                    ])
                writer.writerow([self.model_name, epoch + 1, epoch_loss, validation_loss,
                                 training_accuracy, validation_accuracy, time.time(), time_per_step,
                                 cpu_load, gpu_load, ram_usage])

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Loss/train", epoch_loss, epoch)
            tensorboard_writer.add_scalar("Loss/validation", validation_loss, epoch)
            tensorboard_writer.add_scalar("Accuracy/train", training_accuracy, epoch)
            tensorboard_writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)
            tensorboard_writer.add_scalar("Resource/CPU_load", cpu_load, epoch)
            tensorboard_writer.add_scalar("Resource/GPU_load", gpu_load, epoch)
            tensorboard_writer.add_scalar("Resource/RAM_usage", ram_usage, epoch)

            for name, param in self.named_parameters():
                tensorboard_writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    tensorboard_writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    def _validate_model(self, val_loader, criterion, device, epoch):
        """
        Runs the validation phase using sequence inputs.
        Parameters:
            val_loader: Validation DataLoader, yielding batches of (inputs, target_actions)
                        where inputs: [batch_size, seq_len, input_size]
                        and target_actions: [batch_size, seq_len]
            criterion: Loss function.
            device (str): Device to use (e.g. 'cpu' or 'cuda').
            epoch (int): Current epoch (for logging purposes).
        Returns:
            A tuple of (average validation loss, validation accuracy)
        """
        self.eval()  # Set model to evaluation mode
        val_loss_sum = 0.0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Each batch is a tuple (inputs, target_actions)
                inputs, target_actions = batch
                inputs = inputs.to(device)  # shape: (batch_size, seq_len, input_size)
                target_actions = target_actions.to(device)  # shape: (batch_size, seq_len)

                # Forward pass: expecting outputs of shape (batch_size, seq_len, output_size)
                outputs = self.forward(inputs)

                # Flatten the outputs and targets to shape [batch_size * seq_len, ...]
                batch_size, seq_len, output_size = outputs.shape
                outputs_flat = outputs.contiguous().view(batch_size * seq_len, output_size)
                targets_flat = target_actions.contiguous().view(batch_size * seq_len)

                # Compute loss using the flattened tensors
                loss = criterion(outputs_flat, targets_flat)

                # ✅ Check for invalid loss values before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid loss encountered {epoch + 1}")
                val_loss_sum += loss.item()
                num_batches += 1

                # Calculate accuracy over the flattened outputs as well
                _, predicted = torch.max(outputs_flat, 1)
                total += targets_flat.size(0)
                correct += (predicted == targets_flat).sum().item()

        average_val_loss = val_loss_sum / num_batches if num_batches > 0 else float("inf")
        validation_accuracy = correct / total if total > 0 else 0.0

        return average_val_loss, validation_accuracy