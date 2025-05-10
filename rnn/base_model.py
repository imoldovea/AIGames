# base_model.py
# MazeBaseModel

import csv
import json
import logging
import os
import time
from configparser import ConfigParser

import psutil
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.amp import GradScaler  # ⬅️ NEW import
from tqdm import tqdm

from utils import profile_method

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

TRAINING_PROGRESS_HTML = "training_progress.html"
TRAINING_PROGRESS_PNG = "training_progress.png"
LOSS_FILE = os.path.join(OUTPUT, "loss_data.csv")
LOSS_JSON_FILE = os.path.join(OUTPUT, "loss_data.json")


class MazeBaseModel(nn.Module):
    """
    Base model for solving mazes using PyTorch.
    Subclasses should define their specific architectures by implementing the forward method.
    """

    def __init__(self):
        super(MazeBaseModel, self).__init__()
        self.model_name = "MazeBaseModel"  # Define the model name
        self.last_loss = 1
        self.exit_weight = config.getfloat("DEFAULT", "exit_weight", fallback=5.0)
        # Set up early stopping patience to monitor overfitting
        self.patience = config.getint("DEFAULT", "patience", fallback=5)
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.temp_scaler = TemperatureScaler()

    def _build_one_hot_target(self, targets_flat, exit_weight):
        """
        Converts a flattened sequence of target class indices into a one-hot encoded tensor,
        applying special handling for padding and the exit class.

        This function:
        - Masks out padding values (-100)
        - Converts valid class indices [0–4] into one-hot vectors
        - Applies `exit_weight` to the "exit" class (index 4)
        - Sets all padded rows to 0s (ignored in loss)

        Args:
            targets_flat (Tensor): Flattened 1D tensor of class indices, may contain -100 for padding
            exit_weight (float): Multiplier for the exit class logits

        Returns:
            Tensor: One-hot encoded target tensor of shape [batch_size * seq_len, 5]
        """
        valid_mask = targets_flat != -100
        targets_clipped = targets_flat.clone()
        targets_clipped[~valid_mask] = 0  # dummy class for padding

        # Only validate values that are NOT padding
        if (targets_clipped[valid_mask] < 0).any() or (targets_clipped[valid_mask] > 4).any():
            raise ValueError(f"Found invalid target value(s): {targets_clipped[valid_mask].unique()}")

        one_hot = torch.nn.functional.one_hot(targets_clipped, num_classes=5).float()
        one_hot[~valid_mask] = 0
        one_hot[:, 4] *= exit_weight  # boost exit neuron
        return one_hot

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

    @profile_method(output_file=f"train_model_profile")
    def train_model(self, dataloader, num_epochs=20, learning_rate=0.0001, weight_decay=0.001,
                    device='cpu', tensorboard_writer=None, val_loader=None):
        logging.debug(f"Training {self.model_name} for {num_epochs} epochs on {device}...")

        start_time = time.time()
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training with reduced data set.")
            num_epochs = 2

        use_amp = torch.cuda.is_available()
        scaler = GradScaler()

        self.to(device)
        improvement_threshold = config.getfloat("DEFAULT", "improvement_threshold", fallback=0.01)
        logging.info(
            f"Early stopping patience set to {self.patience} epochs. Training will stop after {self.patience} "
            f"if no improvement over {improvement_threshold} ."
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

        # criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        criterion_ce = self.criterion_ce
        criterion_bce = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            running_loss = 0.0
            running_loss_dir = 0.0
            running_loss_exit = 0.0
            running_valid_targets = 0
            correct = 0
            total = 0

            self.train()
            desc = f"Epoch {epoch + 1} Training Progress"
            iterator = tqdm(dataloader, desc=desc, leave=True)

            epoch_start = time.time()
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(iterator):
                # Assume `batch` is a tuple: (inputs, target_actions)
                #   inputs   : shape (batch_size, seq_len, input_size)
                #   target_actions : shape (batch_size, seq_len)
                inputs, target_actions = batch
                inputs = inputs.to(device)
                target_actions = target_actions.to(device)

                # Forward pass produces outputs of shape [batch_size, seq_len, output_size]
                if use_amp:
                    # Enable automatic mixed precision context for faster GPU training and reduced memory usage
                    with torch.amp.autocast(device_type='cuda'):
                        outputs_dir, outputs_exit = self.forward(inputs)
                        # Compute losses and batch accuracy
                        loss, correct_batch, total_batch, loss_dir_value, loss_exit_value, collision_penalty, collision_rate = self._compute_dual_head_loss_and_accuracy(
                            outputs_dir, outputs_exit, target_actions,
                            criterion_ce, criterion_bce, epoch
                        )
                        correct += correct_batch
                        total += total_batch
                        running_loss += loss.item() * inputs.size(0)
                        # Count valid targets (non-padding)
                        targets_flat = target_actions.view(-1)  # ← flatten targets
                        valid_mask = (targets_flat != -100)
                        valid_targets_in_batch = valid_mask.sum().item()
                        running_valid_targets += valid_targets_in_batch
                else:
                    outputs_dir, outputs_exit = self.forward(inputs)
                    # Compute losses and batch accuracy
                    loss, correct_batch, total_batch, loss_dir_value, loss_exit_value, collision_penalty, collision_rate = self._compute_dual_head_loss_and_accuracy(
                        outputs_dir, outputs_exit, target_actions,
                        criterion_ce, criterion_bce, epoch
                    )
                    correct += correct_batch
                    total += total_batch
                    running_loss += loss.item() * inputs.size(0)
                    # Count valid targets (non-padding)
                    targets_flat = target_actions.view(-1)  # ← flatten targets
                    valid_mask = (targets_flat != -100)
                    valid_targets_in_batch = valid_mask.sum().item()
                    running_valid_targets += valid_targets_in_batch

                # Check for invalid loss values before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid loss encountered at epoch {epoch + 1}, batch {iteration + 1}")
                    raise ValueError(f"Invalid loss encountered {epoch + 1}, batch {iteration + 1}")

                # update progress bar with loss
                iterator.set_postfix(loss=loss.item())

                optimizer.zero_grad()

                if use_amp:
                    # Use AMP-safe backward pass: scales the loss to prevent underflow
                    scaler.scale(loss).backward()

                    # Unscale the gradients before clipping, only every 5 batches
                    if iteration % 5 == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    # Step with the unscaled gradients and update the scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass without AMP
                    loss.backward()

                    # Clip gradients every 5 batches to save compute
                    if iteration % 5 == 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    # Standard optimizer step
                    optimizer.step()

                # Per-batch logging every 10 iterations
                if iteration % 10 == 0:
                    grad_norm = sum((p.grad.norm().item() ** 2 for p in self.parameters() if p.grad is not None)) ** 0.5
                    logging.debug(
                        f"Epoch {epoch + 1} | Batch {iteration + 1} | Loss: {loss.item():.4f} "
                        f"| DirLoss: {loss_dir_value:.4f} | ExitLoss: {loss_exit_value:.4f} | GradNorm: {grad_norm:.4f}"
                    )
                    if tensorboard_writer:
                        step = epoch * len(dataloader) + iteration
                        tensorboard_writer.add_scalar("BatchLoss/train_total", loss.item(), step)
                        tensorboard_writer.add_scalar("BatchLoss/train_dir_only", loss_dir_value, step)
                        tensorboard_writer.add_scalar("BatchLoss/train_exit_only", loss_exit_value, step)
                        tensorboard_writer.add_scalar("GradientNorm/train", grad_norm, step)

                running_loss += loss.item() * inputs.size(0)
                running_loss_dir += loss_dir_value * inputs.size(0)
                running_loss_exit += loss_exit_value * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_loss_dir = running_loss_dir / len(dataloader.dataset)
            epoch_loss_exit = running_loss_exit / len(dataloader.dataset)
            training_accuracy = correct / total if total > 0 else 0.0

            # Validate model (using val_loader with similar reshaping steps)...
            validation_loss, validation_accuracy = self._validate_model(val_loader, device, epoch)

            train_losses["train"].append(epoch_loss)
            train_losses["validation"].append(validation_loss)
            train_accuracies["train"].append(training_accuracy)
            train_accuracies["validation"].append(validation_accuracy)

            epoch_duration = time.time() - epoch_start  # seconds
            epoch_time_minutes = round(epoch_duration, 2)
            epoch_valid_targets = running_valid_targets

            self._monitor_training(
                epoch=epoch,
                num_epochs=num_epochs,
                epoch_loss=epoch_loss,
                scheduler=scheduler,
                validation_loss=validation_loss,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                time_per_step=epoch_time_minutes,
                tensorboard_writer=tensorboard_writer,
                loss_dir_avg=epoch_loss_dir,
                loss_exit_avg=epoch_loss_exit,
                valid_targets=epoch_valid_targets,
                exit_weight=self.exit_weight,
                collision_penalty=collision_penalty,
                collision_rate=collision_rate,
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

    def _compute_dual_head_loss_and_accuracy(
            self,
            outputs_dir,
            outputs_exit,
            target_actions,
            criterion_ce,
            criterion_bce,
            epoch=None
    ):
        collision_penalty = 0.0

        dir_flat = outputs_dir.contiguous().view(-1, 4)
        exit_flat = outputs_exit.contiguous().view(-1)
        targets_flat = target_actions.contiguous().view(-1)

        direction_mask = (targets_flat != -100) & (targets_flat < 4)
        valid_mask = targets_flat != -100
        exit_target = (targets_flat == 4).float()

        # Individual losses
        loss_dir = criterion_ce(dir_flat[direction_mask], targets_flat[direction_mask])
        loss_exit = criterion_bce(exit_flat[valid_mask], exit_target[valid_mask])

        # Dynamically increase exit_weight over training
        exit_weight_base = config.getfloat("DEFAULT", "exit_weight", fallback=5.0)
        num_epochs = config.getint("DEFAULT", "num_epochs", fallback=10)
        exit_weight = exit_weight_base * min(1.0, epoch / (num_epochs * 0.4)) if epoch is not None else exit_weight_base

        # compute collision penalty
        batch_size, seq_len, _ = outputs_dir.shape
        predicted_dir = torch.argmax(outputs_dir, dim=2)  # shape: [B, T]
        base_penalty = config.getfloat("DEFAULT", "wall_penalty", fallback=1.0)
        num_epochs = config.getint("DEFAULT", "num_epochs", fallback=10)
        wall_penalty_weight = base_penalty * min(1.0, epoch / num_epochs) if epoch is not None else base_penalty
        # logging.info(f"Wall penalty weight: {wall_penalty_weight}")

        # Extract the wall context from the input (assumes first 4 features are [N, E, S, W])
        with torch.no_grad():
            wall_context = self.last_input[:, :, :4]  # shape: [B, T, 4]
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            collision_mask = 1 - wall_context[batch_indices, time_indices, predicted_dir]  # 1 if wall
            collision_penalty = collision_mask.float().mean() * wall_penalty_weight
        collision_rate = collision_mask.sum().item() / (batch_size * seq_len)

        total_loss = loss_dir + exit_weight * loss_exit + collision_penalty
        _, predicted = torch.max(dir_flat, dim=1)
        correct = (predicted[valid_mask] == targets_flat[valid_mask]).sum().item()
        total = valid_mask.sum().item()

        return total_loss, correct, total, loss_dir.item(), loss_exit.item(), collision_penalty, collision_rate

    def _monitor_training(
            self,
            epoch, num_epochs, epoch_loss, scheduler,
            validation_loss=0, training_accuracy=0, validation_accuracy=0,
            time_per_step=0., tensorboard_writer=None,
            loss_dir_avg=0., loss_exit_avg=0.,
            valid_targets=9, exit_weight=0, collision_penalty=0, collision_rate=0):
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
                        "model_name", "epoch", "train_loss", "train_loss_dir", "train_loss_exit",
                        "val_loss", "train_acc", "val_acc",
                        "timestamp", "time_per_step", "cpu_load", "gpu_load", "ram_usage",
                        "exit_weight", "valid_targets", "collision_penalty", "collision_rate"
                    ])
                writer.writerow([
                    self.model_name, epoch + 1,
                    epoch_loss, loss_dir_avg, loss_exit_avg,
                    validation_loss,
                    training_accuracy, validation_accuracy,
                    time.time(), time_per_step,
                    cpu_load, gpu_load, ram_usage,
                    exit_weight, valid_targets, round(collision_penalty.item(), 2), round(collision_rate, 2)
                ])

        # Load or initialize existing JSON list
        if os.path.exists(LOSS_JSON_FILE):
            with open(LOSS_JSON_FILE, "r") as jf:
                try:
                    all_logs = json.load(jf)
                except json.JSONDecodeError:
                    all_logs = []
        else:
            all_logs = []

        # Append new entry
        all_logs.append({
            "model_name": self.model_name,
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": validation_loss,
            "train_acc": training_accuracy,
            "val_acc": validation_accuracy,
            "timestamp": time.time(),
            "time_per_step": time_per_step,
            "cpu_load": cpu_load,
            "gpu_load": gpu_load,
            "ram_usage": ram_usage,
        })

        # Write updated list back
        with open(LOSS_JSON_FILE, "w") as jf:
            json.dump(all_logs, jf, indent=2)

        if tensorboard_writer and config.getboolean("DEFAULT", "tensorboard", fallback=False):
            tensorboard_writer.add_scalar("Loss/train", epoch_loss, epoch)
            tensorboard_writer.add_scalar("Loss/validation", validation_loss, epoch)
            tensorboard_writer.add_scalar("Accuracy/train", training_accuracy, epoch)
            tensorboard_writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)
            tensorboard_writer.add_scalar("Resource/CPU_load", cpu_load, epoch)
            tensorboard_writer.add_scalar("Resource/GPU_load", gpu_load, epoch)
            tensorboard_writer.add_scalar("Resource/RAM_usage", ram_usage, epoch)
            tensorboard_writer.add_scalar("Loss/wall_collision", collision_penalty.item(), epoch),
            tensorboard_writer.add_scalar("Loss/collision_rate", collision_penalty.item(), epoch),

            for name, param in self.named_parameters():
                tensorboard_writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    tensorboard_writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    def _validate_model(self, val_loader, device, epoch):
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
        self.eval()

        val_loss_sum = 0.0
        num_batches = 0
        correct = 0
        total = 0

        criterion_ce = self.criterion_ce
        criterion_bce = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in val_loader:
                inputs, target_actions = batch
                inputs = inputs.to(device)
                target_actions = target_actions.to(device)

                outputs_dir, outputs_exit = self.forward(inputs)
                dir_flat = outputs_dir.view(-1, 4)
                exit_flat = outputs_exit.view(-1)
                targets_flat = target_actions.view(-1)

                valid_mask = targets_flat != -100
                exit_target = (targets_flat == 4).float()

                loss_dir = criterion_ce(dir_flat, targets_flat)

                # Safety checks to prevent CUDA crash
                if exit_flat[valid_mask].shape != exit_target[valid_mask].shape:
                    raise RuntimeError(f"exit_flat and exit_target mismatch: "
                                       f"{exit_flat[valid_mask].shape} vs {exit_target[valid_mask].shape}")
                if torch.any(exit_target[valid_mask] < 0) or torch.any(exit_target[valid_mask] > 1):
                    raise RuntimeError("exit_target contains invalid values outside [0, 1]")

                loss_exit = criterion_bce(exit_flat[valid_mask], exit_target[valid_mask])
                exit_weight = config.getfloat("DEFAULT", "exit_weight", fallback=5.0)
                loss = loss_dir + exit_weight * loss_exit

                val_loss_sum += loss.item()
                num_batches += 1

                # Accuracy
                _, predicted = torch.max(dir_flat, dim=1)
                predicted = predicted[valid_mask]
                targets_masked = targets_flat[valid_mask]
                correct += (predicted == targets_masked).sum().item()
                total += targets_masked.size(0)

        average_val_loss = val_loss_sum / num_batches if num_batches > 0 else float("inf")
        validation_accuracy = correct / total if total > 0 else 0.0
        return average_val_loss, validation_accuracy


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
