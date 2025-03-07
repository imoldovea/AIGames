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
    def __init__(self):
        super(MazeBaseModel, self).__init__()
        self.model_name = "MazeBaseModel"  # Define the model name

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_model(self, dataloader, num_epochs=20, learning_rate=0.001, training_samples=100, weight_decay=0.001, device='cpu', tensorboard_writer=None):
        """
        Generic training loop using CrossEntropyLoss and Adam optimizer.

        Parameters:
            dataloader (DataLoader): Dataloader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to train on ('cpu' or 'cuda').

        Returns:
            self: The trained model.
        """
        self.to(device)
        patience = 5
        trigger_times = 0

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience)
        criterion = nn.CrossEntropyLoss()

        self.train()
        train_losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0

            for iteration, (local_context, target_action, steps_number) in enumerate(dataloader):
                local_context = torch.tensor(local_context).to(device).float()
                steps_number = steps_number.to(device).unsqueeze(1).float()  # Shape: [batch_size, 1]
                inputs = torch.cat((local_context, steps_number), dim=1)
                inputs = inputs.unsqueeze(1)

                assert inputs.shape[-1] == 5, f"Expected input features to be 5, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, f"Expected target labels to be 1-dimensional, but got {target_action.dim()} dimensions"

                outputs = self.forward(inputs)
                loss = criterion(outputs, target_action.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if iteration >= training_samples:
                    break

            epoch_loss = running_loss / len(dataloader.dataset)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            scheduler.step(epoch_loss)
            with open(LOSS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name, epoch + 1, epoch_loss])

            train_losses.append(epoch_loss)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/epoch", epoch_loss, epoch)

            if epoch_loss < min(train_losses):
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info(f"Early Stopping Triggered at Epoch {epoch + 1}")
                    break

        last_loss = train_losses[-1] if train_losses else None
        logging.info(f"Training Complete for {self._get_name()}. Final Loss: {last_loss}")
        return last_loss
