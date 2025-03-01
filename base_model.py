#base_model.py
import torch.nn as nn
import torch
from torch import optim
import logging
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import csv


OUTPUT = "output/"
TRAINING_PROGRESS_HTML = "training_progress.html"
TRAINING_PROGRESS_PNG = "training_progress.png"
LOSS_FILE = f"{OUTPUT}loss_data.csv"

class MazeBaseModel(nn.Module):
    def __init__(self):
        super(MazeBaseModel, self).__init__()

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_model(self, dataloader, num_epochs=20, learning_rate=0.001,training_samples=100, device='cpu', tensorboard_writer=None):
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
        # Move the model to the specified device ('cpu' or 'cuda').
        self.to(device)

        # Define the optimizer as Adam and set the learning rate.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Define a learning rate scheduler (Reduce LR when loss plateaus)
        #Step Decay (Reduce LR every 10 epochs)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        #Reduce LR if No Improvement
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Define the loss function as cross-entropy loss.
        criterion = nn.CrossEntropyLoss()

        # Set the model to training mode.
        self.train()

        #Display of training rate
        train_losses = []

        summary_writer = SummaryWriter(log_dir=f"{OUTPUT}model_tensorboard")

        # Loop over the specified number of epochs.
        for epoch in range(num_epochs):
            running_loss = 0.0  # Accumulate loss for the current epoch

            # Iterate through batches of inputs and targets from the dataloader.
            for iteration, (inputs, targets) in enumerate(dataloader):
                if iteration >= training_samples:
                    break

                # Add a sequence length dimension to inputs and move to the specified device.
                inputs = inputs.unsqueeze(1).to(device).float()

                # Move targets to the specified device.
                targets = targets.to(device)

                # Reset the gradients of model parameters.
                optimizer.zero_grad()

                # Perform a forward pass through the model to get the outputs.
                outputs = self.forward(inputs)

                # Compute the loss between the outputs and the targets.
                loss = criterion(outputs, targets)

                # Backpropagate the loss to compute gradients.
                loss.backward()

                # Update the model parameters using the optimizer.
                optimizer.step()

                # Accumulate the loss scaled by the batch size.
                running_loss += loss.item() * inputs.size(0)

            # Calculate the average loss for the current epoch.
            epoch_loss = running_loss / len(dataloader.dataset)

            # Print the epoch number and the corresponding loss.
            logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Update learning rate
            scheduler.step(epoch_loss)

            #Visualise training rate
            summary_writer.add_scalar('Loss/train', epoch_loss, epoch)
            summary_writer.add_scalar(
                'Accuracy/train',
                100.0 * torch.sum(torch.argmax(outputs, dim=1) == targets) / len(targets),
                epoch
            )
            #logging.info(
            #    f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            summary_writer.add_scalar('LearningRate/train', scheduler.get_last_lr()[0], epoch)

            # Append loss value to CSV file
            with open(LOSS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name, epoch + 1, epoch_loss])

            train_losses.append(epoch_loss)

            # Log to TensorBoard if writer is provided
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/epoch", epoch_loss, epoch)

        summary_writer.close()
        last_loss = train_losses[-1] if train_losses else None  # Save the last loss value

        return last_loss