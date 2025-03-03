#base_model.py
import torch.nn as nn
import torch
from torch import optim
import logging
import torch.optim.lr_scheduler as lr_scheduler
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

    def train_model(self, dataloader, num_epochs=20, learning_rate=0.001,training_samples=100, weight_decay=0.001, device='cpu', tensorboard_writer=None):
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
        patience = 5
        trigger_times = 0

        # Define the optimizer as Adam and set the learning rate.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define a learning rate scheduler (Reduce LR when loss plateaus)
        #Step Decay (Reduce LR every 10 epochs)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        #Reduce LR if No Improvement
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience)

        # Define the loss function as cross-entropy loss.
        criterion = nn.CrossEntropyLoss()

        # Set the model to training mode.
        self.train()
        #Display of training rate
        train_losses = []

        # Loop over the specified number of epochs.
        for epoch in range(num_epochs):
            running_loss = 0.0  # Accumulate loss for the current epoch

            # Iterate through batches of inputs, targets, and step numbers from the dataloader.
            for iteration, (local_context, target_action, steps_number) in enumerate(dataloader):
                """
                Iterates through batches from the dataloader and processes them for training.
                
                - local_context: Contextual input data for the model.
                - target_action: Ground truth labels for classification.
                - steps_number: Additional input feature representing step numbers.
                
                Steps:
                1. Move data to the specified device and ensure correct types.
                2. Concatenate local_context and steps_number to form input features.
                3. Compute predictions, loss, and gradients.
                4. Update model parameters using backpropagation.
                """

                # Move data to the appropriate device and ensure correct data types
                local_context = torch.tensor(local_context).to(device).float()
                steps_number = steps_number.to(device).unsqueeze(1).float()  # Shape: [batch_size, 1]

                 # Concatenate local_context and steps_number to form inputs
                inputs = torch.cat((local_context, steps_number), dim=1)
                inputs = inputs.unsqueeze(1)

                assert inputs.shape[-1] == 5, f"Expected input features to be 5, but got {inputs.shape[-1]}"
                assert target_action.dim() == 1, f"Expected target labels to be 1-dimensional, but got {target_action.dim()} dimensions"

                # Perform a forward pass through the model to get the outputs.
                outputs = self.forward(inputs)

                # Compute the loss between the outputs and the targets.
                loss = criterion(outputs, target_action.to(device))

                # Backpropagate the loss to compute gradients.
                optimizer.zero_grad()  # Reset the gradient calculations for the optimizer to prepare for the backward pass.
                loss.backward()
                # Update the model parameters using the optimizer.
                optimizer.step()

                # Accumulate the loss scaled by the batch size.
                running_loss += loss.item() * inputs.size(0)

                # Break the loop if the training sample limit is reached
                if iteration >= training_samples:
                    break

            # Calculate the average loss for the current epoch.
            epoch_loss = running_loss / len(dataloader.dataset)

            # Print the epoch number and the corresponding loss.
            logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Update learning rate
            scheduler.step(epoch_loss)

            # Append loss value to CSV file
            with open(LOSS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name, epoch + 1, epoch_loss])

            train_losses.append(epoch_loss)

            # Log to TensorBoard if writer is provided
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/epoch", epoch_loss, epoch)

            # Early stopping condition
            if epoch_loss < min(train_losses):
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logging.info(f"Early Stopping Triggered at Epoch {epoch + 1}")
                    break

        last_loss = train_losses[-1] if train_losses else None  # Save the last loss value
        logging.info(f"Training Complete for {self._get_name()}. Final Loss: {last_loss}")
        return last_loss