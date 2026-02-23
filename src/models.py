import copy
import sys
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

torch.manual_seed(42)


def is_notebook() -> bool:
    """
    Detect if the code is running inside a Jupyter notebook.
    """
    return "ipykernel" in sys.modules


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseImageModel(nn.Module, ABC):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.network = self._build_network()

    @abstractmethod
    def _build_network(self) -> nn.Module:
        """
        Return the neural network architecture

        Returns
        -------
        nn.Module
            The neural network architecture
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        Returns
        -------
        torch.Tensor
            Logits
        """
        return self.network(x)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float = 1e-3,
        early_stopping_patience: int | None = None,
    ) -> None:
        """
        Train the model using minibatch gradient descent

        Parameters
        ----------
        train_loader : DataLoader
            Training dataset loader
        val_loader : DataLoader
            Validation dataset loader
        epochs : int
            Number of training epochs
        lr : float, optional
            Learning rate, by default 1e-3
        early_stopping_patience : int | None, optional
            Number of epoch possible without a better model
        """
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_val_acc = 0
        best_weights = copy.deepcopy(self.state_dict())
        patience_counter = 0

        with tqdm(
            range(epochs),
            desc="Training Epochs",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="green",
            leave=False,
        ) as epoch_bar:
            for _ in epoch_bar:
                train_loss = self._train_one_epoch(train_loader, optimizer)
                val_loss, val_acc = self._evaluate(val_loader, verbose=False)

                epoch_bar.set_postfix_str(
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc * 100:.2f}%",
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    break

        self.load_state_dict(best_weights)
        self.best_val_acc = best_val_acc

    def _train_one_epoch(self, dataloader: DataLoader, optimizer: torch.optim) -> float:
        """
        Train the model for a single epoch

        Parameters
        ----------
        dataloader : DataLoader
            Loader providing training batches
        optimizer : torch.optim
            Optimizer performing gradient updates

        Returns
        -------
        float
            Mean training loss for the epoch
        """
        self.train()
        total_loss = 0

        with tqdm(
            dataloader,
            desc="Training Batches",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="yellow",
            leave=False,
        ) as batch_bar:
            for x, y in batch_bar:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = self(x)
                loss = self.criterion(logits, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    def _evaluate(self, dataloader: DataLoader, verbose: bool) -> tuple[float, float]:
        """
        Evaluate the model on a dataset without updating weights

        Parameters
        ----------
        dataloader : DataLoader
            Loader providing evaluation batches
        verbose : bool
            Whether to display a tqdm progress bar

        Returns
        -------
        tuple[float, float]
            Mean loss and accuracy
        """
        self.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        iterator = (
            tqdm(
                dataloader,
                desc="Evaluating",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
                colour="magenta",
                leave=False,
            )
            if verbose
            else dataloader
        )

        with torch.no_grad():
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)

                logits = self(x)
                loss = self.criterion(logits, y)

                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss += loss.item()
                if verbose:
                    iterator.set_postfix_str(
                        f"loss={loss.item():.4f}  batch_acc={(preds == y).float().mean().item() * 100:.1f}%",
                    )

        acc = total_correct / total_samples
        return total_loss / len(dataloader), acc

    def predict(self, dataloader: DataLoader) -> list[int]:
        """
        Run inference over a DataLoader and return predicted class indices

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader containing the images

        Returns
        -------
        list[int]
            A flat list of predicted class indices
        """
        self.eval()
        all_preds: list[int] = []

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                logits = self(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())

        return all_preds


class CNNModel(BaseImageModel):
    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),  # 224x224 assumed
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )


class CNNTransferModel(BaseImageModel):
    def _build_network(self) -> nn.Module:
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1,
        )

        for param in backbone.parameters():
            param.requires_grad = False

        backbone.fc = nn.Linear(
            backbone.fc.in_features,
            self.num_classes,
        )

        return backbone


class ViTImageModel(BaseImageModel):
    def _build_network(self) -> nn.Module:
        backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1,
        )

        for param in backbone.parameters():
            param.requires_grad = False

        backbone.heads.head = nn.Linear(
            backbone.heads.head.in_features,
            self.num_classes,
        )

        return backbone
