import cv2
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def show_image(
    img: np.ndarray,
    title: str = "",
    *,
    cmap: str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Display an image using matplotlib

    Parameters
    ----------
    img : np.ndarray
        Image to display
    title : str, optional
        Figure title
    cmap : str | None, optional
        Colormap
    bbox : tuple[int, int, int, int] | None, optional
        bbox to display (x, y, width, height)
    """
    cmap = cmap or "gray" if img.ndim == 2 else None
    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    if bbox is not None:
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()


def show_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "",
) -> None:
    """
    Display a confusion matrix using seaborn

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Labels predicted
    class_names : list[str]
        name of each class
    title : str
        Title of the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(25, 10)
    fig.tight_layout()
    sns.heatmap(
        cm,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.show()
