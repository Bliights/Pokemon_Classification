import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class BoVW:
    """
    Bag of Visual Words (BoVW) encoder
    """

    def __init__(
        self,
        n_words: int = 512,
        batch_size: int = 4096,
        l2_normalize: bool = True,
        power_norm: bool = True,
    ) -> None:
        """
        Initialize the BoVW encoder

        Parameters
        ----------
        n_words : int, optional
            Size of the visual vocabulary, by default 512
        batch_size : int, optional
            Mini-batch size, by default 4096
        l2_normalize : bool, optional
            If True, apply L2 normalization to the final histograms
        power_norm : bool, optional
            If True, apply power normalization (sqrt) to histograms
        """
        self.n_words = n_words
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self.power_norm = power_norm
        self.kmeans: MiniBatchKMeans | None = None

    def _stack_descriptors(self, desc_list: list[np.ndarray]) -> np.ndarray:
        """
        Stack descriptors from multiple images into a single matrix

        Parameters
        ----------
        desc_list : list[np.ndarray]
            List of descriptor matrices, one per image

        Returns
        -------
        np.ndarray
            Stacked descriptor matrix
        """
        stacked = []
        for desc in desc_list:
            if desc is None or desc.size == 0:
                continue
            stacked.append(desc.astype(np.float32))

        if len(stacked) == 0:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(stacked)

    def fit(self, desc_list_train: list[np.ndarray]) -> "BoVW":
        """
        Learn the visual vocabulary

        Parameters
        ----------
        desc_list_train : list[np.ndarray]
            List of descriptor matrices (one per training image)

        Returns
        -------
        BoVW
            The fitted BoVW instance

        Raises
        ------
        ValueError
            If no descriptors are provided
        """
        dataset = self._stack_descriptors(desc_list_train)
        if dataset.size == 0:
            raise ValueError("No descriptors provided to fit the BoVW vocabulary.")

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_words,
            batch_size=self.batch_size,
            random_state=42,
            verbose=0,
        )
        self.kmeans.fit(dataset)
        return self

    def transform(self, desc_list: list[np.ndarray]) -> np.ndarray:
        """
        Encode each image as a BoVW histogram using the learned vocabulary

        Parameters
        ----------
        desc_list : list[np.ndarray]
            List of descriptor matrices (one per training image)

        Returns
        -------
        np.ndarray
            Feature matrix (Each row is the BoVW histogram)

        Raises
        ------
        RuntimeError
            If the vocabulary has not been learned yet
        """
        if self.kmeans is None:
            raise RuntimeError("BoVW vocabulary not fitted. Call fit() first.")

        dataset = np.zeros((len(desc_list), self.n_words), dtype=np.float32)

        for i, desc in enumerate(desc_list):
            if desc is None or desc.size == 0:
                continue

            desc = desc.astype(np.float32)
            words = self.kmeans.predict(desc)
            hist = np.bincount(words, minlength=self.n_words).astype(np.float32)
            dataset[i] = hist

        if self.power_norm:
            dataset = np.sqrt(dataset)

        if self.l2_normalize:
            dataset = normalize(dataset, norm="l2")

        return dataset
