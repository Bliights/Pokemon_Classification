import re
from pathlib import Path

import pandas as pd


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Build a pandas DataFrame using the fanart-dataset folder

    Parameters
    ----------
    data_dir : Path
        Directory containing the dataset images

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
            - `path`: absolute or relative path to the image file (as a string)
            - `pokemon`: extracted Pokémon label (as a string)
    """
    rows = []

    for img_path in data_dir.glob("*.jpg"):
        name = img_path.name.lower()

        # pattern: 000000__pokemon.jpg
        m = re.match(r"\d{6}__([a-z0-9_\-]+)\.jpg$", name)
        if not m:
            continue

        pokemon = m.group(1)

        rows.append(
            {
                "path": str(img_path),
                "pokemon": pokemon,
            },
        )

    df = pd.DataFrame(rows)

    return df.sort_values(["pokemon", "path"]).reset_index(drop=True)
