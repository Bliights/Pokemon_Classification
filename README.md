# Pokemon Classification

## Table of Contents
1. [Overview](#overview)
2. [Development](#development)
3. [Project Structure](#project-structure)
4. [Contributors](#contributors)

## Overview
This project aims to classify Pokémon fan-art images. The task is particularly challenging due to large variations in art style, pose, background, and overall image quality, as well as the presence of noisy or imperfect data. Our objective is to design and evaluate the most effective solution possible under these constraints.

To conduct our experiments, we use an open-source dataset hosted on Hugging Face: [PokeFA - Pokémon Fanart Captioned](https://huggingface.co/datasets/Kev0208/PokeFA-pokemon-fanart-captioned). This dataset provides links to fan-art images along with associated metadata.

We will then compare two different approaches:
1. A classical computer vision pipeline based purely on image processing and handcrafted features
2. A deep learning–based approach, leveraging neural network models for image classification

## Development
This project follows the best practices we currently rely on for building maintainable Python projects. We use:

- **`uv`** for dependency management  
- **`pre-commit`** for automated code quality checks  
- **`Ruff`** for linting and formatting (a VS Code configuration is included)  
- **`Makefile`** to run common commands consistently

### Requirements
- **Python 3.12**
- **`uv`** (recommended) for dependency management and to automatically create the package link for a smoother development workflow
- **`make`** (recommended) to use the provided Makefile commands

### Environment setup
Install dependencies and set up `pre-commit` hooks:
```bash
make install
```

### Code quality
Run linting/formatting checks via `pre-commit`:
```bash
make pre-commit
```

### Dowload the dataset
To download the dataset, we created a script that fetches the dataset metadata from Hugging Face and then downloads safely the corresponding images from their original websites. To run the download step, use:
```bash
make download-dataset
```

In addition, if you want to run the full pipeline (including experiments with data augmentation), you first need to generate the augmented images. We provide a dedicated script for this step as well:

```bash
make data-augmentation
```

## Project Structure
You'll find all the utility functions and classes (Python `.py` files) you'll need for our experiments and tests in the [`src`](./src/) folder. You'll also find the main notebook, [`pokemon_classification.ipynb`](./src/pokemon_classification.ipynb), where we run the experiments, compare the different approaches, and present quickely the results and analysis.

## Contributors


|            Name            |                Email                  |
| :------------------------: | :-----------------------------------: |
|    MOLLY-MITTON Clément    |    clement.mollymitton@gmail.com      |
|       VERBECQ DIANE        |        diane.verbecq@gmail.com        |
