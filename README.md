# Music Genre Classification using Model-Based Machine Learning

## Description

This study examines the effectiveness of logistic regression and feed-forward neural networks in capturing the complexity of a music genre dataset, which includes 11 distinct genre classes.
The data was processed and analyzed, and missing values were addressed. Ten different experiments were conducted, during which five models were trained and evaluated using the dataset.
Inference methods such as Markov Chain Monte Carlo (MCMC) and Variational Inference (VI) were applied, and the resulting outcomes were compared.

A detailed report of the project: [Report](https://github.com/MiladMt11/Model-Based-ML_Spring2023/blob/bb940b3fe411d0de64bdcd3c5721096e7241e051/project%20report.pdf)

* In this project, we leveraged the [Pyro package](https://pypi.org/project/pyro-ppl/), an advanced probabilistic programming framework built on PyTorch, to model complex, highly flexible probabilistic systems.

## Dataset
The dataset used for this project is available [here](https://www.kaggle.com/code/jvedarutvija/music-genre-classification).

## Project Structure and File Descriptions
Descriptive statistical analysis of the dataset + data preprocessing has been done and explained in the following notebooks:
* [descriptive_statistics.ipynb](https://github.com/MiladMt11/Model-Based-ML_Spring2023/blob/bb940b3fe411d0de64bdcd3c5721096e7241e051/descriptive_statistics.ipynb)
* [data_preprocessing.ipynb](https://github.com/MiladMt11/Model-Based-ML_Spring2023/blob/bb940b3fe411d0de64bdcd3c5721096e7241e051/data_preprocessing.ipynb)

Ancestral sampling approach, is the simplest approach for modeling the problem in which the parameters of the model are sampled using a simple Numpy sampling function:
* [ancestral_sampling.ipynb](https://github.com/MiladMt11/Model-Based-ML_Spring2023/blob/bb940b3fe411d0de64bdcd3c5721096e7241e051/ancestral_sampling.ipynb)

The notebook containing models with different choices of priors and inference methods:
* [models.py](https://github.com/MiladMt11/Model-Based-ML_Spring2023/blob/bb940b3fe411d0de64bdcd3c5721096e7241e051/models.py)

The rest of the notebooks are experiments with different combinations of choices of priors, models and inference methods.

The preprocessed [pickle file](https://github.com/MiladMt11/Model-Based-ML_Spring2023/tree/bb940b3fe411d0de64bdcd3c5721096e7241e051/pickle) of the data is also provided for convenient data loading.

## Frameworks & Packages
* [Pytorch](https://pytorch.org/)
* [Pyro](https://pypi.org/project/pyro-ppl/)
* scikit-learn
* Numpy

## Installation

Follow these steps to set up the project locally.

### Prerequisites

You will need to have Python installed on your machine. You can download it [here](https://www.python.org/downloads/).

### Install

1. Clone the repository:

    ```bash
    git clone https://github.com/MiladMt11/Model-Based-ML_Spring2023.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Model-Based-ML_Spring2023
    ```

3. Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Disclaimer
This project is the result of collaborative group work, with contributions from multiple team members, and may not fully represent the views or capabilities of any single individual.

## Contact
We are open to any feedback, suggestions, or questions regarding the projects or the repository. Don't hesitate to contact via email at milad.mtkh@gmail.com
