
# sentiment-analysis

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Architecture](#model-architecture)
7. [Training and Evaluation](#training-and-evaluation)
8. [Results Visualization](#results-visualization)
9. [Prediction Function](#prediction-function)
10. [Future Improvements](#future-improvements)
11. [Contributing](#contributing)
12. [License](#license)

## Project Overview

This project aims to predict mental health status based on textual statements using Natural Language Processing (NLP) techniques and a Convolutional Neural Network (CNN) model. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a prediction function for new statements.

## Installation

To run this project, you need to have Python installed on your system. Clone the repository and install the required packages:

```bash
git clone [https://github.com/kknani24/Sentiment-Analysis-for-Mental-Health-Using-NLP-and-Deep-Learning.git]
cd sentiment-analysis.ipynb
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
pandas
plotly
nltk
scikit-learn
textblob
numpy
wordcloud
matplotlib
tensorflow
```

## Usage

To run the main script:

```bash
python sentiment-analysis.ipynb
```

## Data Preprocessing

The data preprocessing steps include:

1. Loading the data from 'Combined Data.csv'
2. Handling missing values
3. Text cleaning:
   - Lowercasing
   - Removing text in square brackets
   - Removing links and HTML tags
   - Removing punctuation and newlines
   - Removing words containing numbers
4. Tokenization and stopword removal
5. Data augmentation using translation

## Exploratory Data Analysis (EDA)

The EDA phase includes:

1. Displaying basic dataset information
2. Visualizing the distribution of mental health status
3. Analyzing text length distribution
4. Creating a word cloud of cleaned statements
5. Visualizing the proportion of each status category

## Model Architecture

The CNN model architecture:

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])
```

## Training and Evaluation

The model is trained using:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 10
- Validation split: 0.2
- Batch size: 32

Evaluation metrics include:
- Test Accuracy
- Classification Report
- Confusion Matrix

## Results Visualization

Results are visualized using Plotly:
- Histogram of mental health status distribution
- Text length distribution
- Confusion matrix heatmap
- Word cloud of cleaned statements
- Pie chart of status category proportions

## Prediction Function

A `predict_status` function is provided to make predictions on new statements:

```python
predicted_status = predict_status(statement_to_predict, tokenizer, model, label_map, reverse_label_map)
```

## Future Improvements

Potential areas for improvement:
1. Fine-tuning hyperparameters
2. Experimenting with different model architectures (e.g., LSTM, Transformer)
3. Incorporating more features (e.g., sentiment analysis scores)
4. Collecting more diverse data to improve model generalization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
