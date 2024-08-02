
# Sentiment Analysis for Mental Health Using NLP and Deep Learning

## Project Overview
This project aims to analyze mental health statements using Natural Language Processing (NLP) techniques and Deep Learning models. It focuses on classifying mental health status based on textual data, providing insights into the distribution of mental health conditions, and visualizing the results.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Code Structure](#code-structure)
5. [Results](#results)
6. [Visualizations](#visualizations)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/kknani24/Sentiment-Analysis-for-Mental-Health-Using-NLP-and-Deep-Learning.git
cd sentiment-analysis
```


### Set Up the Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


### Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Download NLTK Data

Run the following Python code to download necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Dataset

The project uses a dataset named 'Combined Data.csv'. This dataset contains statements related to mental health and their corresponding status labels. The main columns used are:
- `statement`: The textual content describing mental health experiences.
- `status`: The mental health status label associated with each statement.

Example of loading the dataset:

```python
import pandas as pd

path = 'Combined Data.csv'
df = pd.read_csv(path)

print(df.head())
print(df.info())
```

## Methodology

1. **Data Preprocessing**:
   - Cleaning text data (removing punctuation, links, etc.)
   - Tokenization
   - Removing stop words

   ```python
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(r'\[.*?\]', '', text)
       text = re.sub(r'https?://\S+|www\.\S+', '', text)
       # ... (more preprocessing steps)
       return text

   df['cleaned_statement'] = df['statement'].apply(lambda x: preprocess_text(x))
   ```

2. **Exploratory Data Analysis (EDA)**:
   - Analyzing the distribution of mental health status labels
   - Examining text length distribution

   ```python
   import plotly.express as px

   fig = px.histogram(df, x='status', title='Distribution of Mental Health Status')
   fig.show()
   ```

3. **Data Augmentation**:
   - Using translation techniques to augment the dataset

   ```python
   def augment_text(text):
       try:
           blob = TextBlob(text)
           translated = blob.translate(to='fr').translate(to='en')
           return str(translated)
       except Exception as e:
           return text

   df['augmented_statement'] = df['statement'].apply(augment_text)
   ```

4. **Model Development**:
   - Tokenization and padding of sequences
   - Building a Convolutional Neural Network (CNN) model using TensorFlow/Keras
   - Training the model on the preprocessed data

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout

   model = Sequential([
       Embedding(input_dim=10000, output_dim=128),
       Conv1D(filters=128, kernel_size=5, activation='relu'),
       GlobalMaxPooling1D(),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(len(label_map), activation='softmax')
   ])

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   ```

5. **Evaluation**:
   - Assessing model performance using accuracy, classification report, and confusion matrix

   ```python
   from sklearn.metrics import classification_report, confusion_matrix

   y_pred = model.predict(X_test)
   y_pred_classes = np.argmax(y_pred, axis=1)

   print(classification_report(y_test, y_pred_classes))
   ```

## Code Structure

The main script contains the following sections:

1. Importing libraries
2. Loading and preprocessing data
3. Exploratory Data Analysis
4. Text preprocessing and augmentation
5. Model building and training
6. Evaluation and visualization

## Results

The CNN model achieves [insert accuracy here] accuracy on the test set. Detailed performance metrics, including precision, recall, and F1-score for each class, are provided in the classification report.

```python
# Example of printing the classification report
print(classification_report(y_test, y_pred_classes))
```

## Visualizations

The project includes several visualizations to aid in understanding the data and results:

1. Distribution of Mental Health Status
2. Text Length Distribution
3. Confusion Matrix
4. Word Cloud of Cleaned Statements
5. Proportion of Each Status Category (Pie Chart)

Example of creating a word cloud:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_text = ' '.join(df['cleaned_statement'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Cleaned Statements')
plt.show()
```


## License

[MIT]
