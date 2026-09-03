# 🤖 Machine Learning ChatBot V1.0

A lightweight machine-learning chatbot designed to answer questions related to **Machine Learning fundamentals**.

This project is an experimental **V1.0** built to explore how a simple intent-based chatbot can use a neural network to classify user input and select an appropriate response.

The chatbot learns from a collection of **intents**, where each intent contains:

* A `tag` identifying the intent
* Multiple `patterns` representing possible user questions
* Multiple `responses` that the chatbot can return

The current version contains approximately **200 intents and 601 training patterns**.

> 🚧 **V1.0:** This is an early version of the project. The dataset is intentionally relatively small, and future versions may significantly expand the number of intents and patterns to make the chatbot more capable.

---

## 🧠 How It Works

The chatbot follows a relatively simple machine-learning pipeline:

```text
User Input
    │
    ▼
Tokenization
    │
    ▼
Stemming
    │
    ▼
Bag-of-Words Representation
    │
    ▼
Neural Network
    │
    ├── Hidden Layer — 512 units
    │
    ├── Dropout — 50%
    │
    ├── Hidden Layer — 512 units
    │
    ├── Dropout — 50%
    │
    └── Output Layer — Softmax
    │
    ▼
Predicted Intent
    │
    ▼
Matching Intent
    │
    ▼
Random Response
```

When a user enters a question, the input is tokenized and stemmed before being converted into a **bag-of-words vector**.

The neural network then predicts which intent the input most likely belongs to.

Once the intent has been identified, the program searches `intents.json` for the corresponding tag and randomly selects one of its predefined responses.

---

## 🧩 Intent-Based Dataset

The chatbot does not generate completely new answers.

Instead, it uses an intent-classification approach.

Each intent follows a structure similar to:

```json
{
    "tag": "machine_learning",
    "patterns": [
        "What is machine learning?",
        "Explain machine learning",
        "What does machine learning mean?"
    ],
    "responses": [
        "Machine learning is a field of AI that allows systems to learn from data."
    ]
}
```

The `patterns` are used as training examples, while the `responses` are used after the neural network predicts the corresponding intent.

This makes it easy to expand the chatbot simply by adding new intents and training patterns to `intents.json`.

---

## 📊 Current Dataset

| Metric            |                          V1.0 |
| ----------------- | ----------------------------: |
| Intents           |                          ~200 |
| Training patterns |                           601 |
| Output classes    |                  Intent-based |
| Domain            | Machine Learning fundamentals |
| Training epochs   |                         1,000 |
| Batch size        |                             8 |

The dataset is currently focused primarily on **Machine Learning fundamentals**, with the possibility of expanding into broader **Deep Learning** concepts in future versions.

---

## 🧠 Neural Network Architecture

The neural network is implemented using **TFLearn on top of TensorFlow**.

The current architecture contains:

```text
Input
  ↓
Fully Connected Layer — 512 units
  ↓
Dropout — 50%
  ↓
Fully Connected Layer — 512 units
  ↓
Dropout — 50%
  ↓
Output Layer — Softmax
```

### Hidden Layers

Two fully connected hidden layers are used, with:

```text
512 neurons
512 neurons
```

### Dropout

A dropout rate of **0.5** is applied after each hidden layer:

```python
tflearn.dropout(net, 0.5)
```

This randomly drops a portion of activations during training and is intended to help reduce **overfitting**.

### Output Layer

The final layer uses the **Softmax activation function**:

```python
tflearn.fully_connected(
    net,
    len(output[0]),
    activation="softmax"
)
```

Softmax produces a probability distribution across the available intent classes, allowing the chatbot to select the most likely intent.

---

## ⚙️ Data Preprocessing

Before training, the patterns from `intents.json` go through several preprocessing steps.

### 1. Tokenization

NLTK is used to split sentences into individual words:

```python
nltk.word_tokenize(pattern)
```

For example:

```text
"What is machine learning?"
```

becomes a sequence of tokens.

### 2. Stemming

The project uses NLTK's `LancasterStemmer` to reduce words to their stems.

This allows different forms of a word to be treated more similarly during classification.

### 3. Vocabulary Creation

The chatbot builds a vocabulary from the training patterns and removes duplicate words.

### 4. Bag of Words

Each training sentence is converted into a numerical vector.

For every word in the vocabulary:

```text
1 → word exists in the sentence
0 → word does not exist
```

This transforms natural-language input into a numerical representation that can be processed by the neural network.

### 5. One-Hot Intent Labels

Each intent is represented as an output vector where the corresponding intent position is set to `1`.

This allows the network to learn the relationship between input patterns and intent classes.

---

## 💾 Data & Model Persistence

The project uses **Pickle** to store processed training data:

```text
data.pickle
```

It contains:

* Processed vocabulary
* Intent labels
* Training vectors
* Output vectors

The trained neural-network model is saved separately through TFLearn:

```text
chat.model
```

This allows the program to load an existing trained model instead of retraining it every time the chatbot starts.

---

## 🛠️ Technologies Used

| Technology     | Purpose                          |
| -------------- | -------------------------------- |
| **Python**     | Main programming language        |
| **TensorFlow** | Neural-network framework/backend |
| **TFLearn**    | High-level neural-network API    |
| **NLTK**       | Tokenization and stemming        |
| **NumPy**      | Numerical data manipulation      |
| **JSON**       | Intent dataset management        |
| **Pickle**     | Saving processed training data   |
| **Random**     | Selecting responses              |

---

## 📁 Project Structure

```text
Machine-Learning-ChatBot-V1.0/
│
├── main.py
├── intents.json
├── README.md
│
├── data.pickle          # Generated processed dataset
├── chat.model*          # Generated trained model files
└── log/                 # TensorBoard/TFLearn logs
```

> `data.pickle`, `chat.model*`, and `log/` are generated files and do not need to be manually edited.

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/hamaraff/Machine-Learning-ChatBot-V1.0.git
```

### 2. Install the required libraries

```bash
pip install tensorflow tflearn nltk numpy
```

### 3. Run the chatbot

```bash
python main.py
```

The chatbot will start in the terminal:

```text
Start talking with the bot! (type exit to stop..)

You:
```

Type:

```text
exit
```

to terminate the chatbot.

---

## ✏️ Adding New Knowledge

One of the main goals of this project is making the chatbot easy to expand.

You can add new intents directly to:

```text
intents.json
```

For example, you can create a new intent containing:

```text
tag
patterns
responses
```

Adding more diverse patterns helps the neural network learn different ways users may ask the same question.

Future versions can therefore expand the dataset with:

* More Machine Learning concepts
* Deep Learning concepts
* Neural Network concepts
* Computer Vision
* Natural Language Processing
* Model evaluation
* Optimization algorithms
* More variations of existing questions

---

## 📈 Future Improvements

This project is currently **V1.0**, so there is plenty of room for improvement.

Potential improvements include:

* [ ] Expand the intent dataset
* [ ] Add more training patterns
* [ ] Add more Deep Learning concepts
* [ ] Increase dataset diversity
* [ ] Add a proper validation/test split
* [ ] Track training and validation loss
* [ ] Track training and validation accuracy
* [ ] Evaluate precision, recall and F1-score
* [ ] Experiment with different network architectures
* [ ] Experiment with different dropout rates
* [ ] Optimize training hyperparameters
* [ ] Improve confidence handling
* [ ] Add a graphical or web interface
* [ ] Improve natural-language understanding
* [ ] Experiment with more modern NLP architectures

---

## ⚠️ Current Limitations

This chatbot is intentionally simple.

It is an **intent-classification chatbot**, not a large language model.

Therefore, it does not understand conversations in the same way systems such as modern generative AI models do.

Its responses are limited to the intents and predefined responses contained in `intents.json`.

The relatively small V1.0 dataset also means that the chatbot's knowledge is limited to the concepts and question variations represented in the training data.

Expanding and diversifying the dataset should improve its ability to correctly classify new questions.

---

## 🔬 Project Goal

The main goal of this project is to experiment with the fundamentals of building a machine-learning chatbot from scratch:

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Representation
   ↓
Neural Network
   ↓
Training
   ↓
Intent Classification
   ↓
Response Selection
```

Rather than relying on a pre-trained conversational AI model, this project demonstrates the fundamentals of building an **intent-based neural-network chatbot** using Python, TensorFlow and TFLearn.

---

## 🤝 Contributing & Experimenting

This repository is **public** and available for anyone interested in experimenting with the project.

Feel free to:

* Clone the repository
* Explore the code
* Modify `intents.json`
* Add new intents
* Add new training patterns
* Experiment with the neural-network architecture
* Improve the preprocessing pipeline
* Experiment with different training configurations
* Build your own version of the chatbot

The project is intended as a learning and experimentation platform, so **fork it, modify it, break it, improve it, and build your own version.** 🚀

---

## 📌 Version

**Machine Learning ChatBot — V1.0**

Built as an experimental project focused on learning and applying fundamental machine-learning and neural-network concepts.
