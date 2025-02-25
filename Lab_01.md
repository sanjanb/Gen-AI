# Gensim Word2Vec Conversion and Exploration

This script demonstrates how to convert a GloVe model to Word2Vec format and perform various word relationship and arithmetic operations using Gensim.

## Installation

Use the following command to install the necessary package:

```bash
!pip install gensim
```

## Gensim Word2Vec Conversion

The script uses `gensim.scripts.glove2word2vec` to convert the GloVe file to Word2Vec format and then loads the Word2Vec model using `gensim.models.KeyedVectors`.

### Code

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Paths to the GloVe file and output Word2Vec file
glove_input_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.100d.txt" # Path to GloVe file
word2vec_output_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.50d.word2vec.txt" # Output

# Convert GloVe format to Word2Vec format
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the converted Word2Vec model
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Test the loaded model
print(model.most_similar("king"))
```

## Explore Word Relationships

### **Example 1**: Find Similar Words

Find words similar to "mysore":

```python
similar_to_mysore = model.similar_by_vector(model['mysore'], topn=5)
print(f"Words similar to 'mysore': {similar_to_mysore}")
```

### **Example 2**: Gender Analogy (king - man + woman = queen)

```python
# Perform vector arithmetic
result_vector_1 = model['actor'] - model['man'] + model['woman']
# Find the most similar word
result_1 = model.similar_by_vector(result_vector_1, topn=1)
print(f"'actor - man + woman' = {result_1}")
```

### **Example 3**: Country-City Relationship (India - Delhi + Bangalore)

```python
# Perform vector arithmetic
result_vector_2 = model['india'] - model['delhi'] + model['washington']
# Find the most similar word
result_2 = model.similar_by_vector(result_vector_2, topn=3)
print(f"'India - Delhi + Washington' = {result_2}")
```

## Perform Arithmetic Operations

### **Scaling a Vector**

```python
scaled_vector = model['hotel'] * 2 
result_2 = model.similar_by_vector(scaled_vector, topn=3)
result_2
```

### **Normalizing Vectors**

```python
import numpy as np
normalized_vector = model['fish'] / np.linalg.norm(model['fish'])
result_2 = model.similar_by_vector(normalized_vector, topn=3)
result_2
```

### **Averaging Vectors**

```python
average_vector = (model['king'] + model['woman'] + model['man']) / 3
result_2 = model.similar_by_vector(average_vector, topn=3)
result_2
```

## Calculate Similarity Between Two Words

### Code Example

```python
# Paths to the GloVe file and output Word2Vec file
glove_input_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.100d.txt" # Path to GloVe file
word2vec_output_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.50d.word2vec.txt" # Output

# Convert GloVe format to Word2Vec format
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the converted Word2Vec model
model_50d = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Paths to the GloVe file and output Word2Vec file
glove_input_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.100d.txt" # Path to GloVe file
word2vec_output_file = "/content/drive/MyDrive/Colab Notebooks/Gen-AI lab/glove.6B.50d.word2vec.txt" # Output

# Convert GloVe format to Word2Vec format
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the converted Word2Vec model
model_100d = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Calculate similarity between two words
word1 = "hospital"
word2 = "doctor"

similarity_50d = model_50d.similarity(word1, word2)
similarity_100d = model_100d.similarity(word1, word2)

print(f"Similarity (50d) between '{word1}' and '{word2}': {similarity_50d:.4f}")
print(f"Similarity (100d) between '{word1}' and '{word2}': {similarity_100d:.4f}")
```

## Calculate Distance Between Two Words

```python
# Calculate distance between two words
distance_50d = model_50d.distance(word1, word2)
distance_100d = model_100d.distance(word1, word2)

print(f"Distance (50d) between '{word1}' and '{word2}': {distance_50d:.4f}")
print(f"Distance (100d) between '{word1}' and '{word2}': {distance_100d:.4f}")
```

---

## Summary of Key Functions

| Function/Method               | Description                                      |
|-------------------------------|--------------------------------------------------|
| `glove2word2vec(input, output)` | Converts GloVe format to Word2Vec format        |
| `KeyedVectors.load_word2vec_format()` | Loads a Word2Vec model from the specified file |
| `model.most_similar()`         | Finds most similar words to the given word      |
| `model.similar_by_vector()`    | Finds similar words by vector representation    |
| `model.similarity()`           | Calculates cosine similarity between two words  |
| `model.distance()`             | Calculates cosine distance between two words    |

---

## Conclusion

This notebook demonstrates how to convert GloVe embeddings to Word2Vec format, explore word relationships, perform vector arithmetic, and calculate word similarity and distance. By using Gensim, you can leverage powerful word embeddings to enhance your NLP tasks.
