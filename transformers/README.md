# transformers_demo.py

This script demonstrates how to use **transformers** and **vector embeddings** to answer simple queries, followed by **spaCy entity detection**.

### 🔍 What It Does
- Loads a pre-trained **BERT model** (`bert-base-uncased`) and tokenizer.
- Defines a helper function `get_embedding(sentence)` to convert text into vector embeddings.
- Stores a small knowledge base:
  - "John loves playing cricket"
  - "Sam enjoys working on weekends"
- Embeds a query: `"Who loves cricket ?"`
- Computes **cosine similarity** between the query and knowledge base embeddings.
- Selects the most relevant sentence.
- Uses **spaCy** (`en_core_web_sm`) to detect **PERSON** entities in the sentence.
- Prints the most relevant person or sentence.

### ▶️ Example Output

```text
The most relevant chunk based on the query is: 'John'
```
### ⚙️ Requirements
Install dependencies:

```bash
python -m spacy download en_core_web_sm
```

### ▶️ Running the Demo

```bash
python transformers_demo.py
```

### 📚 Concepts Covered
- **Vector embeddings:** Representing text as numerical vectors

- **Cosine similarity:** Measuring semantic closeness between vectors

- **Transformers (BERT):** Using pre-trained language models for embeddings

- **spaCy NER:** Detecting named entities like people in text