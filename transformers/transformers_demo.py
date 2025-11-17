from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import spacy

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(sentence):
    # Tokenize the sentence and convert it to input IDs
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=51)
    with torch.no_grad():
        # Get the hidden states from BERT model (we use the last layer output)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings

knowledge = [
    'John loves playing cricket',
    'Sam enjoys working on weekends'
]

query = "Who loves cricket ?"

# Get embeddings for the query
query_embedding = get_embedding(query)

knowledge_embeddings = [get_embedding(knowledge) for knowledge in knowledge]

# Compute cosine similarity between query and each chunk
similarities = [cosine_similarity(query_embedding.numpy(), knowledge_embedding.numpy())[0][0] for knowledge_embedding in knowledge_embeddings]

# Find the most relevant chunk (max similarity)
most_relevant_chunk_idx = np.argmax(similarities)

most_relevant_chunk = knowledge[most_relevant_chunk_idx]

nlp = spacy.load("en_core_web_sm")
doc = nlp(most_relevant_chunk)

persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

# Print the most relevant chunk based on the query
if persons and persons[0]:
    print(f"The most relevant chunk based on the query is: '{persons[0]}'")
else:
    print(f"The most relevant chunk based on the query is: '{most_relevant_chunk}'")

