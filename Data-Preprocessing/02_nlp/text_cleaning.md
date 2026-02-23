# Text Cleaning

Text cleaning is an **optional but often critical** preprocessing step. The key rule: clean based on your task and model — not blindly.

> Modern LLMs (GPT, BERT) are pretrained on raw, uncleaned text. Aggressive cleaning can hurt them. Traditional NLP (TF-IDF, Bag-of-Words) benefits more from cleaning.

---

## 1. Lowercasing

Convert all text to lowercase so `"Apple"` and `"apple"` map to the same token.

```python
text = text.lower()
```

**When to skip:** Named Entity Recognition (NER), where case carries meaning (`"US"` vs `"us"`).

---

## 2. Remove Punctuation

Strip characters that add no semantic value for the task.

```python
import re
text = re.sub(r"[^\w\s]", "", text)
```

**When to skip:** Sentiment analysis — `"great!"` vs `"great"` can carry different intensity. Also skip for models that use punctuation as token boundaries.

---

## 3. Remove Stopwords

Stopwords are high-frequency, low-information words: `the`, `is`, `at`, `which`, `on`.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
tokens = [w for w in text.split() if w not in stop_words]
```

**When to skip:** Transformers and LSTMs — they learn which words matter. Removing stopwords can break grammatical structure and hurt contextual models.

---

## 4. Remove HTML / URLs / Special Characters

```python
# Remove HTML tags
text = re.sub(r"<.*?>", "", text)

# Remove URLs
text = re.sub(r"http\S+|www\S+", "", text)

# Remove non-ASCII characters
text = text.encode("ascii", "ignore").decode()

# Remove extra whitespace
text = re.sub(r"\s+", " ", text).strip()
```

---

## 5. Spelling Correction

```python
from textblob import TextBlob
text = str(TextBlob(text).correct())
```

**Caution:** Expensive at scale. Can corrupt domain-specific terms (medical, legal, code).

---

## 6. Stemming vs Lemmatization

Both reduce words to a base form. Lemmatization is linguistically correct; stemming is faster but crude.

```python
# Stemming (fast, crude)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = stemmer.stem("running")   # → "run"

# Lemmatization (accurate, slower)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized = lemmatizer.lemmatize("running", pos="v")  # → "run"
```

**When to use:** Traditional ML pipelines (TF-IDF + Logistic Regression). Skip for deep learning models.

---

## 7. Expanding Contractions

```python
import contractions
text = contractions.fix("I can't do this")  # → "I cannot do this"
```

Useful for models that don't handle contractions well in their vocabulary.

---

## 8. Task-Based Decision Table

| Task | Lowercase | Remove Punctuation | Remove Stopwords | Stem / Lemmatize |
|---|---|---|---|---|
| Sentiment Analysis | ✅ | ⚠️ Careful | ❌ | ❌ |
| Text Classification | ✅ | ✅ | ✅ (traditional) | ✅ (traditional) |
| NER | ❌ | ⚠️ | ❌ | ❌ |
| Machine Translation | ❌ | ❌ | ❌ | ❌ |
| BERT / GPT fine-tuning | ❌ | ❌ | ❌ | ❌ |
| TF-IDF / Bag-of-Words | ✅ | ✅ | ✅ | ✅ |