import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
positive_templates = [
    "I love the {item}, it is {adj} and works perfectly",
    "The {item} is {adj}; excellent experience and great support",
    "Very happy with this {item}, delivery was {adj}",
    "Customer service was {adj} and solved my issue quickly",
    "This is a {adj} product and I would recommend it",
    "The experience feels {adj}, simple and reliable",
    "Really pleased with the {item}; everything works well",
    "Fantastic {item}, the quality is {adj}",
    "Support team was {adj}, helpful and professional",
    "A {adj} experience from start to finish",
    "The update made the {item} much {adj}",
    "I am satisfied because the {item} is {adj}",
]
negative_templates = [
    "I hate the {item}, it is {adj} and barely works",
    "The {item} is {adj}; terrible experience and poor support",
    "Very unhappy with this {item}, delivery was {adj}",
    "Customer service was {adj} and did not solve my issue",
    "This is a {adj} product and I would not recommend it",
    "The experience feels {adj}, confusing and unreliable",
    "Really disappointed with the {item}; nothing works well",
    "Awful {item}, the quality is {adj}",
    "Support team was {adj}, rude and unhelpful",
    "A {adj} experience from start to finish",
    "The update made the {item} much {adj}",
    "I am dissatisfied because the {item} is {adj}",
]
items = ["app", "website", "service", "device", "checkout", "account", "feature", "order", "platform", "subscription", "dashboard", "delivery", "payment", "tool", "product"]
pos_adj = ["excellent", "smooth", "fast", "helpful", "reliable", "wonderful", "easy", "responsive", "impressive", "accurate", "pleasant", "efficient", "friendly", "stable", "great"]
neg_adj = ["awful", "slow", "broken", "unhelpful", "unreliable", "frustrating", "difficult", "unresponsive", "disappointing", "inaccurate", "poor", "inefficient", "rude", "unstable", "bad"]
texts, labels = [], []
for i in range(180):
    texts.append(positive_templates[i % len(positive_templates)].format(item=items[i % len(items)], adj=pos_adj[i % len(pos_adj)]))
    labels.append("positive")
    texts.append(negative_templates[i % len(negative_templates)].format(item=items[(i*3) % len(items)], adj=neg_adj[(i*7) % len(neg_adj)]))
    labels.append("negative")

df = pd.DataFrame({"text": texts, "label": labels}).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("customer_feedback_360.csv", index=False)
print("Dataset rows:", len(df))

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

df["clean_text"] = df["text"].map(clean_text)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])
tfidf_model.fit(X_train, y_train)
pred = tfidf_model.predict(X_test)
print(f"TF-IDF baseline: accuracy={accuracy_score(y_test, pred):.4f}, macro F1={f1_score(y_test, pred, average='macro'):.4f}")
print("Confusion matrix (labels: negative, positive):")
print(confusion_matrix(y_test, pred, labels=["negative", "positive"]))
print(classification_report(y_test, pred))

vectorizer = tfidf_model.named_steps["tfidf"]
classifier = tfidf_model.named_steps["clf"]
terms = np.array(vectorizer.get_feature_names_out())
for class_name, row in zip(classifier.classes_, classifier.coef_):
    top = np.argsort(row)[-10:][::-1]
    print(f"Top words for {class_name}:", terms[top].tolist())

try:
    import gensim.downloader as api
    vectors = api.load("glove-wiki-gigaword-50")
    for word in ["king", "computer"]:
        print(f"5 nearest to {word}:", vectors.most_similar(word, topn=5))
    try:
        analogy = vectors.most_similar(positive=["king", "woman"], negative=["man"], topn=1)[0][0]
        print("king - man + woman ->", analogy)
    except KeyError as exc:
        print("Analogy word unavailable:", exc)

    def sentence_vector(sentence):
        tokens = clean_text(sentence).split()
        known = [vectors[token] for token in tokens if token in vectors]
        return np.mean(known, axis=0) if known else np.zeros(vectors.vector_size)

    Xtr_vec = np.vstack([sentence_vector(s) for s in X_train])
    Xte_vec = np.vstack([sentence_vector(s) for s in X_test])
    emb_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    emb_clf.fit(Xtr_vec, y_train)
    emb_pred = emb_clf.predict(Xte_vec)
    print(f"Mean-GloVe baseline: accuracy={accuracy_score(y_test, emb_pred):.4f}, macro F1={f1_score(y_test, emb_pred, average='macro'):.4f}")
except Exception as exc:
    print("GloVe section skipped (download/network/dependency issue):", repr(exc))
