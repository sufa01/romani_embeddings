import os
import re
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_FOLDER = "pdf цыганский"
CORPUS_FILE = "corpus_romani.txt"
VOCAB_SIZE = 5000
WINDOW_SIZE = 5
SVD_COMPONENTS = 100

# Извлечение текста
def extract_pdf(path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"Ошибка PDF {path}: {e}")
    return text

def extract_txt(path):
    for enc in ["utf-8", "cp1251", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def extract_all(folder):
    if not os.path.exists(folder):
        print(f"Папка '{folder}' не найдена")
        sys.exit(1)
    full_text = []
    files = [f for f in os.listdir(folder) if f.endswith(('.pdf', '.txt'))]
    if not files:
        print(f"В папке '{folder}' нет файлов")
        sys.exit(1)
    for file in tqdm(files, desc="Чтение файлов"):
        print(f"\nОбрабатываю: {file}")
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            full_text.append(extract_pdf(path))
        elif file.endswith(".txt"):
            full_text.append(extract_txt(path))
    return "\n".join(full_text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s\-']", ' ', text)
    text = re.sub(r"\s'|'\s|^-|-$", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Словарь и датасет n-грамм
def build_vocab_and_ngram_dataset(text, vocab_size=VOCAB_SIZE, window=WINDOW_SIZE):
    words = text.split()
    if not words:
        print("текст пустой")
        sys.exit(1)
    word_counts = Counter(words)
    print(f"Уникальных слов: {len(word_counts)}")
    vocab_words = [w for w, _ in word_counts.most_common(vocab_size)]
    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    print(f"Размер словаря: {len(vocab_words)}")

    ngram_matrix = np.zeros((len(vocab_words), len(vocab_words)), dtype=np.float64)
    print(f"Подсчёт n-грамм (окно={window})")
    for i, target in enumerate(tqdm(words, desc="N-граммы")):
        if target not in word_to_id:
            continue
        tid = word_to_id[target]
        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            ctx = words[j]
            if ctx in word_to_id:
                ngram_matrix[tid, word_to_id[ctx]] += 1
    print("Матрица готова")
    return words, vocab_words, word_to_id, ngram_matrix

# SVD-эмбеддинги
def get_svd_embeddings(ngram_matrix, vocab_words, n_components=SVD_COMPONENTS):
    print(f"Получение {n_components}-мерных эмбеддингов")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(ngram_matrix)
    df = pd.DataFrame(embeddings, index=vocab_words)
    df.index.name = 'word'
    print(f"Объяснённая дисперсия: {svd.explained_variance_ratio_.sum():.4f}")
    return svd, df

def find_similar(word, df_embeddings, top_n=10):
    if word not in df_embeddings.index:
        return None
    target = df_embeddings.loc[word].values.reshape(1, -1)
    sims = cosine_similarity(target, df_embeddings.values).flatten()
    top_idx = sims.argsort()[::-1]
    result = []
    for idx in top_idx:
        w = df_embeddings.index[idx]
        if w != word:
            result.append((w, sims[idx]))
            if len(result) >= top_n:
                break
    return result

# Сохранение модели и загрузка
def save_model(svd, df_embeddings, word_to_id, path="svd_model_romani.pkl"):
    model_data = {
        'svd': svd,
        'embeddings': df_embeddings,
        'word_to_id': word_to_id
    }
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Модель сохранена в '{path}'")

def load_model(path="svd_model_romani.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Визуализация   
def visualize_embeddings(df_embeddings, words_to_plot, n_neighbors=5):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD as SVD2D

    # Берём только нужные слова и их ближайших соседей
    all_words = set(words_to_plot)
    for w in words_to_plot:
        sim = find_similar(w, df_embeddings, top_n=n_neighbors)
        if sim:
            all_words.update([s[0] for s in sim])

    subset = df_embeddings.loc[list(all_words & set(df_embeddings.index))]
    if subset.empty:
        print("Нет слов для визуализации")
        return

    # Проекция на 2D
    proj = SVD2D(n_components=2, random_state=42).fit_transform(subset.values)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(subset.index):
        x, y = proj[i]
        plt.scatter(x, y, c='red' if word in words_to_plot else 'blue', s=100)
        plt.annotate(word, (x, y), fontsize=9)
    plt.title("SVD-эмбеддинги (первые 2 компоненты)")
    plt.tight_layout()
    plt.show()

def main():
    raw = extract_all(DATA_FOLDER)
    clean = clean_text(raw)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(clean)
    print(f"Символов: {len(clean)}, слов: {len(clean.split())}")

    all_words, vocab, w2id, mat = build_vocab_and_ngram_dataset(clean)

    svd, df_emb = get_svd_embeddings(mat, vocab)

    save_model(svd, df_emb, w2id)
    df_emb.to_csv("svd_embeddings_romani.csv")

    for i, (w, f) in enumerate(Counter(all_words).most_common(20)):
        print(f"  {i+1}. {w} ({f})")

    top_words = [w for w, _ in Counter(all_words).most_common(100)]
    print(f"Визуализация для топ-100 слов: {top_words}")
    visualize_embeddings(df_emb, top_words)

if __name__ == "__main__":
    main()