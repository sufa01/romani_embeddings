import os
import sys
import re
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import pdfplumber
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD as SVD2D

SOURCE_FOLDER = "pdf цыганский"
CORPUS_FILE = "corpus_romani.txt"
CBOW_SENTENCES_FILE = "cbow_sentences.txt"
SVD_MODEL_FILE = "svd_model_romani.pkl"
CBOW_MODEL_FILE = "cbow_model_romani.bin"
EMBEDDINGS_CSV = "svd_embeddings_romani.csv"
RAW_FOLDER = "raw_corpus"

VOCAB_SIZE = 5000
WINDOW_SIZE = 5
SVD_COMPONENTS = 100
VECTOR_SIZE = 100
CBOW_MIN_COUNT = 5
CBOW_EPOCHS = 10

# Извлечение текста
# def extract_pdf(path):
#     text = ""
#     with pdfplumber.open(path) as pdf:
#         for page in pdf.pages:
#             t = page.extract_text()
#             if t:
#                 text += t + "\n"
#     return text
def extract_pdf(path):
    """
    Извлечение текста из PDF с восстановлением структуры предложений.
    """
    # Признаки служебных строк (типографские пометки)
    SERVICE_PATTERNS = [
        r'учгиз', r'учпедгиз', r'типография', r'печат', r'набор',
        r'тираж', r'заказ', r'бумага', r'бум\s*л', r'печ\s*л',
        r'редактор', r'корректор', r'подписана', r'сдана',
        r'глазлита', r'уполн', r'формат', r'знак\s*на',
        r'^\d+$', r'^\d+[-–]\d+$', 
        r'^[IVXLC]+$',  
        r'^-\w{1,2}$',
        r'кочетков.*дружына',            
        r'^[а-яё]\s[а-яё]\s[а-яё]\s[а-яё]',
        r'москва\s+им$',             
        r'цыг$',            
    ]
    
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                keep_blank_chars=True,
                use_text_flow=False,
                extra_attrs=['size']
            )
            
            if not words:
                continue
            
            lines = []
            current_line = []
            current_y = None
            
            for word in words:
                y = round(word['top'], 1)
                
                if current_y is None:
                    current_y = y
                
                if abs(y - current_y) > 3:
                    if current_line:
                        lines.append(current_line)
                    current_line = [word['text']]
                    current_y = y
                else:
                    current_line.append(word['text'])
            
            if current_line:
                lines.append(current_line)
            page_text = []
            skip_next = False
            
            for i, line in enumerate(lines):
                line_text = ' '.join(line)
                is_service = False
                for pattern in SERVICE_PATTERNS:
                    if re.search(pattern, line_text, re.IGNORECASE):
                        is_service = True
                        break
                if len(line_text.strip()) <= 2:
                    is_service = True
                
                if is_service:
                    skip_next = True
                    continue
                if line_text.endswith('-') and i + 1 < len(lines):
                    continue
                if i > 0 and ' '.join(lines[i - 1]).endswith('-'):
                    prev = page_text.pop() if page_text else ''
                    line_text = prev.rstrip('-') + line_text
                if line_text and line_text[-1] in '.!?…':
                    page_text.append(line_text + '\n')
                else:
                    page_text.append(line_text)
            
            text += ' '.join(page_text) + '\n'

    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  
    text = re.sub(r'\s-\w{1,2}\s', ' ', text)         
    text = re.sub(r'\s+\w\s+', ' ', text)              
    text = re.sub(r'\s+', ' ', text)                   
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)            
    return text.strip()

def extract_txt(path):
    for enc in ["utf-8", "cp1251", "latin-1", "utf-16"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_all_texts(folder):
    files = [f for f in os.listdir(folder) 
             if f.lower().endswith(('.pdf', '.txt'))]
    texts = {}
    for file in tqdm(files, desc="Чтение"):
        path = os.path.join(folder, file)
        name = os.path.splitext(file)[0]
        
        if file.lower().endswith(".pdf"):
            texts[name] = extract_pdf(path)
        else:
            texts[name] = extract_txt(path)
    
    return texts

def save_raw_corpus(texts_dict, output_folder=RAW_FOLDER):
    os.makedirs(output_folder, exist_ok=True)
    for name, text in tqdm(texts_dict.items(), desc="Сохранение raw корпуса"):
        output_path = os.path.join(output_folder, f"{name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"Raw корпус сохранён")

# Очистка и лемматизация
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s\-']", ' ', text)
    text = re.sub(r"\s'|'\s|^-|-$", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """
    лемматизатор rule-based
    """
    rules = [
        (r'эстэ$', ''), (r'эски$', ''), (r'энца$', ''),
        (r'ен$', ''), (r'эн$', ''),
        (r'эс$', ''), (r'эл$', ''), (r'ав$', ''),
        (r'ас$', ''), (r'ам$', ''), (r'ан$', ''),
        (r'э$', ''), (r'и$', ''), (r'ы$', ''), (r'о$', ''),
    ]
    # Стоп-слова
    stop_words = {
        'а', 'и', 'но', 'да', 'на', 'со', 'ко', 'во', 'то',
        'мэ', 'ту', 'ём', 'ёв', 'ой', 'амэ', 'тумэ', 'ёнэ',
        'та', 'тэ', 'чи', 'ин', 'кэ', 'джи', 'кай', 'пал',
    }
    words = text.split()
    lemmatized = []
    for word in words:
        if len(word) <= 2 or word in stop_words:
            lemmatized.append(word)
        else:
            lemma = word
            for pattern, replacement in rules:
                new_word = re.sub(pattern, replacement, word)
                if new_word != word and len(new_word) >= 3:
                    lemma = new_word
                    break
            lemmatized.append(lemma)
    
    return ' '.join(lemmatized)

def process_texts(raw_texts):
    """Очистка и лемматизация всех текстов"""
    processed = {}
    for name, text in tqdm(raw_texts.items(), desc="Обработка"):
        cleaned = clean_text(text)
        lemmatized = lemmatize_text(cleaned)
        processed[name] = lemmatized
    return processed

def split_sentences(text):
    """Разбивка текста на предложения"""
    sentences = re.split(r'[.!?…]+', text)
    return [s.strip() for s in sentences if len(s.strip().split()) > 1]

# SVD
def build_ngram_matrix(text):
    """Построение словаря и матрицы n-грамм"""
    words = text.split()
    word_counts = Counter(words)
    print(f"  Уникальных слов: {len(word_counts)}")
    vocab = [w for w, _ in word_counts.most_common(VOCAB_SIZE)]
    w2id = {w: i for i, w in enumerate(vocab)}
    print(f"  Словарь (top-{VOCAB_SIZE}): {len(vocab)} слов")
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
    for i, target in enumerate(tqdm(words, desc="  N-граммы")):
        if target not in w2id:
            continue
        tid = w2id[target]
        start = max(0, i - WINDOW_SIZE)
        end = min(len(words), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j and words[j] in w2id:
                matrix[tid, w2id[words[j]]] += 1
    return words, vocab, w2id, matrix

def train_svd(matrix, vocab):
    """Обучение SVD"""
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    embeddings = svd.fit_transform(matrix)
    df = pd.DataFrame(embeddings, index=vocab)
    df.index.name = 'word'
    print(f"Объяснённая дисперсия: {svd.explained_variance_ratio_.sum():.4f}")
    return svd, df


def find_similar(word, df, top_n=10):
    """Поиск похожих слов в SVD"""
    if word not in df.index:
        return None
    target = df.loc[word].values.reshape(1, -1)
    sims = cosine_similarity(target, df.values).flatten()
    result = []
    for idx in sims.argsort()[::-1]:
        w = df.index[idx]
        if w != word:
            result.append((w, sims[idx]))
            if len(result) >= top_n:
                break
    return result

# CBoW
def train_cbow(sentences):
    tokenized = [s.split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=CBOW_MIN_COUNT,
        sg=0,
        workers=4,
        epochs=CBOW_EPOCHS,
        seed=42
    )
    print(f"Слов в словаре CBoW: {len(model.wv)}")
    return model

# Визуализайция
def visualize(df=None, cbow_model=None, words_to_plot=None):
    """Визуализация эмбеддингов (SVD или CBoW)"""
    method = "SVD" if df is not None else "CBoW"
    if words_to_plot is None:
        if df is not None:
            words_to_plot = list(df.index[:20])
        elif cbow_model is not None:
            words_to_plot = cbow_model.wv.index_to_key[:20]
    all_words = set(words_to_plot)
    for w in words_to_plot[:10]:
        if df is not None:
            sim = find_similar(w, df, top_n=5)
        else:
            try:
                sim = cbow_model.wv.most_similar(w, topn=5)
            except KeyError:
                continue
        if sim:
            all_words.update([s[0] for s in sim])
    
    if df is not None:
        valid = list(all_words & set(df.index))
        vectors = df.loc[valid].values
        labels = valid
    else:
        vectors, labels = [], []
        for w in all_words:
            if w in cbow_model.wv:
                vectors.append(cbow_model.wv[w])
                labels.append(w)
        vectors = np.array(vectors)
    if len(vectors) == 0:
        return
    
    # 2D проекция
    proj = SVD2D(n_components=2, random_state=42).fit_transform(vectors)
    plt.figure(figsize=(12, 10))
    for i, word in enumerate(labels):
        x, y = proj[i]
        is_target = word in words_to_plot
        plt.scatter(x, y, c='red' if is_target else 'blue', 
                   s=100 if is_target else 50, alpha=1.0 if is_target else 0.5)
        plt.annotate(word, (x, y), fontsize=9 if is_target else 7,
                    alpha=1.0 if is_target else 0.5)
    
    plt.title(f"{method} Embeddings (top-20 words + neighbors)")
    plt.tight_layout()
    plt.savefig(f"{method.lower()}_embeddings_viz.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    raw_texts = extract_all_texts(SOURCE_FOLDER)
    save_raw_corpus(raw_texts, RAW_FOLDER)
    all_raw_sentences = []
    for name, text in raw_texts.items():
        sentences = split_sentences(text)
        all_raw_sentences.extend(sentences)
    processed = process_texts(raw_texts)
    
    # Корпус для SVD
    all_processed = "\n".join(processed.values())
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(all_processed)
    
    # Корпус для CBoW
    cbow_sentences = []
    for sent in all_raw_sentences:
        cleaned = clean_text(sent)
        if len(cleaned.split()) > 1:
            lemmatized = lemmatize_text(cleaned)
            cbow_sentences.append(lemmatized)
    
    with open(CBOW_SENTENCES_FILE, "w", encoding="utf-8") as f:
        for sent in cbow_sentences:
            f.write(sent + "\n")
    
    # Статистика
    total_raw = sum(len(t.split()) for t in raw_texts.values())
    total_proc = sum(len(t.split()) for t in processed.values())
    print(f"Слов до обработки: {total_raw}, после: {total_proc}")
    words, vocab, w2id, matrix = build_ngram_matrix(all_processed)
    svd, df_emb = train_svd(matrix, vocab)
    
    # Сохранение SVD
    with open(SVD_MODEL_FILE, 'wb') as f:
        pickle.dump({'svd': svd, 'embeddings': df_emb, 'word_to_id': w2id}, f)
    df_emb.to_csv(EMBEDDINGS_CSV)
    # Топ-слова
    print("Топ-15 слов:")
    for i, (w, c) in enumerate(Counter(words).most_common(15), 1):
        print(f"  {i:2d}. {w:15s} ({c})")
    
    # Примеры похожих слов SVD
    print("Семантические соседи (SVD):")
    for w in list(vocab[:5]):
        sim = find_similar(w, df_emb, top_n=5)
        if sim:
            print(f"  {w}: {', '.join(f'{s[0]}({s[1]:.3f})' for s in sim)}")
    cbow_model = train_cbow(cbow_sentences)
    cbow_model.save(CBOW_MODEL_FILE)
    cbow_model.wv.save_word2vec_format(
        CBOW_MODEL_FILE.replace('.bin', '.txt'), binary=False
    )
    # Примеры похожих слов CBoW
    print("Семантические соседи (CBoW):")
    for w in cbow_model.wv.index_to_key[:5]:
        sim = cbow_model.wv.most_similar(w, topn=5)
        print(f"  {w}: {', '.join(f'{s[0]}({s[1]:.3f})' for s in sim)}")
    top_words = [w for w, _ in Counter(words).most_common(20)]
    visualize(df_emb, words_to_plot=top_words)
    visualize(cbow_model=cbow_model, 
             words_to_plot=cbow_model.wv.index_to_key[:20])

if __name__ == "__main__":
    main()
