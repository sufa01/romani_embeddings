# Эмбеддинги для цыганского языка (Romani Embeddings)

Пайплайн обработки корпуса цыганских текстов и обучения SVD и CBoW эмбеддингов.

| Папка/Файл | Описание |
|---|---|
| `pdf цыганский/` | Исходные PDF и TXT файлы |
| `raw_corpus/` | Необработанные тексты (извлечённые, но без очистки) — каждый в отдельном txt |
| `corpus_romani.txt` | Очищенный и лемматизированный корпус (одна строка = один текст, для SVD) |
| `cbow_sentences.txt` | Корпус для CBoW (одна строка = одно предложение) |
| `romani_embedding.py` | Полный пайплайн: извлечение, очистка, лемматизация, обучение, визуализация |
| `svd_model_romani.pkl` | Обученная SVD модель + эмбеддинги |
| `svd_embeddings_romani.csv` | SVD векторы в CSV |
| `cbow_model_romani.bin` | Обученная CBoW модель (Gensim) |
| `cbow_model_romani.txt` | CBoW векторы в текстовом формате |
| `svd_embeddings_viz.png` | Визуализация SVD эмбеддингов |
| `cbow_embeddings_viz.png` | Визуализация CBoW эмбеддингов |

## Запуск

```bash
pip install pdfplumber numpy pandas scikit-learn tqdm gensim matplotlib
python romani_embedding.py
