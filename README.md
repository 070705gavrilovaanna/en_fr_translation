# API для перевода с английского на французский: сравнение Seq2Seq+Attention и T5

Проект по реализации машинного перевода с английского на французский. Сравниваются две модели: самописная Seq2Seq с механизмом внимания (обучена с нуля на 185k парах) и предобученный T5 (fine-tuned на том же датасете). Лучшая модель (T5) развёрнута в виде API на Hugging Face Spaces.

**API на Hugging Face:** [ссылка на Space](https://huggingface.co/spaces/user43242422/translator-en-fr)

**Google Colab:** [ссылка](https://colab.research.google.com/drive/1S897DtYzaJb4SYFq2A25OYjNXRGvp9dm#scrollTo=fK66TWxQyfC9)

**Kaggle:** [ссылка на датасет](https://www.kaggle.com/datasets/alincijov/bilingual-sentence-pairs?select=fra.txt)

---

## Модели

| Модель | Архитектура | Размер словаря |
|--------|-------------|----------------|
| Seq2Seq + Attention | Encoder (BiLSTM) + Decoder (LSTM) | EN: 26k, FR: 44k |
| T5-small | Transformer (fine-tuned) | ~32k токенов |

---

## Сравнение переводов

| Английский | Seq2Seq | T5 |
|------------|---------|-----|
| I want to go to the store | je veux au travail. | Je veux aller au magasin. |
| I was very tired | j'étais été | J'étais très fatiguée. |
| I'd like to talk to you | il me faut | Je t'aimerais te parler. |
| I'm looking forward to seeing you | je ne peux pas un | J'espère te voir. |
| I don't know what to do | je dois y | Je ne sais pas quoi faire. |

## Выводы

- **Seq2Seq** уловил базовые закономерности, но переводы часто неверны или бессмысленны
- **T5** даёт грамматически правильные и осмысленные переводы
- Для качественного перевода с нуля нужно значительно больше данных и вычислительных ресурсов

## API на Hugging Face

Лучшая модель (T5) развёрнута в виде REST API.


- `notebooks/en_fr_translation.ipynb` - обучение и сравнение моделей
- `api/app.py` - код API (FastAPI)
- `api/Dockerfile` - контейнеризация для Hugging Face Spaces
- `api/requirements.txt` - зависимости
