import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Для WordNetLemmatizer
nltk.download('punkt_tab')  # Для Punkt tokenizer

# Загрузка данных
data = pd.read_csv('reviews.csv')

# Преобразование метки настроений в бинарные значения
data['label'] = data['sentiment'].apply(lambda label: 1 if label == 'positive' else 0)


# Функция для предварительной обработки текста
def preprocess_text(text):
    # Проверка на NaN или None
    if pd.isna(text):
        return ""

    # Удаление неалфавитные символы
    text = re.sub(r'[^\w\s]', '', str(text))
    # Замена n пробелов на один
    text = re.sub(r'\s+', ' ', text)
    # Удаление цифр
    text = re.sub(r'\d', '', text)
    # Токен текст
    text = word_tokenize(text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    text = [t for t in text if t.lower() not in stop_words]
    # Лемм текст
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t) for t in text]
    return ' '.join(text)


# Применение предварительной обработки к каждому отзыву
tqdm.pandas()
data['processed'] = data['review'].progress_apply(preprocess_text)

# Сохранение обработанных данных в новый CSV файл
processed_data = data[['processed', 'label']]
processed_data.to_csv('reviews_preprocessed.csv', index=False, header=['review', 'label'])

processed_data.head()