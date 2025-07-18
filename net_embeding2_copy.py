import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix
import scipy.sparse as sp
from pymorphy3 import MorphAnalyzer
import pickle
from scipy.sparse import csr_matrix
import os

class TextProcessor:
    def __init__(self, templates, column_name='templates', vectorizer_path=None, stop_words_file='russian'):
        """
        Инициализация класса для обработки текстовых данных.
        
        :param templates: Список шаблонов для сравнения.
        :param stop_words_file: Путь к файлу со стоп-словами.
        :param vectorizer_path: Путь к файлу с сохраненным TfidfVectorizer (опционально) (.pkl).
        :param column_name: Название столбца в таблице переговоров (шаблонов)
        """
        
        self.morph = MorphAnalyzer()
        self.lemma_cache = {} # Кэш для ускорения лемматизации
        self.column_name = column_name
        self.vectorizer_path = vectorizer_path
        self.stop_words = self.load_stop_words(stop_words_file)
        self.cos_matrix = None # Матрица косинусного сходства
        self.ngrams = None # N-граммы фразы
        self.cos_similarities = [] # Список косинусных сходств
        self.templates = templates # Шаблоны
        self.original_ngrams = None  # Сохраняем исходную фразу
        #Шаблоны
        if isinstance(templates, list): # Если шаблоны переданы как список
            if isinstance(templates[0], str):
                self.original_templates = self.templates
            else:
                self.tfidf_templ = None
        elif self.is_end_file(templates, '.pkl'):
            self.load_data_with_pickle(templates, templates=True)
        #Если csv то загружаем его
        elif self.is_end_file(templates, '.csv'):
            try:
            # Чтение CSV-файла
                df = pd.read_csv(templates)
                # Проверка наличия столбца
                if self.column_name not in df.columns:
                    raise ValueError(f"Столбец '{self.column_name}' не найден в файле {templates}.")
                
                # Удаление пустых значений и преобразование в список
                templates = df[self.column_name].dropna().tolist()
                self.templates = templates
                self.original_templates = templates
                print(f"Загружено {len(templates)} шаблонов из файла {templates}.")
                self.tfidf_templ = None
        
            except Exception as e:
                raise RuntimeError(f"Ошибка при чтении CSV-файла: {e}")
        else:
            self.tfidf_templ = None
            self.original_templates = self.templates
        self.templates = [self.lemmatize_text(template) for template in self.original_templates]
            
    @staticmethod
    def is_end_file(file_path, end=None):
        """
        Проверяет, заканчивается ли файл на указанное расширение.
        
        :param file_path: Путь к файлу.
        :param end: Расширение для проверки (например, '.npz').
        :return: True, если файл заканчивается на указанное расширение, иначе False.
        """
        _, extension = os.path.splitext(file_path)
        return extension.lower() == end

    @staticmethod
    def min_max_normalize(matrix):
        """
        Нормализация матрицы

        :param matrix: Исходная матрица.
        :return: Нормализованная матрица.
        """
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val == min_val:  # Защита от деления на ноль
            return matrix / max_val if max_val != 0 else matrix
        return (matrix - min_val) / (max_val - min_val)
    
    def generate_substrings(self, sentence, ngrams_len=(2, 34+3)):
        """
        Генерация n-грамм из предложения с ограничением максимальной длины.
        
        :param sentence: Исходное предложение.
        :param max_ngram_length: Максимальная длина n-грамм (по умолчанию 5).
        :return: Список n-грамм.
        """
        words = sentence.split()
        ngrams = []
        
        # Генерация n-грамм длиной от 2 до max_ngram_length
        for n in range(ngrams_len[0], min(ngrams_len[1] + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.append(ngram)
        
        return ngrams

    def load_tfidf_vectorizer(self, file_path):
        """
        Загружает обученный TfidfVectorizer из файла.
        
        :param file_path: Путь к файлу с сохраненным объектом.
        :return: Загруженный объект TfidfVectorizer.
        """
        with open(file_path, 'rb') as file:
            vectorizer = pickle.load(file)
        
        # print(f"TfidfVectorizer успешно загружен из файла: {file_path}")
        return vectorizer

    def save_tfidf_vectorizer(self, file_path):
        """
        Сохраняет обученный TfidfVectorizer в файл.
        
        :param file_path: Путь, в какой файл сохранить.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.vectorizer, file)
            print('TfidfVectorizer успешно сохранён.')
    
    def load_stop_words(self, file_path):
        """
        Загрузка стоп-слов из файла.
        
        :param file_path: Путь к файлу со стоп-словами.
        :return: Список стоп-слов.
        """
        with open(file_path, 'rt', encoding='utf-8') as f:
            return f.read().split('\n')
    
        
    def lemmatize_text(self, text):
        """
        Лемматизация строки с помощью pymorphy3 
        
        :param text: Текстовое сообщение.
        :return: Лемматизированная строка.
        """
        words = re.findall(r'\w+|\W+', text)
        return ''.join(
            self.lemma_cache[word] if word in self.lemma_cache
            else self.lemma_cache.setdefault(word, self.morph.parse(word)[0].normal_form)
            if re.match(r'\w+', word) else word
            for word in words
        )
        

    def save_data_with_pickle(self, file_path, phrase=False, templates=False):
        """
        Сохранение данных в формате .pkl.
        Сохраняются: 
        original_phrase - исходная фраза (до лемматизации), 
        ngrams_before - n-граммы после лемматизации,
        tfidf_after - tfidf матрица после лемматизации (готовая к сравнению)
        Аналогично шаблон.
        Примечение: Один из параметров phrase или templates должен быть True.
        
        :param file_path: Путь, куда сохраняется файл.
        :param phrase: Если передан True, то сохраняем эмбеддинг фразы.
        :param templates: Если передан True, то сохраняем эмбеддинг шаблона.
        """
        if phrase:
            data = {
                "original_phrase": self.original_phrase,  # Исходная фраза
                "ngrams_before": self.original_ngrams,    # Список n-грамм до лемматизации
                "tfidf_after": self.tfidf_matrix          # TF-IDF матрица после лемматизации
            }
            print("Эмбеддинг фразы успешно сохранён.")
        elif (templates and phrase) or (not templates and not phrase):
            raise ValueError("Один из аргументов должен быть True (templates или phrase)")
        else:
            try:
                data = {
                    "original_templates": self.original_templates,  # Исходные шаблоны
                    "ngrams_before": self.original_ngrams,          # Список n-грамм до лемматизации
                    "tfidf_after": self.tfidf_templ                 # TF-IDF матрица после лемматизации
                }
                print("Эмбеддинг шаблонов успешно сохранён.")
            except AttributeError:
                raise AttributeError("Эмбеддинг не сформирован, запустите метод run().")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_data_with_pickle(self, file_path, phrase=False, templates=False):
        """
        Метод загрузки в формате .pkl, аналогичный сохранению. 
        
        :param file_path: Путь, куда сохраняется файл.
        :param phrase: Если передан True, то сохраняем эмбеддинг фразы.
        :param templates: Если передан True, то сохраняем эмбеддинг шаблона.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if phrase:
            self.original_phrase = data["original_phrase"]  # Загружаем исходную фразу
            self.ngrams = data["ngrams_before"]
            self.tfidf_matrix = data["tfidf_after"]
            print("TF-IDF матрица фразы успешно загружена.")
        elif (templates and phrase) or (not templates and not phrase):
            raise ValueError("Один из аргументов должен быть True (templates или phrase)")
        else:
            self.original_templates = data["original_templates"]  # Загружаем исходные шаблоны
            self.templates = data["ngrams_before"]
            self.tfidf_templ = data["tfidf_after"]
            print("TF-IDF матрица шаблонов успешно загружена.")

    def load_templates_from_csv(self, file_path, column_name="templates"):
        """
        Загрузка шаблонов из CSV-файла.
        
        :param file_path: Путь к CSV-файлу.
        :param column_name: Название столбца, содержащего шаблоны (по умолчанию "templates").
        :return: Список шаблонов.
        """
        try:
            df = pd.read_csv(file_path)
            if column_name not in df.columns:
                raise ValueError(f"Столбец '{column_name}' не найден в файле {file_path}.")
            templates = df[column_name].dropna().tolist()  # Убираем пустые значения
            print(f"Загружено {len(templates)} шаблонов из файла {file_path}.")
            return templates
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении CSV-файла: {e}")
    
    def validate_data(self):
        """
        Проверяет корректность данных и загружает необходимые компоненты.
        """
        #Фразы
        if self.is_end_file(self.phrase, '.pkl'):
            self.load_data_with_pickle(self.phrase, phrase=True)

        else:
            self.tfidf_matrix = None
        self.p = self.lemmatize_text(self.original_phrase)
        #Векторайзер
        if self.vectorizer_path is not None:
            if not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError(f"Файл '{self.vectorizer_path}' не найден.")
            if not self.is_end_file(self.vectorizer_path, '.pkl'):
                raise ValueError(f"Файл '{self.vectorizer_path}' должен иметь расширение .pkl")
            self.vectorizer = self.load_tfidf_vectorizer(self.vectorizer_path)
            self.tfidf_templ = None
        else:
            self.vectorizer = TfidfVectorizer(max_features=200, stop_words=self.stop_words)

    def preprocess_data(self, ngrams_len=(2, 34+3)):
        """
        Предобработка данных: генерация n-грамм, лемматизация и преобразование в TF-IDF матрицу.
        :param ngrams_len: Диапазон длин n-грамм.
        """
        self.original_ngrams = self.ngrams  # Сохраняем исходную фразу
        
        if self.tfidf_matrix is None:
            self.original_ngrams = self.generate_substrings(self.phrase, ngrams_len)
            self.phrase = self.lemmatize_text(self.phrase)
            self.ngrams = self.generate_substrings(self.phrase, ngrams_len)
        self.ngrams = [self.lemmatize_text(ngram) for ngram in self.ngrams]
        
        if self.vectorizer_path is None:
            self.vectorizer.fit(self.ngrams + self.templates)
            # Преобразуем n-граммы и шаблоны в TF-IDF матрицы
            self.tfidf_matrix = self.vectorizer.transform(self.ngrams)
            self.tfidf_templ = self.vectorizer.transform(self.templates)
        else:
            if self.tfidf_matrix is None:
                self.tfidf_matrix = self.vectorizer.transform(self.ngrams)
            if self.tfidf_templ is None:
                self.tfidf_templ = self.vectorizer.transform(self.templates)
        

    def compare_substrings_with_templates(self):
        """
        Сравнение шаблонов с n-граммами.
        
        :return: 
            - results: Список кортежей (номер шаблона, шаблон, максимальное сходство, номер n-граммы, соответствующая n-грамма).
            - best_template_info: Информация о шаблоне с максимальным сходством и соответствующей n-грамме.
        """
        # Вычисление косинусного сходства между шаблонами и n-граммами

        self.cos_matrix = cosine_similarity(self.tfidf_templ, self.tfidf_matrix)

        self.p = self.p.split(' ')
        s = [template.split(' ') for template in self.templates]

        lens_ = np.array([len(list(set(self.p).intersection(set(template)))) for template in s])
        self.cos_matrix = self.cos_matrix * lens_[:, np.newaxis]    
        # Векторизованное вычисление максимальных значений сходства и индексов n-грамм
        # max_similarities = np.max(self.cos_matrix, axis=1)
        # best_ngram_indices = np.argmax(self.cos_matrix, axis=1)
        
        # # Создание списка результатов
        # results = [
        #     (
        #         i + 1,  # Номер шаблона
        #         self.templates[i],  # Шаблон
        #         max_similarities[i],  # Максимальное сходство
        #         best_ngram_indices[i] + 1,  # Номер n-граммы
        #         self.ngrams[best_ngram_indices[i]]  # Соответствующая n-грамма
        #     )
        #     for i in range(len(self.templates))
        # ]
        
        # # Поиск шаблона с максимальным сходством
        # overall_max_similarity = np.max(max_similarities)  # Наибольшее значение сходства
        # best_template_index = np.argmax(max_similarities)  # Индекс шаблона
        # best_template = self.templates[best_template_index]  # Сам шаблон
        # best_ngram_index = best_ngram_indices[best_template_index]  # Индекс n-граммы
        # best_ngram = self.ngrams[best_ngram_index]  # Соответствующая n-грамма
        
        # # Формирование информации о лучшем шаблоне
        # best_template_info = {
        #     "template": best_template,
        #     "similarity": overall_max_similarity,
        #     "ngram": best_ngram
        # }
        
        # return results, best_template_info

    def analyze_similarity_with_metrics(self, thresholds):
        """
        Анализ схожести документов с использованием метрик accuracy и f1-score.
        
        :param thresholds: Список порогов для проверки косинусного сходства.
        """
        true_labels = []
        cos_similarities = []
    
        # Определение истинных меток (1 - максимальное сходство >= порог, 0 - иначе)
        for i in range(self.cos_matrix.shape[0]):
            max_similarity = np.max(self.cos_matrix[i])
            cos_similarities.append(max_similarity)
            true_labels.append(1 if max_similarity >= thresholds[0] else 0)

        # Оценка качества для каждого порога
        print("\nРезультаты:")
        for threshold in thresholds:
            predicted_labels = [1 if sim >= threshold else 0 for sim in cos_similarities]
            acc = accuracy_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            print(f"Порог {threshold:.2f}: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")

        # Построение KDE графика с использованием seaborn
        index0 = np.where(np.array(true_labels) == 0)
        index1 = np.where(np.array(true_labels) == 1)
        cos_similarities = np.array(cos_similarities)

        plt.figure(figsize=(10, 6))
        sns.kdeplot(cos_similarities[index0], label='Разные метки', shade=True, color='blue')
        sns.kdeplot(cos_similarities[index1], label='Одинаковые метки', shade=True, color='orange')
        plt.legend(loc='upper right')
        plt.title('Распределение косинусного сходства')
        plt.xlabel('Косинусное сходство')
        plt.ylabel('Плотность')
        plt.show()
        
        return true_labels, cos_similarities

    
    def run(self, phrase, ngrams_len=(2, 34+3)):
        """
        Основной метод для выполнения всех шагов обработки текста.
        :param phrase: Входная фраза.
        :param ngrams_len: Диапазон длин n-грамм.
        """
        # Предобработка данных
        
        self.phrase = phrase
        
        # self.templates = templates
        # self.stop_words = self.load_stop_words(stop_words_file)
        # self.vectorizer_path = vectorizer_path
        
        self.original_phrase = phrase 
        
        self.validate_data()
        # self.save_data_with_pickle('all_.pkl')
        # self.load_data_with_pickle(self.phrase)
        self.preprocess_data(ngrams_len)
        # self.save_data_with_pickle('all_.pkl')

        # Сравнение n-грамм с шаблонами
        self.compare_substrings_with_templates()
    
        # Поиск максимального элемента в матрице косинусного сходства
        max_value = np.max(self.cos_matrix)
        # print(self.cos_matrix)
        max_index = np.unravel_index(np.argmax(self.cos_matrix), self.cos_matrix.shape)
        
        #### Для тестов и для записи в таблицу####
        self.max_value = max_value
        self.max_similarity = self.original_templates[max_index[0]]
        self.orig_templ = self.original_templates[max_index[0]]
        self.ish_ngr = self.original_ngrams[max_index[1]]
        ##############
        print("Confidence:", max_value)
        print("Фраза:", self.original_phrase)  # Вывод исходной фразы
        print("Номер шаблона с максимальным сходством:", max_index[0] + 1)  # Номер шаблона
        print("Шаблон:", self.original_templates[max_index[0]])
        print("Исходная n-грамма:", self.original_ngrams[max_index[1]])
        print()

# templates_ = ['Машинист поезда <Число> на приближении к станции <Название>.',
#   'Слушаю Вас машинист поезда <Число> <ФАМИЛИЯ>.',
#   'По маршруту следования в выходной горловине на стрелке  <Число>   скорость не более <Число> км/ч ДСП <ФАМИЛИЯ>.',
#   'Понятно в выходной горловине на стрелке <Число> скорость не более <Число> км/ч Машинист <ФАМИЛИЯ>.',
#   'Дежурный по станции <ФАМИЛИЯ> слушаю.',
#   'Я машинист поезда  <Число> <ФАМИЛИЯ> на приближении к станции Следую с неисправными устройствами безопасности.',
#   'Понятно Пропускаетесь по I главному напроход (или без остановок) входной и выходной открыты ДСП <ФАМИЛИЯ>.',
#   'Пропускаетесь по <Число> боковому без остановки (или напроход) ДСП <ФАМИЛИЯ>.',
#   'Понятно по <Число> боковому без остановки (или напроход) машинист <ФАМИЛИЯ>']

processor = TextProcessor(templates='test.csv', column_name='test_name')
processor.run('Привет, машинист поезда номер 5 на приближении к станции Бабынино.')