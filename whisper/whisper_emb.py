import whisper
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
from whisper import DecodingResult
from whisper.decoding import DecodingOptions
from whisper.tokenizer import get_tokenizer

class WhisperAnalyst:
    def __init__(self, model_size='small', device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size).to(self.device)
        self.tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language='ru',
            task='transcribe',
        )
    
    def write_data_to_csv(self, filename, data):
        """Записывает данные в CSV файл."""
        with open(filename, mode='a+', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            for array, label in data:
                array_str = json.dumps(array.tolist())
                writer.writerow([array_str, str(label)])
    
    def read_data_from_csv(self, filename):
        """Читает данные из CSV файла."""
        embeddings = []
        labels = []
        
        with open(filename, mode='r', encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Пропускаем заголовок
            
            for row in reader:
                try:
                    embedding_str = row[0]
                    label = int(row[1])
                    embedding_json = json.loads(embedding_str)
                    embedding_array = np.array(embedding_json)
                    t = torch.from_numpy(embedding_array)
                    embeddings.append(t)
                    labels.append(label)
                except Exception as e:
                    print(f"Ошибка при обработке строки: {row}. Причина: {str(e)}")
        return embeddings, labels
    
    def get_embedding(self, file_):
        """Получает эмбеддинг для аудиофайла."""
        print(file_)
        audio = whisper.load_audio(file_)
        tensor = whisper.log_mel_spectrogram(audio)
        trim_tensor = whisper.pad_or_trim(tensor, length=3000)
        segment = trim_tensor.to(self.device)
        
        options = DecodingOptions(temperature=0)
        decode_result = self.model.decode(segment, options)
        embedding = decode_result.embedding
        return embedding
    
    def process_dataset(self, dir_, filename):
        """Обрабатывает датасет и записывает эмбеддинги в CSV."""
        if not os.path.exists(filename):  # Если файл не существует, создаём его
            with open(filename, mode='w', newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(['embedding', 'label'])  # Добавляем заголовок
        
        for root, dirs, files in os.walk(dir_, topdown=False):
            for name in files:
                if name.endswith('.ipynb'):  # Пропускаем файлы Jupyter Notebook
                    continue
                file_path = os.path.join(root, name)
                
                emb = self.get_embedding(file_path)
                emb = emb.tolist()
                emb = np.array(emb)
                
                label = int(file_path.split('/')[-2])
                
                data_list = [(emb, label)]
                print(emb.shape, label)
                self.write_data_to_csv(filename, data_list)
    
    def compare_metric(self, embeddings, labels):
        """Сравнивает метрики (косинусное сходство) между эмбеддингами."""
        new_labels = []
        cos_list = []
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):  # Начинаем со следующего элемента после текущего
                if labels[i] == labels[j]:
                    new_labels.append(1)
                else:
                    new_labels.append(0)
                
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[i].squeeze(-2), embeddings[j].squeeze(-2)
                )
                cos_list.append(similarity.item())
        
        index0 = np.where(np.array(new_labels) == 0)
        index1 = np.where(np.array(new_labels) == 1)

        cos_list = np.array(cos_list)
        print(cos_list)
        plt.hist(cos_list[index0], alpha=0.5, label='Different Labels')
        plt.hist(cos_list[index1], alpha=0.5, label='Same Labels')
        plt.legend()
        plt.show()

# Использование класса
processor = WhisperAnalyst(model_size='small')

dir_ = 'dataset'
filename = 'embeddings.csv'

# Обработка датасета
processor.process_dataset(dir_, filename)

# Чтение данных из CSV
embeddings, labels = processor.read_data_from_csv(filename)

# Сравнение метрик
processor.compare_metric(embeddings, labels)