# import torch
# from transformers import BertTokenizer
# from sentence_transformers import SentenceTransformer, models

# device = 'cpu'

# class BertForSTS(torch.nn.Module):
#     def __init__(self):
#         super(BertForSTS, self).__init__()
#         # Инициализация BERT и слоя пулинга
#         self.bert = models.Transformer("DeepPavlov/rubert-base-cased", max_seq_length=128)
#         self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
#         self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

#     def forward(self, input_data):
#         # Получение эмбеддингов предложений
#         output = self.sts_bert(input_data)['sentence_embedding']
#         return output


# # Инициализация токенизатора и модели
# tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# model_emb = BertForSTS()


# def run(phrase, templates, model=model_emb, tokenizer=tokenizer):
#     """
#     Вычисляет семантическое сходство между фразой и списком шаблонов.
    
#     Args:
#         phrase (str): Входная фраза.
#         templates (list): Список шаблонов.
#         model: Модель для вычисления эмбеддингов.
#         tokenizer: Токенизатор для предобработки текста.
    
#     Returns:
#         dict: Результат сравнения с ключами:
#             - confidence: Значение схожести.
#             - template_number: Номер шаблона с максимальным сходством.
#             - best_template: Лучший шаблон.
#             - original_phrase: Исходная фраза.
#             - best_ngram: Исходная n-грамма.
#     """
#     # Токенизация входной фразы
#     phrase_input = tokenizer(
#         phrase,
#         padding='max_length',
#         max_length=128,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     # Удаление token_type_ids, если они есть
#     if 'token_type_ids' in phrase_input:
#         del phrase_input['token_type_ids']

#     # Токенизация всех шаблонов
#     template_inputs = [
#         tokenizer(
#             template,
#             padding='max_length',
#             max_length=128,
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)
#         for template in templates
#     ]

#     # Удаление token_type_ids из шаблонов
#     for inp in template_inputs:
#         if 'token_type_ids' in inp:
#             del inp['token_type_ids']

#     # Получение эмбеддингов для входной фразы
#     with torch.no_grad():
#         phrase_embedding = model(phrase_input)

#     # Получение эмбеддингов для всех шаблонов
#     template_embeddings = []
#     with torch.no_grad():
#         for inp in template_inputs:
#             emb = model(inp)
#             template_embeddings.append(emb)

#     # Объединение эмбеддингов шаблонов в один тензор
#     template_embeddings = torch.cat(template_embeddings, dim=0)  # Размерность: [num_templates, embedding_dim]

#     # Вычисление косинусного сходства между фразой и каждым шаблоном
#     similarities = torch.nn.functional.cosine_similarity(
#         phrase_embedding[0], template_embeddings, dim=1
#     ).cpu().numpy()

#     # Нахождение индекса шаблона с максимальной схожестью
#     best_index = similarities.argmax()
#     best_similarity = similarities[best_index]

#     # Формирование результата
#     result = {
#         "confidence": best_similarity,
#         "template_number": best_index + 1,  # Номер шаблона (1-based)
#         "best_template": templates[best_index],
#         "original_phrase": phrase,
#         "ish_ngramm": phrase 
#     }
#     print(f"Confidence: {result['confidence']:.7f}")
#     print(f"Фраза: {result['original_phrase']}")
#     print(f"Номер шаблона с максимальным сходством: {result['template_number']}")
#     print(f"Шаблон: {result['best_template']}")
#     print(f"N-грамма: {result['ish_ngramm']}")
#     return result


# # Пример использования
# templates = [
#     # "Да, слушаю Вас, машинист локомотива на подходе к станции Бабынино.",
#     # "Слушаю Вас машинист поезда <Число> <ФАМИЛИЯ>.",
#     "Прибываете на <NUMBER>-й путь тупиковый ДСП Шилов."
# ]

# phrase = "Аллё, да, прибываете на пятый путь выходной ДСП 1 Панков."

# result = run(phrase, templates)

# # Вывод результата в требуемом формате
# print(f"Confidence: {result['confidence']:.7f}")
# print(f"Фраза: {result['original_phrase']}")
# print(f"Номер шаблона с максимальным сходством: {result['template_number']}")
# print(f"Шаблон: {result['best_template']}")

import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models


class BertForSTS:
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", max_seq_length=128, device='cpu'):
        """
        Инициализация модели для вычисления семантического сходства.
        
        Args:
            model_name (str): Название предобученной модели BERT.
            max_seq_length (int): Максимальная длина последовательности.
            device (str): Устройство для выполнения вычислений ('cpu' или 'cuda').
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = self._initialize_model(model_name, max_seq_length)
        self.model.to(self.device)

    def _initialize_model(self, model_name, max_seq_length):
        """
        Инициализация модели BERT для вычисления эмбеддингов.
        
        Args:
            model_name (str): Название предобученной модели BERT.
            max_seq_length (int): Максимальная длина последовательности.
        
        Returns:
            SentenceTransformer: Инициализированная модель.
        """
        bert = models.Transformer(model_name, max_seq_length=max_seq_length)
        self.pooling_layer = models.Pooling(bert.get_word_embedding_dimension())
        return SentenceTransformer(modules=[bert, self.pooling_layer])

    def tokenize(self, text):
        """
        Токенизация текста.
        
        Args:
            text (str): Входной текст.
        
        Returns:
            dict: Токенизированный текст в формате PyTorch.
        """
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Удаление token_type_ids, если они есть
        if 'token_type_ids' in tokenized:
            del tokenized['token_type_ids']
        
        return tokenized

    def get_embeddings(self, input_data):
        """
        Получение эмбеддингов для входных данных.
        
        Args:
            input_data (dict): Токенизированные данные.
        
        Returns:
            torch.Tensor: Эмбеддинги предложений.
        """
        with torch.no_grad():
            embeddings = self.model(input_data)['sentence_embedding']
        return embeddings

    def run(self, phrase, templates):
        """
        Вычисление семантического сходства между фразой и списком шаблонов.
        
        Args:
            phrase (str): Входная фраза.
            templates (list): Список шаблонов.
        
        Returns:
            dict: Результат сравнения с ключами:
                - confidence: Значение схожести.
                - template_number: Номер шаблона с максимальным сходством.
                - best_template: Лучший шаблон.
                - original_phrase: Исходная фраза.
                - ish_ngramm: Исходная n-грамма.
        """
        # Токенизация входной фразы
        phrase_input = self.tokenize(phrase)

        # Токенизация всех шаблонов
        template_inputs = [self.tokenize(template) for template in templates]

        # Получение эмбеддингов для входной фразы
        phrase_embedding = self.get_embeddings(phrase_input)

        # Получение эмбеддингов для всех шаблонов
        template_embeddings = torch.cat(
            [self.get_embeddings(inp) for inp in template_inputs], dim=0
        )

        # Вычисление косинусного сходства между фразой и каждым шаблоном
        similarities = torch.nn.functional.cosine_similarity(
            phrase_embedding[0], template_embeddings, dim=1
        ).cpu().numpy()

        # Нахождение индекса шаблона с максимальной схожестью
        best_index = similarities.argmax()
        best_similarity = similarities[best_index]

        # Формирование результата
        result = {
            "confidence": best_similarity,
            "template_number": best_index + 1,  # Номер шаблона (1-based)
            "best_template": templates[best_index],
            "original_phrase": phrase,
            "ish_ngramm": phrase
        }

        # Вывод результатов в консоль
        print(f"Confidence: {result['confidence']:.7f}")
        print(f"Фраза: {result['original_phrase']}")
        print(f"Номер шаблона с максимальным сходством: {result['template_number']}")
        print(f"Шаблон: {result['best_template']}")
        print(f"N-грамма: {result['ish_ngramm']}")

        return result


# # Пример использования
# if __name__ == "__main__":
#     templates = [
#         "Прибываете на <NUMBER>-й путь тупиковый ДСП Шилов."
#     ]

#     phrase = "Аллё, да, прибываете на пятый путь выходной ДСП 1 Панков."

#     sts_model = BertForSTS()
#     result = sts_model.run(phrase, templates)

#     # Вывод результата в требуемом формате
#     print(f"Confidence: {result['confidence']:.7f}")
#     print(f"Фраза: {result['original_phrase']}")
#     print(f"Номер шаблона с максимальным сходством: {result['template_number']}")
#     print(f"Шаблон: {result['best_template']}")