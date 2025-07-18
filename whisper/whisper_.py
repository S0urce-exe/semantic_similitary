import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Загрузка модели и токенизатора Whisper
model_name = "openai/whisper-base"  # Можно использовать "openai/whisper-small" или другие варианты
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = 'cpu'  # Или 'cuda', если доступна GPU


def get_whisper_embedding(text, model=model, tokenizer=tokenizer):
    """
    Получает эмбеддинг текста с использованием модели Whisper.
    
    Args:
        text (str): Входной текст.
        model: Модель Whisper.
        tokenizer: Токенизатор для предобработки текста.
    
    Returns:
        torch.Tensor: Эмбеддинг текста.
    """
    # Токенизация текста
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        # Передаем входные данные в модель
        outputs = model(**inputs)  # Используем распаковку словаря для передачи всех необходимых аргументов
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Усредняем по последовательности

    return embeddings


def calculate_similarity(phrase, templates):
    """
    Вычисляет семантическое сходство между фразой и списком шаблонов с использованием Whisper.
    
    Args:
        phrase (str): Входная фраза.
        templates (list): Список шаблонов.
    
    Returns:
        dict: Результат сравнения с ключами:
            - confidence: Значение схожести.
            - template_number: Номер шаблона с максимальным сходством.
            - best_template: Лучший шаблон.
            - original_phrase: Исходная фраза.
            - best_ngram: Исходная n-грамма.
    """
    # Получение эмбеддинга входной фразы
    phrase_embedding = get_whisper_embedding(phrase)

    # Получение эмбеддингов всех шаблонов
    template_embeddings = torch.cat(
        [get_whisper_embedding(template) for template in templates]
    )

    # Вычисление косинусного сходства между фразой и шаблонами
    similarities = torch.nn.functional.cosine_similarity(
        phrase_embedding, template_embeddings, dim=1
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
        "best_ngram": phrase  # Для простоты используем всю фразу как n-грамму
    }

    return result


# Пример использования
templates = [
    "Да, слушаю Вас, машинист локомотива на подходе к станции Бабынино.",
    "Слушаю Вас машинист поезда <Число> <ФАМИЛИЯ>.",
    "Прибываете на <NUMBER>-й путь тупиковый ДСП Шилов."
]

phrase = "Аллё, да, прибываете на пятый путь выходной ДСП 1 Панков."

result = calculate_similarity(phrase, templates)

# Вывод результата в требуемом формате
print(f"Confidence: {result['confidence']:.7f}")
print(f"Фраза: {result['original_phrase']}")
print(f"Номер шаблона с максимальным сходством: {result['template_number']}")
print(f"Шаблон: {result['best_template']}")
print(f"Исходная n-грамма: {result['best_ngram']}")