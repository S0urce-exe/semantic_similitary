import sys
import html
from PySide6.QtWidgets import (
    QApplication, QComboBox, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit, QFileDialog, QWidget
)
from PySide6.QtCore import Qt
import numpy as np
from net_embeding2 import TextProcessor


class TextProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оценка смысловой идентичности GUI")
        self.setGeometry(100, 100, 800, 600)

        # Создаем экземпляр TextProcessor
        self.processor = None

        # Основной виджет и макет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Поле для ввода фразы
        self.phrase_label = QLabel("Введите фразу:")
        self.phrase_input = QTextEdit()
        self.phrase_input.setPlaceholderText("Введите текст для анализа...")
        self.layout.addWidget(self.phrase_label)
        self.layout.addWidget(self.phrase_input)

        # Кнопка для загрузки шаблонов
        self.template_button = QPushButton("Загрузить шаблоны из CSV")
        self.template_button.clicked.connect(self.load_templates)
        self.layout.addWidget(self.template_button)

        # Кнопка для выполнения анализа
        self.analyze_button = QPushButton("Выполнить анализ")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.analyze_button)

        # Выбор метрики
        self.metric_layout = QHBoxLayout()
        self.metric_label = QLabel("Выберите метрику:")
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Косинусное сходство", "Расстояние Джаро-Винклера", "Расстояние Левенштейна"])
        self.metric_layout.addWidget(self.metric_label)
        self.metric_layout.addWidget(self.metric_combo)
        self.layout.addLayout(self.metric_layout)

        # Выбор модели
        self.model_layout = QHBoxLayout()
        self.model_label = QLabel("Выберите модель:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TF-IDF", "BERT", "Whisper"])
        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_combo)
        self.layout.addLayout(self.model_layout)

        # Поле для вывода результатов
        self.result_label = QLabel("Результаты анализа:")
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.result_output)

    def load_templates(self):
        """Загружает шаблоны из CSV-файла."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл с шаблонами", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_path:
            try:
                # Инициализируем TextProcessor с загруженными шаблонами
                self.processor = TextProcessor(templates=file_path, column_name="Форма передачи текста (команда, указание, сообщения)  и действия работников ")
                self.result_output.append(f"Шаблоны успешно загружены из файла: {file_path}")
            except Exception as e:
                self.result_output.append(f"Ошибка при загрузке шаблонов: {e}")

    def run_analysis(self):
        """Выполняет анализ введенной фразы."""
        if not self.processor:
            self.result_output.append("Сначала загрузите шаблоны!")
            return

        phrase = self.phrase_input.toPlainText().strip()
        if not phrase:
            self.result_output.append("Введите фразу для анализа!")
            return

        # Получаем выбранную метрику
        dct = {
            'Косинусное сходство': 'cos',
            'Расстояние Джаро-Винклера': 'jaro_winkler',
            'Расстояние Левенштейна': 'lev'
        }
        
        selected_metric = self.metric_combo.currentText()
        metric_value = dct.get(selected_metric)

        selected_model = self.model_combo.currentText()

        try:
            if selected_model == 'TF-IDF':
                # Выполняем анализ с учетом выбранной метрики
                self.processor.run(phrase=phrase, ngrams_len=(2, 37), metric=metric_value.lower())
                max_index = np.argmax(self.processor.cos_matrix)  # Индекс максимального значения
                template_number = max_index // self.processor.cos_matrix.shape[1] + 1  # Номер шаблона (1-based)
                original_template = self.processor.original_templates[template_number - 1]
                # Форматируем результаты с использованием HTML для цветного текста
                original_template_escaped = html.escape(original_template)
                original_phrase, confidence, template, n_gramm = self.processor.original_phrase, f'{self.processor.max_value:.4f}', original_template_escaped, self.processor.ish_ngr
                # result_text = (
                #     f'<br>Фраза: <b><span style="color: red;">{html.escape(self.processor.original_phrase)}</span></b><br>'
                #     f'Confidence: <b><span style="color: green;">{self.processor.max_value:.4f}</span></b><br>'
                #     f'Номер шаблона с максимальным сходством: <b><span style="color: green;">{template_number}</span></b><br>'
                #     f'Шаблон: <b><span style="color: red;">{original_template_escaped}</span></b><br>'
                #     f'Исходная n-грамма: <b><span style="color: red;">{html.escape(self.processor.ish_ngr)}</span></b><br>'
                #     f'Метрика: <b><span style="color: green;">{html.escape(selected_metric)}</span></b><br>'
                # )
                # self.result_output.insertHtml(result_text)
            elif selected_model == 'BERT':
                # from bert.net_embeding import run
                from bert.net_embeding import BertForSTS
                sts_model = BertForSTS()
                result = sts_model.run(phrase=phrase, templates=self.processor.original_templates)

                original_phrase, confidence, template, template_number, n_gramm = result['original_phrase'], result['confidence'], result['best_template'], result['template_number'], result['ish_ngramm']
            elif selected_model == "Whisper":
                print('dfg')
            result_text = (
                    f'<br>Фраза: <b><span style="color: red;">{html.escape(original_phrase)}</span></b><br>'
                    f'Confidence: <b><span style="color: green;">{confidence}</span></b><br>'
                    f'Номер шаблона с максимальным сходством: <b><span style="color: green;">{template_number}</span></b><br>'
                    f'Шаблон: <b><span style="color: red;">{template}</span></b><br>'
                    f'Исходная n-грамма: <b><span style="color: red;">{html.escape(n_gramm)}</span></b><br>'
                    f'Метрика: <b><span style="color: green;">{html.escape(selected_metric)}</span></b>'
                )
            
            self.result_output.insertHtml(result_text)
            
        except Exception as e:
            self.result_output.append(f"Ошибка при выполнении анализа: {e}")
            print(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextProcessorApp()
    window.show()
    sys.exit(app.exec())