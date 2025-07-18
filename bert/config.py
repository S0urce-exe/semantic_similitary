#from default_values.py, перенос.
SAMPLING_RATE = 16000
DT = 29
D = SAMPLING_RATE * DT
N_D = 30

#старые константы, которые, КАЖЕТСЯ, нигде не используются.
# длина эпизода записи в с. 
#dt = DT # длина отрывка в с
#d = SAMPLING_RATE * DT # длина отрывка в числе точек
# = N_D # коэффициент перекрытия
# частота записей
#sampling_rate =  SAMPLING_RATE

# имена моделей
#DIR_MODEL= '/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v3/'

SETTINGS_FILENAME = 'asp.ini'
                 
DIR_MODEL= '../Base Classification/'
DIR_MODEL= '/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v3/'
dir_model = DIR_MODEL

#Представление параметров моделей в виде объекта.
#Нужно для автоматизации загрузки моделей.
MODELS_PARAMS = {0:{'asp_onnx':1,
                    'model_name_speech_to_text':[dir_model+'whisper-tiny_onnx', "openai/whisper-base", "openai/whisper-tiny"],
                  #   'vocabulary_path':dir_model + 'pretrained_models/speech_model/fine_tuned_whisper_large_new/vocab.json'},
                  'vocabulary_path':'vocabulary_norm.json'},
                 1:{'asp_onnx':0,
                    'model_name_speech_to_text':["jonatasgrosman/wav2vec2-large-xlsr-53-russian"]*3,
                  #   'vocabulary_path':dir_model + 'pretrained_models/speech_model/fine_tuned_whisper_large_new/vocab.json'},
                    'vocabulary_path':'vocabulary_norm.json'},
                 2:{'asp_onnx':2,
                    'model_name_speech_to_text':[dir_model + "pretrained_models/speech_model/fine_tuned_whisper_large_new"]*3,
                  #   'vocabulary_path':dir_model + 'pretrained_models/speech_model/fine_tuned_whisper_large_new/vocab.json'}
                  'vocabulary_path':'vocabulary_norm.json'},
                 3:{'asp_onnx':3,
                    'model_name_speech_to_text':[dir_model + "pretrained_models/speech_model/fine_tuned_whisper_large_new"]*3,
                  #   'vocabulary_path':dir_model + 'pretrained_models/speech_model/fine_tuned_whisper_large_new/vocab.json'}
                  'vocabulary_path':'vocabulary_norm.json'}, 
                    }
#Номер варианта набора моделей из объекта выше
VARIANT = 3
dir_model_e  = dir_model + 'pretrained_models/'    
# model_name_speech_to_emo = [dir_model+"wav2vec2-xls-r-300m-emotion-ru_onnx", "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"]
# model_name_text_to_emo = [dir_model+"rubert-tiny2-russian-emotion-detection_onnx", "Djacon/rubert-tiny2-russian-emotion-detection"]
model_name_speech_to_emo = [dir_model_e + "wav2vec2-xls-r-300m-emotion-ru_onnx", dir_model_e + "wav2vec2-xls-r-300m-emotion-ru"]
model_name_text_to_emo = [dir_model_e + "rubert-tiny2-russian-emotion-detection_onnx", dir_model_e + "emotion_text/rubert-tiny2-russian-emotion-detection"]


class_list = {0:['enthusiasm', 'happiness', 'neutral','positive', 'surprise','other','normal'], 1:['angry', 'disgust', 'fear', 'sadness', 'sad']}
emo_class = {'normal':0, 'positive':0, 'surprise':0.3, 'neutral':0,'enthusiasm':0.1, 'happiness':0, 'joy':0.5,'angry':1, 'anger':1, 'disgust':0.5, 'fear':0.3, 'sadness':0.8, 'sad':0.8, 'other':0.5 }
weigth_model = [0.5, 0.5]
WER_COEFF = 0.9
# resd
# {'anger', 'disgust', 'enthusiasm', 'fear', 'happiness', 'neutral', 'sadness'}

