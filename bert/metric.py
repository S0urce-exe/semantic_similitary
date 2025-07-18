import os
import sys

from difflib import SequenceMatcher
import jellyfish
import numpy as np
import pandas as pd
import json
import config
import net_embeding as net
# import seaborn as sns
import matplotlib.pyplot as plt
import torch

from num2words import num2words
import pymorphy2

def pymorphy2_311_hotfix():
    from inspect import getfullargspec
    from pymorphy2.units.base import BaseAnalyzerUnit

    def _get_param_names_311(klass):
        if klass.__init__ is object.__init__:
            return []
        args = getfullargspec(klass.__init__).args
        return sorted(args[1:])

    setattr(BaseAnalyzerUnit, '_get_param_names', _get_param_names_311)

python_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
print("Работает версия Python", python_version)
if sys.version_info.minor>=11:
    pymorphy2_311_hotfix()
normal_word = pymorphy2.MorphAnalyzer(lang='ru',)


# чтение словаря для модели
with open(config.MODELS_PARAMS[config.VARIANT]['vocabulary_path']  , 'r', encoding='utf-8') as f:
      vocabulary = json.load(f)
print(max(list(vocabulary.values())), len(list(vocabulary.values())))
voc_len = len(vocabulary) + 20
vocabulary['<MASK>'] = voc_len - 2
vocabulary['<NUMBER>'] = voc_len - 1 
vocabulary['<UNKNOWN>'] = voc_len - 3 
print(max(list(vocabulary.values())), len(list(vocabulary.values())), voc_len)

# чтение нейросети для эмбединга
try:
    embeding_net = net.load_net()
    # embeding_net = 
except:
	embeding_net = net.Embeding(n=256, voc_len=voc_len ) 
	net.save_net(embeding_net)
	print('NEW Embeding Model')
embeding_net = net.Embeding(n=256, voc_len=voc_len ) 
net.save_net(embeding_net)

def norm_word(s):
    # print(s)
    s_list = s.split(' ')
    s_list = [normal_word.normal_forms(si)[0] for si in s_list]
    s = ' '.join(s_list)
    return s

def num2string(s):
    # поиск и замена цифр на слова
    line = s
    s_new = ' '
    numbers = ''.join(c if c.isdigit() else ' ' for c in line).split()
    for i in range(len(numbers)):
        k = s.find(numbers[i])
        

        s_new = s_new + ' ' +s[:k] + ' ' + num2words(numbers[i], lang='ru') + ' '
        
        s = s[k+len(numbers[i]):]
    s_new = s_new = s_new + ' ' + s   + ' '  
    return s_new 

def diarization_manual(df_speeker):
    # парсинг диаризации из разметки
    
    s = df_speeker.iloc[:,0].unique()
    t0 = df_speeker.iloc[:,3].values.min()
    t1 = int(df_speeker.iloc[:,3].values.max()) + 1
    t = -np.ones([t1, 1])
    for k, si in enumerate(s):
        dff = df_speeker.loc[df_speeker.iloc[:, 0]==si, :]
        for i in range(dff.shape[0]):
            row = dff.iloc[i, :].values
            tt1 = int(row[3])
            tt2  = int(row[5])
            t[tt1:tt2, 0] = k
    return t, t1

def count_time_diarization(d_auto, d_manual):
    # оценка метрик времени диаризации
    acc = []
    for s in set(d_manual.tolist()):
        if s!= -1:
            ind = np.where(d_manual==s)
            gi = d_auto[ind].astype(np.int32)
            g_unique = list(set(gi))
            if len(gi)!=0:
                acci = np.array([np.sum(gi == 0) for g_acc in g_unique]) / len(gi)
            else:
                acci = [0]
            # print(acci)    
            # acc += [np.sum(acci) * len(gi)/len(d_auto)]  
            acc += [ np.sum(gi == 0)/len(d_auto) ]
            print(gi)
    ind = np.where(d_manual==-1)
    gi = d_auto[ind].astype(np.int32)
    acc += [np.sum(gi != 0)/len(d_auto) ]   
    print(gi)   
    print(acc)
    # metric_t = 1 - np.sum(acc) * (len(set(d_auto.tolist()))/len(set(d_manual.tolist())))
    metric_t = 1 - np.sum(acc) #* (len(set(d_auto.tolist()))/len(set(d_manual.tolist())))
    return metric_t

def count_time_emotion(d_auto, d_manual):
    # оценка метрик времени эмоций
    acc = np.abs(d_auto - d_manual)
    metric_t = 1 - np.mean(acc)
    return metric_t

def calculate_wer(reference, hypothesis):
    # оценка качества транскрибирования 
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer

def calculate_wer_soft(reference, hypothesis):
    # оценка качества транскрибирования  с разрешением одной-двух ошибок в слове
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    k = 0
    substitutions = 0
    # for i in range(len(ref_words )):
    if 1:
         rez_d = [[ jellyfish.jaro_similarity(ref,hyp) for hyp in hyp_words] for ref in ref_words]


        # print(rez_d)
        #  plt.imshow(rez_d)
        #  plt.show()

         k_ind = np.argmax(rez_d, axis=1)

         for k, vv in enumerate(list(zip(ref_words, np.array(hyp_words)[k_ind].tolist()))):

            if rez_d[k][k_ind[k]]<config.WER_COEFF:
                # print(vv, rez_d[k][k_ind[k]])
                substitutions += 1
            
        
        #  plt.imshow(pd.DataFrame(rez_d, index=ref_words, columns=hyp_words).values)
        #  plt.show()
         
    # print('wer : ',substitutions,len(ref_words) , len(hyp_words) )
    # substitutions = sum((1 for ref, hyp in zip(ref_words, hyp_words) if jellyfish.jaro_similarity(ref,hyp)>0.8))
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    total_words = len(ref_words)
    if (total_words>0):
        wer = (substitutions + deletions + insertions) / total_words
    else:
        wer = 0  
    # print(wer )       
    return wer

def zerok(k):
        zero_k = torch.zeros([1, voc_len]).float()
        zero_k[0, k] = 1.0
        vect = embeding_net(zero_k).cpu().detach().numpy()
        return vect

def find_shablon(ref_embedings, s):
    """
    dff - таблица pandas с шаблонами общения
    s - строка текущего разбора

    формируем эмбединги (случайная нейронка) и сравниваем их с текущей строкой ввода s 
    """
    
    d = []
    hyp_words = s.split(' ')

    # hyp_embeding = np.mean([ zerok(vocabulary[si]) if si in list(vocabulary.keys()) else zerok(-3)   for si in hyp_words if len(s)>0  ], axis=0)
    hyp_embeding = net.embeding_run(s).cpu().detach().numpy()
    if not isinstance(hyp_embeding, type(np.array([0.0]))):
            hyp_embeding = np.array([[100] * 768])
            # hyp_embeding = np.array([[100] * 256])
    rez_d = []

    for ref_embeding in ref_embedings:        
        rez_d.append(((np.dot(ref_embeding.reshape(-1), hyp_embeding.reshape(-1)))/(np.linalg.norm(hyp_embeding) + 1)/(np.linalg.norm(ref_embeding) + 1) -  1))
    # print(rez_d)
    k = np.argmax(rez_d)
    dist = -rez_d[k]  if rez_d[k]!=0.0 else 0.0 
    # номер шалона , который наиболее близок и лучшее расстояние 
    
    return k, dist
    


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def embeding_match(emb1, emb2, n = 512, dist=0):
    '''
    emb1 - Nxn array, 
    emb2 - 1xn array
    '''
    if len(emb1.shape)==2:
        n = emb1.shape[1]
        
    assert emb1.shape[1]==emb2.shape[1], print("no equvivalent shape")
    if dist==0:
            D = np.sum((emb1.reshape(-1,n) - emb2.reshape(1, -1)) ** 2, axis=1)
    return D 

def embeding(text):
    embeding = np.zeros([1,voc_len])
    list_text = text.split(' ')
    for word in list_text:
          embeding[vocabulary[word]] = 1

def getStringAuto(df_auto, dt = 1):
    # print(df_manual)
    new_df = pd.DataFrame({ })
    delta_speech_nooverlay = dt
    new_df['speech auto'] = 0
    new_df['speech'] = ''
    new_df['emotion auto'] = 0
    new_df['emotion'] = 0
    new_df['time'] = 0
    new_df['diarization'] = 0    
    s_auto = ''
    if delta_speech_nooverlay==1:
        new_df['speech auto'] = df_auto.speech_eff.values
        new_df['emotion auto'] = df_auto.emotion.values
        new_df['speech'] = ''
        new_df['emotion'] = 0
        new_df['time'] = df_auto.index.values
        new_df['diarization auto'] = df_auto.diarization.values
        new_df['diarization'] = -1
    else:
        for k in range(df_auto.shape[0]):
            text = df_auto.speech_eff.values[k]
            dtext = len(text) // int(delta_speech_nooverlay)
            list_text = [ text[m*dtext:(m+1)*dtext] for m in range(int(delta_speech_nooverlay))]
            for m in range(delta_speech_nooverlay):
                new_df['speech auto'].values[m + k * delta_speech_nooverlay] = list_text[m]
                new_df['emotion auto'].values[m + k * delta_speech_nooverlay] = df_auto.emotion.values[k]
                new_df['speech'].values[m + k * delta_speech_nooverlay] = ''
                new_df['emotion'].values[m + k * delta_speech_nooverlay] = 0
                new_df['time'].values[m + k * delta_speech_nooverlay] = df_auto.index.values[k]
                new_df['diarization'].values[m + k * delta_speech_nooverlay] = df_auto.diarization.values[k]           
    new_df.fillna('', inplace=True)
    s_auto = ' '.join([new_df["speech auto"].values[i] for i in range(new_df.shape[0]) if new_df["speech auto"].values[i] != np.nan ]).replace('.', ' ').replace(',', ' ').replace('  ',' ').lower()    
    s_auto  = s_auto.replace('  ',' ')      
    s_auto = num2string(s_auto)
    return s_auto

def parcer_result_table(df_manual, df_auto, dt = 1):
    # print(df_manual)
    new_df = pd.DataFrame({ })
    delta_speech_nooverlay = dt
    new_df['speech auto'] = 0
    new_df['speech'] = ''
    new_df['emotion auto'] = 0
    new_df['emotion'] = 0
    new_df['time'] = 0
    new_df['diarization'] = 0
    s_manual = ''
    s_auto = ''
    if delta_speech_nooverlay==1:
        new_df['speech auto'] = df_auto.speech_eff.values
        new_df['emotion auto'] = df_auto.emotion.values
        new_df['speech'] = ''
        new_df['emotion'] = 0
        new_df['time'] = df_auto.index.values
        new_df['diarization auto'] = df_auto.diarization.values
        new_df['diarization'] = -1
    else:
        for k in range(df_auto.shape[0]):
            text = df_auto.speech_eff.values[k]
            dtext = len(text) // int(delta_speech_nooverlay)
            list_text = [ text[m*dtext:(m+1)*dtext] for m in range(int(delta_speech_nooverlay))]
            for m in range(delta_speech_nooverlay):
                new_df['speech auto'].values[m + k * delta_speech_nooverlay] = list_text[m]
                new_df['emotion auto'].values[m + k * delta_speech_nooverlay] = df_auto.emotion.values[k]
                new_df['speech'].values[m + k * delta_speech_nooverlay] = ''
                new_df['emotion'].values[m + k * delta_speech_nooverlay] = 0
                new_df['time'].values[m + k * delta_speech_nooverlay] = df_auto.index.values[k]
                new_df['diarization'].values[m + k * delta_speech_nooverlay] = df_auto.diarization.values[k]
                
    ind = [i for i in range(df_manual.shape[0]) if df_manual.iloc[:,0].values[i][:6]=='Спикер']
    inde = [i for i in range(df_manual.shape[0]) if df_manual.iloc[:,0].values[i][:6]=='Эмоция']
    dff = df_manual.iloc[ind,:]
    for i in range(dff.shape[0]):
        row = dff.iloc[i,:]
        # print(row[8])
        speeker = int(row[0][6:])
        t1 = round(float(row[3]))
        dt = round(float(row[7]))
        print(row[8])
        if row[8] == np.nan:
            row[8] = ' '
        row[8] =  str(row[8])
        row[8] = " " +row[8] + ' '
        if int(dt)>0:

            dtext = len(row[8]) // int(dt)
        else:
            dtext = 0    
        list_text = [ row[8][k*dtext:(k+1)*dtext] for k in range(int(dt))] + [row[8][dtext*dt:] + '  ']
        
        for k in range(dt):
            if int(t1+ k)>=new_df.shape[0]:
                new_df.append({'time':int(t1+ k)})
                
            new_df['speech'].values[int(t1+ k)] = list_text[k] 
            
            new_df['diarization'].values[int(t1+ k)] = speeker
        
            # print(new_df)
            # print('*******************************')
    new_df['speech'].values[int(t1+ k)] +=  list_text[-1] 
    dff = df_manual.iloc[inde,:]
    for i in range(dff.shape[0]):
        row = dff.iloc[i,:]

        t1 = round(float(row[3]))
        dt = round(float(row[7]))
        dtext = int(row[8])
        
        for k in range(dt):
            if int(t1+ k)>=new_df.shape[0]:
                new_df.append({'time':int(t1+ k)})
                
            new_df['emotion'].values[int(t1+ k)] = dtext         
    new_df.fillna('', inplace=True)
    s_auto = ' '.join([new_df["speech auto"].values[i] for i in range(new_df.shape[0]) if new_df["speech auto"].values[i] != np.nan ]).replace('.', ' ').replace(',', ' ').replace('  ',' ').lower()
    s_manual = ''.join([new_df.speech.values[i] for i in range(new_df.shape[0]) if new_df.speech.values[i] != np.nan ]).replace('.', ' ').replace(',', ' ').replace('  ',' ').lower()
    s_manual = s_manual.replace('  ',' ')    
    s_auto  = s_auto.replace('  ',' ')  
    s_manual = num2string(s_manual)
    s_auto = num2string(s_auto)
    wer = calculate_wer_soft(s_manual,s_auto)

    ind = [ i for i in range(df_manual.shape[0]) if df_manual.iloc[i, 0][:6]=='Спикер']
    df_speeker = df_manual.iloc[ind,:]
    t_diarisation, tk  = diarization_manual(df_speeker)
    new_df['diarization'].values[:tk] = np.array(t_diarisation).reshape(-1)
    # print(new_df)

    diarisation_metric = count_time_diarization(new_df['diarization auto'].values, new_df['diarization'].values)
    emotion_metric = count_time_emotion(new_df['emotion auto'].values, new_df['emotion'].values)


        
    return new_df, {'string_auto':s_auto, 'string_manual':s_manual, 'wer':wer, "f1_emotion":emotion_metric, "f1_diarization":diarisation_metric}

def culculate_metric(reference, hypothesis, n = 1):
    # print(reference, hypothesis)
    reference = reference.replace('  ', ' ')
    hypothesis = hypothesis.replace('  ', ' ')
    # print(reference, hypothesis)
    list_ref = reference.split(' ')

    list_hip = hypothesis.split(' ')
    # print(list_ref, list_hip)
    k0 = 0
    wer = []
    for i in range(len(list_ref)-n):
        n_gram_ref = ' '.join(list_ref[i:i+n])
        r_distance = []
        for j in range(k0, len(list_hip)-n):
            n_gram_hip = ' '.join(list_hip[i:i+n])

            r_distance.append(calculate_wer_soft(n_gram_ref ,n_gram_hip ))
        k_dist = np.argmax(r_distance)
        wer.append(r_distance[k_dist])
        # print(r_distance[k_dist], ' '.join(list_hip[k_dist:k_dist+n]), n_gram_ref  )
        if r_distance[k_dist]>0.7:
            k0 = k_dist
        # print(k0, r_distance)
    return np.sum(wer)/len(list_ref)      

def get_ref_embeddings(df_shnablo):
    # ref_embedings = []
    # for ref_ in df_shnablo.iloc[:,[-1]].values.tolist():
    #     # print(ref_)
    #     ref_words = ref_[0].split(' ')
    #     ref_embeding = np.mean([zerok(vocabulary[si]) if si in list(vocabulary.keys()) else zerok(-3)   for si in ref_words if len(si)>0  ], axis=0)
    #     if not isinstance(ref_embeding, type(np.array([0.0]))):
    #         ref_embeding = np.array([[100] * 512])
    #     ref_embedings.append(ref_embeding)
    ref_embedings =  net.gen_embeding(df_shnablo.iloc[:,[-1]].values.reshape(-1).tolist()).cpu().detach().numpy()    
    return ref_embedings

if __name__=='__main__':
    import difflib as df

    # Перед тем как сравнивать, разобьем
    # тексты на строки


    # print('START ')

    path_manual = '/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v4/audio_speech_interface_02112024_pythononly/rez/04000040_A01.csv'
    path_automation = "/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v4/audio_speech_interface_02112024_pythononly/rez/rez_2024-11-15 17.15.59_source_''04000040_A01'.csv"
    
    path_manual = '/home/boss/Рабочий стол/РЖД_проверены/test/csv/04010036_A09.csv'
    path_automation = "/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v4/audio_speech_interface_02112024_pythononly/rez/rez_2024-11-27 16.27.23_source_''04010036_A09'.csv"

    df_auto = pd.read_csv(path_automation)
    df_manual = pd.read_csv(path_manual, header=None)

    new_df, rez = parcer_result_table(df_manual, df_auto, dt = 1)
    s1 = rez['string_auto']
    s2 = rez['string_manual']
    print(' auto :', s1)
    print(' manual :', s2)
    print(rez)
    print(new_df)

    # получение наиболее близкого шаблона (df_shablon - таблица шаблона, k номер строки лучшего шаблона)
    df_shnablo = pd.read_csv('/home/boss/Рабочий стол/анализ эмоциональности/APP_ASR_Emotion_v4/audio_speech_interface_02112024_pythononly/dispetcher_3.csv', header=0, index_col=0)
    print(df_shnablo.head())

    # генерация эмбедингов строк таблицы разговоров 
    ref_embedings = []
    for ref_ in df_shnablo.iloc[:,[-1]].values.tolist():
        print(ref_)
        ref_words = ref_[0].split(' ')
        ref_embeding = np.mean([zerok(vocabulary[si]) if si in list(vocabulary.keys()) else zerok(-3)   for si in ref_words if len(si)>0  ], axis=0)
        
        if not isinstance(ref_embeding, type(np.array([0.0]))):
            ref_embeding = np.array([[100] * 512])
        ref_embedings.append(ref_embeding)
    
    ref_embedings =  net.gen_embeding(df_shnablo.iloc[:,[-1]].values.reshape(-1).tolist()).cpu().detach().numpy()

    # работа сновым сообщением
    s1_test = s1
    s1_test = norm_word(s1)
    print(s1_test)
    k, dist = find_shablon(ref_embedings , s1_test)
    # вернем номер шалона , который наиболее близок, и лучшее расстояние 
    print(s1, df_shnablo.iloc[k,-2], k, dist)    
    print('END')
  