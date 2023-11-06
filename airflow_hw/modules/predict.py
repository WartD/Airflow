import pandas as pd
import dill
import os
import json
import logging
import datetime as dt

# path = os.environ.get('PROJECT_PATH', '..\\')             # Относительный адрес основного каталога
directed_path = os.path.expanduser('~/airflow_hw/')         # Формирование адреса
json_cat = 'data/test/'                                     # Адрес каталога с запросами
m_cat = 'data/models/'                                      # Адрес каталога с моделью

def file_dir(filename):                                      # Функция для генерации адреса
    return directed_path + filename                              # Основной каталог и адрес файла

def predict() -> None:                                       # Основная функция для предсказания
    json_path = file_dir(json_cat)                           # Путь к каталогу для тестов
    json_src = os.listdir(json_path)                         # Список файлов в каталоге

    m_name = os.listdir(file_dir(m_cat))[0]                  # Относительный путь к модели (свежая в списке)
    m_src = os.path.join(file_dir(m_cat), m_name)            # Путь до файла модели
    with open(m_src, 'rb') as pkl:                           # Загрузка данных через joblib не сработала
        model = dill.load(pkl)                                   # Использован dill
    logging.info(f'Model loaded from file: {m_name}')        # Запись в лог о загрузке модели


    def refill(odf):                                         # Функция добавления недостающих данных под модель
        def short_model(x):                                      # Дополнительная функция записи краткой модели
            if not pd.isna(x):                                       # Если ячейка не пуста
                return x.lower().split(' ')[0]                       # Возвращение первого слова
            else:                                                    # или:
                return x                                             # возвращение исходного значения

        odf.loc[:, 'short_model'] = odf['model'].apply(short_model)  # Столбец с кратким названием модели
        odf.loc[:, 'age_category'] = odf['year'].apply(
            lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
        )                                                            # Столбец c категорией возраста
        return odf                                                   # Возвращение преобразованного датафрейма


    def check_json_files(json_list):                                 # Функция проверки списка файлов
        ret_files = []                                                   # Пустой список
        for f in json_list:                                              # Цикл по именам файлов
            if '.json' in f:                                                 # Если встречается нужное расширение
                ret_files.append(f)                                              # Добавить имя файла в список
        return ret_files                                                 # Возвращение списка файлов


    ret_json = check_json_files(json_src)                                   # Формирование списка файлов
    qmsg = 'Queries in .json format found: ' + str(len(ret_json))           # Текст сообщения
    logging.info(qmsg)                                                      # Сообщение о количестве запросов

    for j in ret_json:                                   # Цикл по именам файлов
        with open(json_path + j) as q:                       # Чтение данных из файла
            qform = json.load(q)                             # Преобразование в JSON
        qdf = refill(pd.DataFrame.from_dict([qform]))        # Преобразование в датафрейм с заполением
        if j == json_src[0]:                                     # Если первая строка, то
            predf = qdf                                              # Собираем датафрейм на ее основе
        else:                                                    # остальные -
            predf = pd.concat([predf, qdf], axis=0)                  # добавляем ниже
    predf = predf.reset_index().drop(['index'], axis=1)          # Переиндексация датафрейма
    predf['predicted_price_cat'] = model.predict(predf)          # Столбец с результатом предсказания
    logging.info('Prediction for selected queries ready!')       # Сообщение о выполнении презсказания

    show_columns = ['id', 'price', 'predicted_price_cat']                 # Список столбцов для показа
    upload_dir = file_dir('data/predictions/')                            # Путь до каталога для сохранения
    pred_name = f'pred_{dt.datetime.now().strftime("%Y%m%d%H%M")}.csv'    # Имя файла с датой
    predf[show_columns].to_csv(upload_dir + pred_name)                    # Сохранение среза с предсказаниями
    logging.info(f'Prediction saved in file: {pred_name}')                # Сообщение об успешном сохранении


if __name__ == '__main__':                               # Запуск основной программы
    predict()

