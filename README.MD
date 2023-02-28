### ⚠️Readme⚠️
Версия Python 3.9, Elasticsearch 6.8.2

#### 1. Установка зависимостей
Для установки зависимостей:
```bash
$ pip install -r ./doc/requirements.txt
```
#### 2. Настройка узла Elasticsearch
Для запуска узла Elasticsearch использовать docker-compose:
```bash
$ sudo sysctl -w vm.max_map_count=262144

$ docker-compose up -d 
```

#### 3. Сбор базы данных
Перед запуском сбора необходимо удостовериться, что создан экземпляр Elasticsearch  

Также необходимо скачать дамп с ресурса wikidata(3.5GB):  
https://archive.org/details/wikidata-json-20150202  
И поместить его в директорию Vector-base-NEL

Запустить сбор сущность-описание:
```bash
$ python ./src/data_preparation/wd_dump_to_es.py
```

#### 4. Обучение модели
Для обучения модели необходимо запустить команду ниже, все параметры обучения установлены внутри

```bash
$ python ./src/disambiguator/LearnModel.py
```

#### 5. Запуск инференса
Для запуска инференса необходимо запустить следующую команду

```bash
$ python ./src/vector_base_nel/vector_base_nel.py -s "текст пользователя"
```