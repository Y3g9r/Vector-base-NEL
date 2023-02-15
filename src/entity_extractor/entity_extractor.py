from deeppavlov import evaluate_model

model = build_model('ner_collection3_bert', download=True)

ner_model(['Президент США Джо Байден из страха перенес подрыв "Северных потоков" с июня на сентябрь 2022 года, при этом многие участники этой операции считали ее безумной, заявил в интервью Berliner Zeitung известный американский журналист-расследователь Сеймур Херш'])