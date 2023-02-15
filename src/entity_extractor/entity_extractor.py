import os
os.environ['CUDA_VISIBLE_DEVICES']='cpu'

from deeppavlov import build_model
from razdel import sentenize

text = "Алексей А.В. и Пётр М.Е пошли к реке. Никто не видел Артёма.. . Но все знали, что он рядом!"

sentences = list(sentenize(text))
sentences = [list(sentence)[2] for sentence in sentences]

ner_model = build_model('ner_collection3_bert', download=False)

sentences_count = len(sentences)
sentences_entities = ner_model(sentences)

