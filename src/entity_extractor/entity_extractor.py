from razdel import sentenize
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc

#self.text = "Алексей А.В. из Артёмовска и Пётр М.Е из Магадана пошли к реке. Никто не видел Артёма.. . Но все знали, что он рядом!"


class EntityExtractor():
    def __init__(text: str):
        self.text = text
        self.sentences = list(sentenize(text))
        self.sentences = [list(sentence)[2] for sentence in sentences]

        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(emb)

        self.entity_positions = {}
        self.entity_text = {}

    def _apply_tag_ner(self, sentece: str):
        doc = Doc(sentence)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        return doc

    def get_entitys_positions(self):
        for sentence in self.sentences:
            doc = self._apply_tag_ner(sentence)

            self.entity_positions[sentences.index[sentence]] = []
            for span in doc.spans:
                entity_positions.append([span.start, span.stop])

        return entity_positions

    def get_entitys_text(self):
        for sentence in self.sentences:
            doc = self._apply_tag_ner(sentence)

            self.entity_text[sentences.index[sentence]] = []
            for span in doc.spans:
                entity_text.append(span.text)

        return entity_text


print(doc.spans[0])
