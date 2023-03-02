from razdel import sentenize
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc

class EntityExtractor():
    def __init__(self, text: str):
        self.text = text
        self.sentences = self.get_sentences()

        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)

        self.entity_positions = []
        self.entity_text = []

    def _apply_tag_ner(self, sentence: str):
        doc = Doc(sentence)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        return doc

    def get_sentences(self):
        try:
            sentences = list(sentenize(self.text))
            sentences = [list(sentence)[2] for sentence in sentences]
        except Exception as e:
            print(e)
            exit(0)


        return sentences

    def get_entitys_positions(self):
        for i, sentence in enumerate(self.sentences):
            doc = self._apply_tag_ner(sentence)

            self.entity_positions.append([])
            for span in doc.spans:
                self.entity_positions[i].append([span.start, span.stop])

        return self.entity_positions

    def get_entitys_text(self):
        for i, sentence in enumerate(self.sentences):
            doc = self._apply_tag_ner(sentence)

            self.entity_text.append([])
            for span in doc.spans:
                self.entity_text[i].append(span.text)

        return self.entity_text


# text = "Косой косил косой"
# entity_extractor = EntityExtractor(text)
#
# print(entity_extractor.get_entitys_positions())
# print(entity_extractor.get_entitys_text())
# print(entity_extractor.get_sentences())
