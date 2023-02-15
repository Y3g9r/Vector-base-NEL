import pymorphy2
from entity_extractor import EntityExtractor
from static.static import IP


class CandidatesSelector():

    def __init__(self, es_host, candidates_limit: int = 5):
        self._morph = pymorphy2.MorphAnalyzer(lang='ru')
        self.candidates_limit = candidates_limit
        self.es = elasticsearch.Elasticsearch(es_host)

        self.index = "wikipedia-ru"
        self.type = "item"

    def _entity_normalizer(self, entity: str):
        return _morph.parse(entity)[0].normal_form

    def _get_candidates_from_elastic(self, entity: str):
        body = {
            "query": {
                "match": {
                    "labels.ru": entity
                }
            }
        }

        return self.es.search(index=self.index, doc_type=self.type, body=body, size = self.candidates_limit)


    def get_candidates(self, sentence_entitys_list: list):
        sentence_candidates = []

        for sentence_entitys in sentences_entitys_list:
            for entity in sentence_entitys:
                sentence_candidates = self._get_candidates_from_elastic(\
                    self._entity_normalizer(entity))
                print(sentence_candidates)
                # for i, clean_entity in enumerate(sentence_candidates):
                #     sentence_candidates[



sentence_entitys_list = [["Артёма",'Организация объединённых наций'],['МВД']]
candidate_selector = CandidatesSelector(IP)
candidate_selector.get_candidates(sentence_entitys_list)
