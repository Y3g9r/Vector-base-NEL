import pymorphy2

class CandidatesSelector():

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer(lang='ru')


    def _entity_normalizer(self, entity: str):
        return _morph.parse(entity)[0].normal_form
