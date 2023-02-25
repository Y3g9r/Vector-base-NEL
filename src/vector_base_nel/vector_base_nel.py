from entity_extractor.entity_extractor import EntityExtractor
from candidates_selector.candidates_selector import CandidatesSelector
from disambiguator.disambiguator import make_predict
import argparse

def make_links(args):
    texts_list = args.string

    entity_extractor = EntityExtractor(texts_list)
    entitys_positions = entity_extractor.get_entitys_positions()
    entitys_text = entity_extractor.get_entitys_text()
    sentences = entity_extractor.get_sentences()

    candidate_selector = CandidatesSelector()
    definitions = candidate_selector.get_candidates(entitys_text)

    make_predict(sentences, entitys_positions, definitions, "cuda:0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--string", type=str, default=0,
                        help='Введите строку')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_links(args)