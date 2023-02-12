import datetime
from mediawiki import MediaWiki
import argparse
import elasticsearch
from elasticsearch import helpers
import concurrent.futures
import gzip
import json
import traceback
import time
from tqdm import tqdm

class WikiDataItem():

    def __init__(self, wikidata_json):

        self.type = wikidata_json["type"]
        self.id = wikidata_json["id"]
        self.labels = wikidata_json["labels"]
        self.descriptions = wikidata_json["descriptions"]
        self.aliases = wikidata_json["aliases"]
        self.sitelinks = None

        if self.type == "item":
            self.sitelinks = wikidata_json["sitelinks"]
        else:
            self.datatype = wikidata_json["datatype"]

        #self.lastrevid = wikidata_json["lastrevid"]

        self.language_label = None
        self.language = None



    def _set_language(self, language):
        self.language = language

    def _set_any_language_label(self, priorities=["ru", "en", "fr", "de", "es", "it"]):
        for lang in priorities:
            if lang in self.labels:
                self.language_label = self.labels[lang]["value"]
                self._set_language(lang)
                return self.labels[lang]["value"]
        try:
            lang = list(self.labels.keys())[0]
        except:
            return

        self.language_label = self.labels[lang]["value"]
        self._set_language(lang)


    def __str__(self):
        return str(self.labels) + "\n" + str(self.descriptions)



class ElasticDumper:

    def __init__(self, es_host, wikipedia_index, max_threads=100,max_threads=100, download_wiki=False):

        self.es = elasticsearch.Elasticsearch(es_host)
        self.max_threads = max_threads
        self.download_wiki = download_wiki

        if not self.es.indices.exists(wikipedia_index):
            self.es.indices.create(wikipedia_index)

        self.wikipedia_index = wikipedia_index
        self.necessery_langs = ["ru", "en"]
        self.wikipriority = ["ru", "en", "fr", "de", "es", "it", "sv", "pl", "nl"]

    def write_lines(self, items_list):
        actions = self._prepare_actions(items_list)
        for i in actions:
            if not i:
                print("NONE")
        helpers.bulk(self.es, actions)

    def _prepare_actions(self, items_list):

        if len(items_list) > self.max_threads:
            num_threads = self.max_threads
        else:
            num_threads = len(items_list)

        futures = []
        actions = []
        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            for item in items_list:
                futures.append(executor.submit(self.__prepare_action, item=item))
        for future in futures:
            try:
                res = future.result()
                if res:
                    actions.append(res)
            except Exception:
                print(traceback.format_exc())
                pass
        return actions

    def __prepare_action(self, item):
        action = {
            "_index": "wikipedia-unknown-lang",
            "_type": item.type,
            "_id": item.id,
            "_source": {
                "timestamp": datetime.datetime.now()
            }
        }

        if item.labels:
            labels = {item.labels[l]["language"]: item.labels[l]["value"] for l in item.labels if
                      item.labels[l]["language"] in self.necessery_langs}
            action["_source"]["labels"] = labels

        if item.descriptions:
            descriptions = {item.descriptions[l]["language"]: item.descriptions[l]["value"] for l in item.descriptions
                            if item.descriptions[l]["language"] in self.necessery_langs}
            action["_source"]["descriptions"] = descriptions
        if item.aliases:
            aliases = {}
            for alias_lang in item.aliases:
                if alias_lang in self.necessery_langs:
                    aliases[alias_lang] = [kv["value"] for kv in item.aliases[alias_lang]]
            if aliases:
                action["_source"]["aliases"] = aliases

        item._set_any_language_label(self.wikipriority)

        action["_source"]["primary_language"] = item.language
        action["_source"]["language_label"] = item.language_label
        if item.language:
            action["_index"] = f"wikipedia-{item.language}"

        if self.download_wiki:
            try:
                wiki = MediaWiki(lang=item.language)
                title, lang = self.__get_good_sitelink(item.sitelinks, item.language,
                                                       supported_langs=list(wiki.supported_languages.keys()))
                if title:
                    action["_index"] = f"wikipedia-{lang}"
                    wiki.set_api_url(lang=lang)
                    page = wiki.page(title)

                    summary = page.summary
                    content = page.content
                    url = page.url

                    if url:
                        action["_source"]["url"] = url
                    if content:
                        action["_source"]["content"] = content
                    if summary:
                        action["_source"]["summary"] = summary


            except Exception as e:
                print(e)
                pass
                # print(item.language_label)
                # print(list(item.sitelinks.keys()))
                # print(traceback.format_exc())

        return action

    def __get_good_sitelink(self, sitelinks, lang="ru", supported_langs=["ru", "en"]):
        wiki_lang = wiki_lang = f"{lang}wiki"

        if wiki_lang in sitelinks:
            return sitelinks[wiki_lang]["title"], lang

        for lang in self.necessery_langs:
            wiki_lang = f"{lang}wiki"
            if wiki_lang in sitelinks:
                return sitelinks[wiki_lang]["title"], lang

        for lang in self.wikipriority:
            wiki_lang = f"{lang}wiki"
            if wiki_lang in sitelinks:
                return sitelinks[wiki_lang]["title"], lang

        for lang in supported_langs:
            wiki_lang = f"{lang}wiki"
            if wiki_lang in sitelinks:
                return sitelinks[wiki_lang]["title"], lang

        return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='Input bz2 file', type=str, default="latest-all.json.bz2")
    parser.add_argument('-s', "--start-line", type=int, default=0,
                        help='Начать с этой строчки')
    parser.add_argument('-e', "--end-line", type=int, help="Закончить на этой строчке",
                        default=100000000000000000)

    return parser.parse_args()


def dump(args):
    i = 0
    batch_size = 10000
    start_line = args.start_line
    end_line = args.end_line
    tasks = []
    addr = "192.168.102.129"
    dumper = ElasticDumper(addr, "wikipedia")
    with gzip.open(args.input, "r") as bz2_file:
        for line in tqdm(bz2_file):
            if i < start_line - 1:
                i += 1
                continue
            if line[0] == '[' or line[0] == ']':
                continue
            line = line.strip()[:-1]
            line = line.decode('utf8')
            try:
                line = json.loads(line)
                line = WikiDataItem(line)
            except:
                continue
            if line.type == "item":
                tasks.append(line)
            if len(tasks) >= batch_size:
                try:
                    dumper.write_lines(tasks)

                    tasks = []
                except Exception as e:
                    # print("ERROR")
                    print(str(e))
                    # print(traceback.format_exc())


if __name__ == "__main__":
    args = parse_args()
    dump(args)
