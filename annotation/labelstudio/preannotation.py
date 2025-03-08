import spacy
import json
from random import randint
from gliner import GLiNER
from label_studio_sdk.client import LabelStudio


def split_text_into_chunks(text):
    chunk_size = int(len(text)/3)
    chunk_1 = text[:chunk_size]
    chunk_2 = text[chunk_size:2*chunk_size]
    chunk_3 = text[2*chunk_size:]
    return [chunk_1, chunk_2, chunk_3], chunk_size


def gliner_template(start, end, label, score, text):
    result = {
        "id": f"{randint(100000, 999999)}",
        "from_name": "cause_span",
        "to_name": "text",
        "type": "labels",
        "readonly": False,
        "value": {
            "start": start,
            "end": end,
            "score": score,
            "text": text,
            "labels": [label]
        }
    }
    return result


def ner_template(start, end, label, text):
    result = {
        "id": f"{randint(100000, 999999)}",
        "from_name": "cause_span",
        "to_name": "text",
        "type": "labels",
        "readonly": False,
        "value": {
            "start": start,
            "end": end,
            "score": 0.50,
            "text": text,
            "labels": [
                "Inflation"
            ]
        }
    }
    return result


def pos_template(start, end, label, text):
    result = {
        "id": f"{randint(100000, 999999)}",
        "from_name": "cause_span",
        "to_name": "text",
        "type": "labels",
        "readonly": False,
        "value": {
            "start": start,
            "end": end,
            "score": 0.50,
            "text": text,
            "labels": [
                "Inflation"
            ]
        }
    }
    return result


if __name__ == "__main__":
    LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
    API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
    client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    li = client.projects.get(id=4).get_label_interface()
    labels = sorted(li.get_tag("cause_span").labels)
    threshold = 0.3

    # spacy ner
    nlp = spacy.load("en_core_web_sm")
    print("spacy loaded")
    # gliner
    model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1")
    print("gliner loaded")

    with open(f'./export/annotation_project_5.json', 'r') as f:
        annotations_json = json.load(f)

    target_label = ["GPE", "PERSON", "ORG"]
    ner_blacklist = ["END", "GMT"]
    pre_annotations_ner = []
    pre_annotations_pos = []
    pre_annotations_gliner = []
    task_id = []

    for i, annotation in enumerate(annotations_json):
        task_id.append(annotation["id"])
        text = annotation["data"]["text"]

        ## spacy tags
        doc = nlp(text)
        ner = []
        for ent in doc.ents:
            entity = [int(ent.start_char), int(ent.end_char), ent.text]
            if ent.label_ in target_label:
                ner.append(entity)
        pos = [(w.idx, w.idx + len(w.text), w.text) for w in doc if w.pos_ in ["VERB"]]
        ner_result = [ner_template(ent[0], ent[1], None, ent[2]) for ent in ner if ent[2] not in ner_blacklist]
        pos_result = [pos_template(ent[0], ent[1], None, ent[2]) for ent in pos]
        pre_annotations_ner.append(ner_result)
        pre_annotations_pos.append(pos_result)

        ## gliner tags
        chunks, chunk_size = split_text_into_chunks(text)

        gliner_result = []
        for i, chunk in enumerate(chunks):
            chunk_result = []
            entities = model.predict_entities(chunk, labels, threshold=threshold)
            for ent in entities:
                label = [ent['label']]
                if label:
                    tagged_entity = {"label": label[0], "score": ent['score'], "start": ent['start'] + i * chunk_size,
                                     "end": ent['end'] + i * chunk_size, "text": ent["text"]}
                    result = gliner_template(ent['start'] + i * chunk_size, ent['end'] + i * chunk_size, label[0],
                                             ent["score"], ent["text"])
                    chunk_result += [result]
                else:
                    chunk_result += []

            gliner_result += chunk_result
        pre_annotations_gliner.append(gliner_result)

    ner_predictions = dict(zip(task_id, pre_annotations_ner))
    pos_predictions = dict(zip(task_id, pre_annotations_pos))
    gliner_predictions = dict(zip(task_id, pre_annotations_gliner))
    with open('./export/gliner_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(gliner_predictions, f, indent=4)

    LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
    API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
    client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    for i, tid in enumerate(task_id):
        print(i)
        client.predictions.create(task=tid, result=pre_annotations_gliner[i], model_version="gliner")
        client.predictions.create(task=tid, result=pre_annotations_ner[i], model_version="ner")
        client.predictions.create(task=tid, result=pre_annotations_pos[i], model_version="pos")