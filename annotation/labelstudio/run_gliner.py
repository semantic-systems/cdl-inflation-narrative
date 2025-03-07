import spacy
import json
from random import randint
from gliner import GLiNER
from label_studio_sdk.client import LabelStudio


def split_text_into_chunks(text, chunk_size=250, overlap=30):
    """
    Splits text into chunks of a given word length with overlap.
    Returns chunks along with their start positions in the original text (character index).
    """
    words = text.split()
    chunks = []
    start_positions = []  # Stores starting character index of each chunk in the original text

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        # Find the character-based start position of this chunk in the original text
        char_start = text.find(chunk, start_positions[-1] + 1 if start_positions else 0)

        chunks.append(chunk)
        start_positions.append(char_start)
        start = end - overlap  # Move forward with overlap

    return chunks, start_positions


def adjust_ner_indices(ner_results, start_positions):
    """
    Adjusts NER entity indices to match their positions in the full text.
    :param ner_results: List of NER results per chunk. Each result is a list of dicts with 'start' and 'end'.
    :param start_positions: List of starting character indices for each chunk.
    :return: Adjusted NER results with corrected indices.
    """
    adjusted_entities = []

    for chunk_idx, entities in enumerate(ner_results):
        char_offset = start_positions[chunk_idx]  # Get the chunk's starting character position

        for entity in entities:
            adjusted_entity = entity.copy()
            adjusted_entity["start"] += char_offset  # Adjust start index
            adjusted_entity["end"] += char_offset  # Adjust end index
            adjusted_entities.append(adjusted_entity)

    return adjusted_entities


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




LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
li = client.projects.get(id=4).get_label_interface()
labels = sorted(li.get_tag("cause_span").labels)
threshold = 0.3

# spacy ner
nlp = spacy.load("en_core_web_sm")

# gliner
model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1")

with open(f'./export/annotation_project_4.json', 'r') as f:
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
    # predictions = predictions_template(ner, pos)
    ner_result = [ner_template(ent[0], ent[1], None, ent[2]) for ent in ner if ent[2] not in ner_blacklist]
    pos_result = [pos_template(ent[0], ent[1], None, ent[2]) for ent in pos]
    pre_annotations_ner.append(ner_result)
    pre_annotations_pos.append(pos_result)

    ## gliner tags
    chunks, start_positions = split_text_into_chunks(text)
    print(chunks)
    print(start_positions)
    gliner_result = []
    for chunk in chunks:
        chunk_result = []
        entities = model.predict_entities(chunks, labels, threshold=threshold)
        for ent in entities:
            label = [ent['label']]
            if label:
                tagged_entity = {"label": label, "score": ent['score'], "start": ent['start'], "end": ent['end'],
                                 "text": ent["text"]}
                # Step 3: Adjust indices
                corrected_ner_results = adjust_ner_indices(tagged_entity, start_positions)
                result = [gliner_template(ent["start"], ent["end"], ent["label"], ent["score"], ent["text"]) for ent in
                          corrected_ner_results]
                chunk_result.append(result)

        gliner_result.append(chunk_result)
    pre_annotations_gliner.append(gliner_result)

print(len(pre_annotations_ner))
print(len(pre_annotations_pos))
print(len(task_id))
print(len(pre_annotations_gliner))

