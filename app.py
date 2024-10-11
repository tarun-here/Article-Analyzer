from flask import Flask, render_template, request
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForTokenClassification
import torch

app = Flask(__name__)

summarizer_tokenizer = AutoTokenizer.from_pretrained("my_summarization_tokenizer")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("my_summarization_model")

ner_tokenizer = AutoTokenizer.from_pretrained("our_ner_model")
ner_model = AutoModelForTokenClassification.from_pretrained("our_ner_model")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def summarize(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer_model.generate(**inputs, num_beams=4, max_length=150, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def get_ner(text):
    truncated_text = text[:512]  

    inputs = ner_tokenizer(truncated_text, return_tensors="pt")

    with torch.no_grad():
        outputs = ner_model(**inputs)

    predicted_labels = torch.argmax(outputs.logits, dim=2)

    ner_results = []
    current_entity = None
    for token_id, label_id in enumerate(predicted_labels[0]):
        label = ner_model.config.id2label[label_id.item()]
        if label.startswith("B-"):  
            if current_entity:
                ner_results.append(current_entity)
            current_entity = {"word": ner_tokenizer.decode(truncated_text.input_ids[0][token_id]), "entity_group": label[2:]}  
        elif label.startswith("I-"):  
            if current_entity:
                current_word = ner_tokenizer.decode(truncated_text.input_ids[0][token_id])  
                if current_word.startswith("##"):
                    current_entity["word"] += current_word.replace("##", "")
                else:
                    ner_results.append(current_entity)
                    current_entity = {"word": current_word, "entity_group": label[2:]}
        else:  
            if current_entity:
                ner_results.append(current_entity)
                current_entity = None
    if current_entity:
        ner_results.append(current_entity)

    processed_ner_results = []
    prev_entity = None
    for entity in ner_results:
        if prev_entity and entity["entity_group"] == prev_entity["entity_group"] and \
           entity["word"].startswith("##"):
            prev_entity["word"] += entity["word"].replace("##", "")
        else:
            if prev_entity:
                processed_ner_results.append(prev_entity)
            prev_entity = entity
    if prev_entity:
        processed_ner_results.append(prev_entity)

    return processed_ner_results


def get_sentiment(text):
    inputs = sentiment_model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        logits = sentiment_model.model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()
    sentiment_result = {
        "label": sentiment_model.model.config.id2label[predicted_class_idx],
        "score": torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    }

    return sentiment_result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', show_summary=False)

    elif request.method == 'POST':
        url = request.form['url']
        analysis_type = request.form['analysis_type']
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title

        if analysis_type == "summary":
            summary = summarize(text)
            return render_template(
                'index.html',
                summary=summary,
                text=text,
                show_summary=True,
                title=title,
                analysis_type=analysis_type
            )

        elif analysis_type == "ner":
            ner_results = get_ner(text)
            return render_template(
                'index.html',
                text=text,
                show_summary=True,
                title=title,
                ner_results=ner_results,
                analysis_type=analysis_type
            )

        elif analysis_type == "sentiment":
            sentiment_result = get_sentiment(text)
            return render_template(
                'index.html',
                text=text,
                show_summary=True,
                title=title,
                sentiment_result=sentiment_result,
                analysis_type=analysis_type
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)