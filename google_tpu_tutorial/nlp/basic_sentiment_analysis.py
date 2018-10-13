from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

if __name__ == "__main__":
    client = language.LanguageServiceClient()
    text = u'Hello, World!'
    document = types.Document(
            content=text,
            type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    print('Text: {}'.format(text))
    print('Sentiment: {} {}'.format(sentiment.score, sentiment.magnitude))
