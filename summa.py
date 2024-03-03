# import nltk
# text="i am happy.நான் மகிழ்ச்சியாக இல்லை"

# sentences = nltk.sent_tokenize(text)
#     # sentence_results = []
# for sentence in sentences:
#         sentence_result = {}
#         print(sentence)
#         # translated_sentence = sentence
#         # if language != 'en':
#         #     translated_sentence = translate_text(sentence, language, 'en')


# import nltk
# text="i am happy. நான் மகிழ்ச்சியாக இல்லை"
# sentences = nltk.sent_tokenize(text)
# print(sentences[0])
# # sentence_results = []
#     for sentence in sentences:
#         sentence_result = {}
#         translated_sentence = sentence
#         if language != 'en':
#             translated_sentence = translate_text(sentence, language, 'en')
import nltk
from langdetect import detect
from collections import defaultdict

text = "i am happy.\நான் மகிழ்ச்சியாக இல்லை"

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)

# Create a defaultdict to store sentences based on their language
sentences_by_language = defaultdict(list)

# Iterate through each sentence
for sentence in sentences:
    # Detect the language of the sentence
    lang = detect(sentence)
    # Add the sentence to the list corresponding to its language
    sentences_by_language[lang].append(sentence)

# Print sentences grouped by language
for lang, lang_sentences in sentences_by_language.items():
    print(f"Language: {lang}")
    print(" ".join(lang_sentences))
    print()


text = "i am happy.நான் மகிழ்ச்சியாக இல்லை"
sentences = text.split(".")
# Remove empty strings from the list
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
print(sentences[0])

