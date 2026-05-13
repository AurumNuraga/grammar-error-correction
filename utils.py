import random
from datasets import Dataset
import json
import subprocess

def load_dataset_from_file(filepath):
    input_texts, target_texts = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            noisy = data['text']
            corrected = apply_edits(noisy, data['edits']) if data['edits'] else noisy
            input_texts.append(noisy)
            target_texts.append(corrected)
    return Dataset.from_dict({"input_text": input_texts, "target_text": target_texts})

def apply_edits(text, edits):
    edits_sorted = sorted(edits[0][1], key=lambda x: x[0], reverse=True)
    for start, end, replacement, _ in edits_sorted:
        if replacement is None:
            replacement = ""
        text = text[:start] + replacement + text[end:]
    return text

def read_missp_file(filepath):
    pairs = []
    current_correct = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            if word.startswith('$'):
                current_correct = word[1:] 
            elif current_correct:
                pairs.append((word, current_correct))
    return pairs

templates = [
    "Yesterday at school, we discussed {}.",
    "I tried {} for the first time today.",
    "My teacher asked me about {}.",
    "During lunch, we talked about {}.",
    "I saw someone using {} in the hallway.",
    "In class, we practiced using {}.",
    "After school, I thought about {}.",
    "We wrote down {} as part of our homework.",
    "I heard {} while chatting with friends.",
    "At the cafeteria, {} was a hot topic.",
    "My friend explained {} to me today.",
    "We used {} during our group project.",
    "I learned how to say {} correctly.",
    "Have you seen the word {} before?",
    "Many people confuse the spelling of {}.",
    "I found {} in the document.",
    "This sentence contains the word {}.",
    "Could you check the spelling of {}?",
    "The word {} looks unusual here.",
    "Sometimes {} is spelled incorrectly.",
    "Please verify if {} is right.",
    "He typed {} by mistake.",
    "Is {} the correct spelling?",
    "She wrote {} instead of the right word.",
    "I often misspell the word {}.",
    "The word {} is highlighted as an error."
]


templates1 = [
    "At school today, I heard {}, do you know what it means?",
    "During lunch, we talked about {}, it was interesting.",
    "My friend used {} when talking about homework.",
    "I just learned {} in class yesterday.",
    "In today's lesson, the teacher mentioned {}.",
    "When we were talking about food, I used {}.",
    "At the cafeteria, I heard someone say {}, it was funny.",
    "Have you ever heard {} at school?",
    "While eating together, I shared a story about {}.",
    "During group discussion, I used {}.",
    "When studying languages, {} comes up a lot.",
    "In language class, we learned the meaning of {}.",
    "During break time, my friends were talking about {}.",
    "Hey, have you heard the word {}? It's interesting.",
    "I came across the word {}, what do you think about it?",
    "Do you know what {} means?",
    "I was reading and saw the word {}.",
    "Let's talk about the word {}.",
    "The word {} caught my attention.",
    "Can you explain the meaning of {}, please?",
    "I've been wondering about the word {}.",
    "Is {} a common word around here?",
    "What's your take on the word {}? ",
    "I noticed {} in that sentence.",
    "The word {} is quite unique.",
    "Have you ever used the word {}? "
]

def make_sentence_pairs(pairs):
    sentence_pairs = []
    for typo_word, correct_word in pairs:
        template = random.choice(templates)
        input_sentence = template.format(typo_word)
        target_sentence = template.format(correct_word)
        sentence_pairs.append((input_sentence, target_sentence))
    return sentence_pairs

def make_sentence_pairs1(pairs):
    sentence_pairs = []
    for typo_word, correct_word in pairs:
        template = random.choice(templates1)
        input_sentence = template.format(typo_word)
        target_sentence = template.format(correct_word)
        sentence_pairs.append((input_sentence, target_sentence))
    return sentence_pairs

def wrap_text_by_words(text, width=80):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))