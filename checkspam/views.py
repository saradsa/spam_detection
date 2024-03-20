from django.shortcuts import HttpResponse, render, redirect
import re
from collections import defaultdict
import csv
import math

# Create your views here.
def index(request):
    return render(request, "checkspam/index.html")


class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.spam_word_count = defaultdict(int)
        self.ham_word_count = defaultdict(int)
        self.spam_total = 0
        self.ham_total = 0
        self.alpha = 1

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        return text.split()
    
    def train(self, emails, labels):
        for email, label in zip(emails, labels):
            words = self.preprocess(email)
            for word in words:
                self.vocab.add(word)
                if label == 1:
                    self.spam_word_count[word] += 1  # Incrementing the specific word count for spam
                    self.spam_total += 1
                else:
                    self.ham_word_count[word] += 1  # Incrementing the specific word count for ham
                    self.ham_total += 1

    
    def predict(self, email):
        words = self.preprocess(email)
        spam_prob = self.calculate_probability(words, self.spam_word_count, self.spam_total, self.ham_total)
        ham_prob = self.calculate_probability(words, self.ham_word_count, self.ham_total, self.spam_total)
        return 1 if spam_prob > ham_prob else 0
    
    def calculate_probability(self, words, word_count, total, other_total):
        prob = 0
        vocab_size = len(self.vocab)
        for word in words:
            word_count_with_smoothing = word_count[word] + self.alpha if word in word_count else self.alpha
            total_with_smoothing = total + self.alpha * vocab_size
            prob += math.log(word_count_with_smoothing / total_with_smoothing)
        prob += math.log(total / (total + other_total))
        return prob
    
emails = []
labels = []

def read_csv(file_path):
    emails = []
    labels = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(1 if row['Category'] == 'spam' else 0)
            emails.append(row['Message'])

    return emails, labels







def checkspam(request):
    if request.method == 'POST':
        email_text = request.POST.get('inputmail')

        #Read csv data
        emails, labels = read_csv('phisingmail.csv')
        

        classifier = NaiveBayesClassifier()
        classifier.train(emails, labels)

        prediction = classifier.predict(email_text)
        if prediction == 1:
            result = 'spam'
        else:
            result = 'ham'

        context = {'spamham': result}

        return render(request, "checkspam/display.html", context)