# import json,os,sys,re


from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier

newsTrainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = ["?!#%&"]))

# You need to train the system passing each text one by one to the trainer module.
newsSet =[
    {'symptoms': 'pain chest', 'disease': 'hypertensive disease'},
    {'symptoms': 'shortness of breath', 'disease': 'hypertensive disease'},
    {'symptoms': 'dizziness', 'disease': 'hypertensive disease'},
    {'symptoms': 'asthenia', 'disease': 'hypertensive disease'},
    {'symptoms': 'fall', 'disease': 'hypertensive disease'},
    {'symptoms': 'syncope', 'disease': 'hypertensive disease'},
    {'symptoms': 'vertigo', 'disease': 'hypertensive disease'},
    {'symptoms': 'sweat sweating increased', 'disease': 'hypertensive disease'},
    {'symptoms': 'palpitation', 'disease': 'hypertensive disease'},
    {'symptoms': 'nausea', 'disease': 'hypertensive disease'},
    {'symptoms': 'angina pectoris', 'disease': 'hypertensive disease'},
    {'symptoms': 'pressure chest', 'disease': 'hypertensive disease'},
    {'symptoms': 'polyuria', 'disease': 'diabetes'},
    {'symptoms': 'polydypsia', 'disease': 'diabetes'},
    {'symptoms': 'shortness of breath', 'disease': 'diabetes'},
    {'symptoms': 'asthenia', 'disease': 'diabetes'},
    {'symptoms': 'nausea', 'disease': 'diabetes'},
    {'symptoms': 'orthopnea', 'disease': 'diabetes'},
    {'symptoms': 'sweat sweating increased', 'disease': 'diabetes'},
    {'symptoms': 'unresponsiveness', 'disease': 'diabetes'},
    {'symptoms': 'mental status changes', 'disease': 'diabetes'},
    {'symptoms': 'vertigo', 'disease': 'diabetes'},
    {'symptoms': 'vomiting', 'disease': 'diabetes'},
    {'symptoms': 'labored breathing', 'disease': 'diabetes'},
    {'symptoms': 'feeling suicidal', 'disease': 'depressive disorder'},
    {'symptoms': 'suicidal', 'disease': 'depressive disorder'},
    {'symptoms': 'hallucinations auditory', 'disease': 'depressive disorder'},
    {'symptoms': 'feeling hopeless', 'disease': 'depressive disorder'},
    {'symptoms': 'weepiness', 'disease': 'depressive disorder'},
    {'symptoms': 'sleeplessness', 'disease': 'depressive disorder'},
    {'symptoms': 'motor retardation', 'disease': 'depressive disorder'},
    {'symptoms': 'irritable mood', 'disease': 'depressive disorder'},
    {'symptoms': 'blackout', 'disease': 'depressive disorder'},
    {'symptoms': 'mood depressed', 'disease': 'depressive disorder'},
    {'symptoms': 'hallucinations visual', 'disease': 'depressive disorder'},
    {'symptoms': 'worry', 'disease': 'depressive disorder'},
    {'symptoms': 'agitation', 'disease': 'depressive disorder'},
    {'symptoms': 'tremor', 'disease': 'depressive disorder'},
    {'symptoms': 'verbal auditory hallucinations', 'disease': 'depressive disorder'},
    {'symptoms': 'energy increased', 'disease': 'depressive disorder'},
    {'symptoms': 'difficulty', 'disease': 'depressive disorder'},
    {'symptoms': 'nightmare', 'disease': 'depressive disorder'},
    {'symptoms': 'unable to concentrate', 'disease': 'depressive disorder'},
    {'symptoms': 'homelessness', 'disease': 'depressive disorder'},
    {'symptoms': 'cough', 'disease': 'pneumonia'},
    {'symptoms': 'fever', 'disease': 'pneumonia'},
    {'symptoms': 'decreased translucency', 'disease': 'pneumonia'},
    {'symptoms': 'shortness of breath', 'disease': 'pneumonia'},
    {'symptoms': 'rale', 'disease': 'pneumonia'},
    {'symptoms': 'productive cough', 'disease': 'pneumonia'},
    {'symptoms': 'pleuritic pain', 'disease': 'pneumonia'},
    {'symptoms': 'yellow phlegm', 'disease': 'pneumonia'},
    {'symptoms': 'breath sounds decreased', 'disease': 'pneumonia'},
    {'symptoms': 'chill', 'disease': 'pneumonia'},
    {'symptoms': 'rhonchus', 'disease': 'pneumonia'},
    {'symptoms': 'green phlegm', 'disease': 'pneumonia'},
    {'symptoms': 'non-productive cough', 'disease': 'pneumonia'},
    {'symptoms': 'wheezing', 'disease': 'pneumonia'},
    {'symptoms': 'distress respiratory', 'disease': 'pneumonia'},
    {'symptoms': 'night sweat', 'disease': 'pneumonia'},
    {'symptoms': 'shortness of breath', 'disease': 'heart failure'},
    {'symptoms': 'orthopnea', 'disease': 'heart failure'},
    {'symptoms': 'jugular venous distention', 'disease': 'heart failure'},
    {'symptoms': 'rale', 'disease': 'heart failure'},
    {'symptoms': 'dyspnea', 'disease': 'heart failure'},
    {'symptoms': 'cough', 'disease': 'heart failure'},
    {'symptoms': 'wheezing', 'disease': 'heart failure'},
    {'symptoms': 'dysarthria', 'disease': 'heart failure'},
    {'symptoms': 'asthenia', 'disease': 'heart failure'},
    {'symptoms': 'numbness', 'disease': 'heart failure'},
    {'symptoms': 'fever', 'disease': 'infection'},
    {'symptoms': 'erythema', 'disease': 'infection'},
    {'symptoms': 'decreased translucency', 'disease': 'infection'},
    {'symptoms': 'hepatosplenomegaly', 'disease': 'infection'},
    {'symptoms': 'chill', 'disease': 'infection'},
    {'symptoms': 'pruritus', 'disease': 'infection'},
    {'symptoms': 'diarrhea', 'disease': 'infection'},
    {'symptoms': 'abscess bacterial', 'disease': 'infection'},
    {'symptoms': 'swelling', 'disease': 'infection'},
    {'symptoms': 'pain', 'disease': 'infection'},
    {'symptoms': 'apyrexial', 'disease': 'infection'},
    {'symptoms': 'fever', 'disease': 'infection'},
    {'symptoms': 'dysuria', 'disease': 'IUT'},
    {'symptoms': 'hematuria', 'disease': 'IUT'},
    {'symptoms': 'renal angle tenderness', 'disease': 'IUT'},
    {'symptoms': 'lethargy', 'disease': 'IUT'},
    {'symptoms': 'asthenia', 'disease': 'IUT'},
    {'symptoms': 'hyponatremia', 'disease': 'IUT'},
    {'symptoms': 'hemodynamically stable', 'disease': 'IUT'},
    {'symptoms': 'distress respiratory', 'disease': 'IUT'},
    {'symptoms': 'difficulty passing urine', 'disease': 'IUT'},
    {'symptoms': 'mental status changes', 'disease': 'IUT'},
    {'symptoms': 'consciousness clear', 'disease': 'IUT'},
    {'symptoms': 'chill', 'disease': 'anemia'},
    {'symptoms': 'guaiac positive', 'disease': 'anemia'},
    {'symptoms': 'monoclonal', 'disease': 'anemia'},
    {'symptoms': 'ecchymosis', 'disease': 'anemia'},
    {'symptoms': 'tumor cell invasion', 'disease': 'anemia'},
    {'symptoms': 'haemorrhage', 'disease': 'anemia'},
    {'symptoms': 'pallor', 'disease': 'anemia'},
    {'symptoms': 'asthenia', 'disease': 'anemia'},
    {'symptoms': 'fatigue', 'disease': 'anemia'},
    {'symptoms': 'heme positive', 'disease': 'anemia'},
    {'symptoms': 'back pain', 'disease': 'anemia'},
    {'symptoms': 'orthostasis', 'disease': 'anemia'},
    {'symptoms': 'hyponatremia', 'disease': 'anemia'},
    {'symptoms': 'dizziness', 'disease': 'anemia'},
    {'symptoms': 'shortness of breath', 'disease': 'anemia'},
    {'symptoms': 'pain', 'disease': 'anemia'},
    {'symptoms': 'rhonchus', 'disease': 'anemia'},
    {'symptoms': 'arthralgia', 'disease': 'anemia'},
    {'symptoms': 'swelling', 'disease': 'anemia'},
    {'symptoms': 'fever', 'disease': 'dementia'},
    {'symptoms': 'fall', 'disease': 'dementia'},
    {'symptoms': 'unresponsiveness', 'disease': 'dementia'},
    {'symptoms': 'lethargy', 'disease': 'dementia'},
    {'symptoms': 'agitation', 'disease': 'dementia'},
    {'symptoms': 'ecchymosis', 'disease': 'dementia'},
    {'symptoms': 'syncope', 'disease': 'dementia'},
    {'symptoms': 'rale', 'disease': 'dementia'},
    {'symptoms': 'unconscious state', 'disease': 'dementia'},
    {'symptoms': 'cough', 'disease': 'dementia'},
    {'symptoms': 'bedridden', 'disease': 'dementia'},
    {'symptoms': 'unsteady gait', 'disease': 'dementia'},
    

    ]



for news in newsSet:
    newsTrainer.train(news['symptoms'], news['disease'])

# When you have sufficient trained data, you are almost done and can start to use
# a classifier.
newsClassifier = Classifier(newsTrainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = ["?!#%&"]))

# Now you have a classifier which can give a try to classifiy text of news whose
# category is unknown, yet.
unknownInstance = "pain fever coughing"
classification = newsClassifier.classify(unknownInstance)

# the classification variable holds the possible categories sorted by 
# their probablity value
print(classification)