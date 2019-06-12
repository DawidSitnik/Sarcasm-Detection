import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
#learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#reading file
file = 'headlines.json'
data = pd.read_json(file, lines=True)

#basic informations
print('\n')
print(data.info())
print('\n')
print(data.describe())
print('\n')

#------------- VISUALIZATION ---------------
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(background_color='white', stopwords = STOPWORDS,
                max_words = 200, max_font_size = 100,
                random_state = 17, width=800, height=400)

plt.figure(figsize=(16, 12))
wordcloud.generate(str(data.loc[data['is_sarcastic'] == 1, 'headline']))
plt.imshow(wordcloud);
plt.title("The Most Frequent Sarcastic Words")

plt.figure(figsize=(16, 12))
wordcloud.generate(str(data.loc[data['is_sarcastic'] == 0, 'headline']))
plt.imshow(wordcloud);
plt.title("The Most Frequent Non-Sarcastic Words")

plt.figure(figsize=(16, 12))
data['is_sarcastic'].value_counts().plot(kind='bar')
plt.ylabel('amount')
plt.xlabel('category')

plt.figure(figsize=(16, 12))
data.loc[data['is_sarcastic'] == 1, 'headline'].str.len().apply(np.log1p).hist(label='sarcastic', alpha=.5)
data.loc[data['is_sarcastic'] == 0, 'headline'].str.len().apply(np.log1p).hist(label='not sarcastic', alpha=.5)
plt.title("Length of Headlines")
plt.legend();

# plotting most frequent words in sarcastic and not sarcastic
nonSarcastic = data.loc[data['is_sarcastic'] == 0 ]
sarcastic = data.loc[data['is_sarcastic'] == 1]

plt.figure(figsize=(16, 12))
nonSarcasticBar = pd.Series(' '.join(nonSarcastic['headline']).lower().split()).value_counts()[:10].plot(kind='bar',
                            title="Most Frequent Words of not Sarcastic Comments")
nonSarcasticBar.set_xlabel("Word")
nonSarcasticBar.set_ylabel("Count")

plt.figure(figsize=(16, 12))
sarcasticBar = pd.Series(' '.join(sarcastic['headline']).lower().split()).value_counts()[:10].plot(kind='bar',
                            title="Most Frequent Words of Sarcastic Comments")
sarcasticBar.set_xlabel("Word")
sarcasticBar.set_ylabel("Count")
#------------- END OF VISUALIZATION ---------------

xTrain, xTest, yTrain, yTest = train_test_split(data['headline'], data['is_sarcastic'], random_state=0)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

#defining models
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', random_state=17, max_iter=1000,)
logit_RandomForestClassifier = RandomForestClassifier(n_estimators = 31, random_state = 32)
logit_SVC = SVC(kernel = 'sigmoid', gamma = 1.0)
logit_GBoost = GradientBoostingClassifier( n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   random_state =5)

logit_Mnb = MultinomialNB(alpha = .2)

# pipeline
tfidf_logit_pipeline = Pipeline([('tfidf', tfidf),
                                ('logit', logit)])
tfidf_logit_pipeline_RandomForestClassifier = Pipeline([('tfidf', tfidf), ('logit', logit_RandomForestClassifier)])
tfidf_logit_pipeline_SVC = Pipeline([('tfidf', tfidf), ('logit', logit_SVC)])
tfidf_logit_pipeline_GBoost = Pipeline([('tfidf', tfidf), ('logit', logit_GBoost)])
tfidf_logit_pipeline_Mnb = Pipeline([('tfidf', tfidf), ('logit', logit_Mnb)])

#fitting models
tfidf_logit_pipeline.fit(xTrain, yTrain)
# tfidf_logit_pipeline_RandomForestClassifier.fit(xTrain, yTrain)
# tfidf_logit_pipeline_SVC.fit(xTrain, yTrain)
# tfidf_logit_pipeline_GBoost.fit(xTrain, yTrain)
# tfidf_logit_pipeline_Mnb.fit(xTrain, yTrain)

#counting predictions for models
valid_pred = tfidf_logit_pipeline.predict(xTest)
# valid_pred_RandomForestClassifier = tfidf_logit_pipeline_RandomForestClassifier.predict(xTest)
# valid_pred_SVC = tfidf_logit_pipeline_SVC.predict(xTest)
# valid_pred_GBoost = tfidf_logit_pipeline_GBoost.predict(xTest)
# valid_pred_Mnb = tfidf_logit_pipeline_Mnb.predict(xTest)

#printing accuracy of models
print('Accuracy of Logistic Regression classifier on test set: {:.3f}'.format(accuracy_score(yTest, valid_pred)))
# print('Accuracy of Random Forest Classifier classifier on test set: {:.3f}'.format(accuracy_score(yTest, valid_pred_RandomForestClassifier)))
# print('Accuracy of SVC classifier on test set: {:.3f}'.format(accuracy_score(yTest, valid_pred_SVC)))
# print('Accuracy of GBoost classifier on test set: {:.3f}'.format(accuracy_score(yTest, valid_pred_GBoost)))
# print('Accuracy of Mnb classifier on test set: {:.3f}'.format(accuracy_score(yTest, valid_pred_Mnb)))


# save the model to disk
from joblib import dump
dump(tfidf_logit_pipeline, './models/LinearRegression-model.joblib')
# dump(tfidf_logit_pipeline_RandomForestClassifier, './models/RandomForestClassifier-model.joblib')
# dump(tfidf_logit_pipeline_SVC, './models/SVC-model.joblib')
# dump(tfidf_logit_pipeline_GBoost, './models/GBoost-model.joblib')
# dump(tfidf_logit_pipeline_Mnb, './models/Mnb-model.joblib')


# cm_lrc = confusion_matrix(yTest,valid_pred_SVC)
# f, ax = plt.subplots(figsize =(5,5))
# sns.heatmap(cm_lrc,annot = True,linewidths=0.5,linecolor="gray",fmt = ".0f",ax=ax)
# plt.title('Confusion matrix of SVC')
# plt.ylabel('Predicted label')
# plt.xlabel('True label')

#showing weights
import eli5
print(eli5.format_as_text(eli5.explain_weights(tfidf_logit_pipeline)))

plt.show()
