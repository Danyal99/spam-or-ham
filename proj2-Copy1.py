#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize Otter
import otter
grader = otter.Notebook()


# # Project 2: Spam/Ham Classification
# ## Feature Engineering, Logistic Regression, Cross Validation
# ## Due Date: Monday 11/30, 11:59 PM PST
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In this project, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this project, you should feel comfortable with the following:
# 
# - Feature engineering with text data
# - Using `sklearn` libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# ## Warning
# This is a **real world** dataset– the emails you are trying to classify are actual spam and legitimate emails. As a result, some of the spam emails may be in poor taste or be considered inappropriate. We think the benefit of working with realistic data outweighs these innapropriate emails, and wanted to give a warning at the beginning of the project so that you are made aware.

# ## Disclaimer about `sns.distplot()`
# 
# This project was designed for a slightly older version of seaborn, which does not support the new `displot` method taught in Lecture 9. Instead, in this project will occasionally call `distplot` (with a `t`). As you may have noticed in several of the previous assignments, use of the `distplot` function triggers a deprecation warning to notify the user that they should replace all deprecated functions with the updated version. Generally, warnings should not be suppressed but we will do so in this assignment to avoid cluttering.
# 
# See the seaborn documentation on [distributions](https://seaborn.pydata.org/tutorial/distributions.html) and [functions](https://seaborn.pydata.org/tutorial/function_overview.html) for more details.

# In[2]:


# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 1
# 1c | 2
# 2 | 3
# 3a | 2
# 3b | 2
# 4 | 2
# 5 | 2
# 6a | 1
# 6b | 1
# 6c | 2
# 6d | 2
# 6e | 1
# 6f | 3
# 7 | 6
# 8 | 6
# 9 | 3
# 10 | 15
# Total | 55

# # Part I - Initial Analysis

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the unlabeled test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to the autograder for evaluation.

# In[4]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[5]:


original_training_data['subject'] = original_training_data['subject'].fillna('')

display(original_training_data.isna().sum())


# In[6]:


grader.check("q1a")


# ### Question 1b
# 
# In the cell below, print the text of the `email` field for the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[7]:


first_ham = original_training_data[original_training_data['spam'] == 0].iloc[0]['email']
first_spam = original_training_data[original_training_data['spam'] == 1].iloc[0]['email']
print(first_ham)
print(first_spam)


# In[8]:


grader.check("q1b")


# <!-- BEGIN QUESTION -->
# 
# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->

# _The spam email still has all of its HTML tags in the email, while the ham email does not._

# <!-- END QUESTION -->
# 
# 
# 
# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and **validating** the models that we train.  We therefore need to split the training data into separate training and validation datsets.  You will need this **validation data** to assess the performance of your classifier once you are finished training. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[9]:


# This creates a 90/10 train-validation split on our labeled data

from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones as you try to increase the accuracy of your model.

# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[10]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array = np.zeros(shape=(len(texts), len(words)))
    for i in range(len(words)):
        contains = texts.str.contains(words[i])
        for x in range(len(indicator_array)):
            indicator_array[x][i] = contains.iloc[x]
    return indicator_array.astype(int)


# In[11]:


grader.check("q2")


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](images/training_conditional_proportions.png)
# 
# You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[12]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has a `type` column and some columns corresponding to words. You can think of each row as a sentence, and the value of 1 or 0 indicates the number of occurences of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into entries in a variable column. Notice how `word_1` and `word_2` become entries in `variable`; their values are stored in the value column."))
display(df.melt("type"))


# <!-- BEGIN QUESTION -->
# 
# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# manual: True
# format: image
# points: 2
# -->

# In[13]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts
words = words_in_texts(['increase', 'sex', 'issues', 'border', 'click'], train['email'])
email_type = train['spam']
increase = [i[0] for i in words]
sex = [i[1] for i in words]
issues = [i[2] for i in words]
border = [i[3] for i in words]
click = [i[4] for i in words]

words_df = pd.DataFrame({
    'increase' : increase,
    'sex' : sex,
    'issues' : issues,
    'border' : border,
    'click' : click,
    'type' : email_type
})

melted = words_df.melt('type')
melted['value'] = melted['value'].replace(0, 'ham').replace(1, 'spam')
plt.figure(figsize=(20,10));
sns.barplot(x='type', y='variable', hue='value', data=melted, ci=None);
plt.title('Frequency of Words in Spam/Ham Emails');
plt.xlabel('proportion');
plt.ylabel('words');


# <!-- END QUESTION -->
# 
# 
# 
# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 

# <!-- BEGIN QUESTION -->
# 
# ### Question 3b
# 
# ![training conditional densities](images/training_conditional_densities.png)
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# manual: True
# format: image
# points: 2
# -->

# In[14]:


ham_lengths = train[train['spam'] == 0]['email'].str.len()
spam_lengths = train[train['spam'] == 1]['email'].str.len()

plt.figure(figsize=(20,10));

sns.distplot(a=ham_lengths, hist=None);
sns.distplot(a=spam_lengths, hist=None);

plt.legend(['Ham', 'Spam'])
plt.xlim(0, 50000);
plt.title('Distribution of Email Body Length');
plt.xlabel('Length of email body');
plt.ylabel('Distribution');

plt.savefig('training_conditional_densities.png')


# <!-- END QUESTION -->
# 
# 
# 
# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of only 0's and 1's). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[15]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[16]:


grader.check("q4")


# ### Question 5
# 
# Now that we have matrices, we can build a model with `scikit-learn`! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the model's training accuracy below. You should get an accuracy of around $0.75$
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[17]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

y_train_predicted = model.predict(X_train)

training_accuracy = np.sum(y_train_predicted == Y_train) / len(Y_train)
print("Training Accuracy: ", training_accuracy)


# In[18]:


grader.check("q5")


# ## Evaluating Classifiers

# That doesn't seem too shabby! But the classifier you made above isn't as good as the accuracy would make you believe. First, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure. Accuracy on the training set doesn't always translate to accuracy in the real world (on the test set). In future parts of this analysis, we will hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# To be clear, we label spam emails as 1 and ham emails as 0. These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The two graphics below may help you understand precision and recall visually:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# ### Question 6a
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below (feel free to hard code your answers for this part):
# 
# *Tests in Question 6 only check that you have assigned appropriate types of values to each response variable, but do not check that your answers are correct.*
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[19]:


Y_train_ham = Y_train[Y_train == 0]
Y_train_spam = Y_train[Y_train == 1]


# In[20]:


zero_predictor_fp = 0
zero_predictor_fn = len(Y_train_spam)
zero_predictor_fp, zero_predictor_fn


# In[21]:


grader.check("q6a")


# ### Question 6b
# 
# What is the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do **NOT** use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# -->

# In[22]:


zero_predictor_acc = np.count_nonzero(np.zeros(len(Y_train)) == Y_train) / len(Y_train)
zero_predictor_recall = 0.0
zero_predictor_acc, zero_predictor_recall


# In[23]:


grader.check("q6b")


# <!-- BEGIN QUESTION -->
# 
# ### Question 6c
# 
# Provide brief explanations of the results from 6a and 6b. Why do we observe each of these values (FP, FN, accuracy, recall)?
# 
# <!--
# BEGIN QUESTION
# name: q6c
# manual: True
# points: 2
# -->

# _Since our zero_predictor classifies everything as 0 (not spam), there is no way that we will classify anything as spam. This is why the number of FP's is 0. This is also why the recall rate is 0.0 (we can never correctly classify a spam email). However, for the number of FN's, we are able to classify all of the ham emails correctly, so the number of FN's is the same as the total number of ham emails in the training set. Lastly, the accuracy rate is very close to the accuracy of the logistic regression model. This is because the training set has a larger proporton of ham emails than spam emails. Since the classifier is binary, we are able to correctly classify all of the ham emails which gives the zero_predictor a decently high accuracy._

# <!-- END QUESTION -->
# 
# ### Question 6d
# 
# Compute the precision, recall, and false-alarm rate of the `LogisticRegression` classifier created and trained in Question 5. Do **NOT** use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6d
# points: 2
# -->

# In[24]:


predicted_df = pd.DataFrame()
predicted_df['predicted'] = y_train_predicted
predicted_df['actual'] = train['spam']
predicted_df


# In[25]:


predicted_spam = predicted_df[predicted_df['predicted'] == 1]
og_spam = predicted_df[predicted_df['actual'] == 1]
og_ham = predicted_df[predicted_df['actual'] == 0]

logistic_predictor_precision = np.count_nonzero(predicted_spam['predicted'] == predicted_spam['actual']) / len(predicted_spam)
logistic_predictor_recall = np.count_nonzero(og_spam['predicted'] == og_spam['actual']) / len(og_spam)
logistic_predictor_far = np.count_nonzero(og_ham['predicted'] != og_ham['actual']) / len(og_ham)


# In[26]:


grader.check("q6d")


# <!-- BEGIN QUESTION -->
# 
# ### Question 6e
# 
# Are there more false positives or false negatives when using the logistic regression classifier from Question 5?
# 
# <!--
# BEGIN QUESTION
# name: q6e
# manual: True
# points: 1
# -->

# In[27]:


#false positives
display(np.count_nonzero(og_ham['predicted'] != og_ham['actual']))

#false negatives
display(np.count_nonzero(og_spam['predicted'] != og_spam['actual']))


# _There are more false negatives than false positives in the logistic regression classifier._

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ### Question 6f
# 
# 1. Our logistic regression classifier got 75.76% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# 1. Given the word features we gave you above, name one reason this classifier is performing poorly. Hint: Think about how prevalent these words are in the email set.
# 1. Which of these two classifiers would you prefer for a spam filter and why? Describe your reasoning and relate it to at least one of the evaluation metrics you have computed so far.
# 
# <!--
# BEGIN QUESTION
# name: q6f
# manual: True
# points: 3
# -->

# _1. This level of accuracy is only slightly more accurate than predicting 0 for every email. 2. The words that we have selected are not super prevalent in either spam or ham emails. This means that spam emails that do not have these words in very high proportions, so they are often being mislabeled as ham simply because these words are not common. 3. The logistic regression classifier will work better as a spam filter. The number of false negatives for the zero_predictor is higher than the number of false negatives for the logistic regression classifier. This is because the zero_predictor incorrectly classifies all of the spam emails, while the logistic regression classifier gets some of them correct._

# <!-- END QUESTION -->
# 
# 
# 
# # Part II - Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Gradescope.
# 
# **Gradescope limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on 70% of the test set when submitting to Gradescope, but we will be evaluating your model on the entire test set so try to score slightly above 88% on gradescope if you can.
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!'s were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You may want to use your **validation data** to evaluate your model and get a better sense of how it will perform on the test set.* Note, however, that you may overfit to your validation set if you try to optimize your validation accuracy too much.
# 
# ---

# <!-- BEGIN QUESTION -->
# 
# ### Question 7: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked or didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q7
# manual: True
# points: 6
# -->

# _1. Aside from my visualization in question 8, I also decided to look at the 300 most frequently occuring words in spam emails in the training set. I used regex to extract individual words, and then I sorted them by how often they appear. I then used these words as features and reused the words_in_text function from earlier._
# 
# _2. At first, I tried digging through the emails, looking for individual words that would improve my model's accuracy. Although this worked for the first 5-6 words, it quickly became difficult to manually find new words that would improve accuracy. With my current method, I was easily able to find a large amount of words that are indicative of spam emails._
# 
# _3. I was surprised to see that there were a large amount of spam emails with very similar features to ham emails. For example, at one point I calculated the proportion of non-word characters in emails, and the proportions were relatively similar across spam and ham emails. Additionally, as seen in my visualization from Question 8, there is significant overlap in the proportion of exclamation marks and white spaces. I expected spam and ham emails to be more obviously distinct._

# <!-- END QUESTION -->
# 
# 
# 
# ### Question 8: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. 
# 
# Include:
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel free to create as many plots as you want in your process of feature selection, but select only one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, **as long as it comes with thoughtful commentary.** Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# <!-- BEGIN QUESTION -->
# 
# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q8
# manual: True
# format: image
# points: 6
# -->

# In[28]:


# Write your description (2-3 sentences) as a comment here:

# This scatterplot is plotting the relationship between the proportion of white spaces and proportion of exclamation
# marks in email subjects. As the plot shows, there seems to be positive correlation between these two variables 
# among spam emails, while most of the ham email points are clustered together. I used this visualization to build 
# a feature that takes this relationship into account since there are areas of the plot without any ham points.

# Write the code to generate your visualization here:
email_lengths = train['subject'].str.len()
exclamations = train['subject'].str.extractall(r"([\W])").groupby(level=0)[0].apply(list).str.len() / email_lengths
white_space = train['subject'].str.extractall(r"(\s)").groupby(level=0)[0].apply(list).str.len() / email_lengths

chars_df = pd.DataFrame({
    'exclamations' : exclamations,
    'white_space' : white_space,
    'spam' : train['spam']
})


plt.figure(figsize=(20,10));
sns.scatterplot(x='exclamations', y='white_space', data=chars_df, hue='spam')
plt.xlabel('Proportion of Exclamation Marks');
plt.ylabel('Proportion of White Spaces');
plt.title('Proportion of Exclamation Marks vs Proportion of White Spaces in Subjects of Emails');


# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ### Question 9: ROC Curve
# 
# In most cases we won't be able to get 0 false positives and 0 false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover that they have cancer until it's too late, whereas a patient can just receive another screening for a false positive.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The ROC curve shows this trade off for each possible cutoff probability. In the cell below, plot a ROC curve for your final classifier (the one you use to make predictions for Gradescope) on the training data. Refer to Lecture 19 or [Section 17.7](https://www.textbook.ds100.org/ch/17/classification_sensitivity_specificity.html) of the course text to see how to plot an ROC curve.
# 
# <!--
# BEGIN QUESTION
# name: q9
# manual: True
# points: 3
# -->

# In[37]:


from sklearn.metrics import roc_curve

# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes

words_list_model_probabilities = new_model.predict_proba(new_X_train)[:, 1]
false_positive_rate_values, sensitivity_values, thresholds = roc_curve(train['spam'], words_list_model_probabilities, pos_label=1)

plt.figure(figsize=(20,10));
plt.plot(false_positive_rate_values, sensitivity_values)
plt.xlabel('False Positive Rate (1 - Specificity)');
plt.ylabel('Sensitivity');
plt.title('words_list_model ROC Curve');


# <!-- END QUESTION -->
# 
# # Question 10: Test Predictions
# 
# The following code will write your predictions on the test dataset to a CSV file. **You will need to submit this file to the "Project 2 Test Predictions" assignment on Gradescope to get credit for this question.**
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. **Please make sure you've saved your predictions to `test_predictions` as this is how part of your score for this question will be determined.**
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# **Note: You may submit up to 4 times a day. If you have submitted 4 times on a day, you will need to wait until the next day for more submissions.**
# 
# Note that this question is graded on an absolute scale based on the accuracy your model achieves on the overall test set, and as such, your score does not depend on your ranking on Gradescope. Your public Gradescope results are based off of your classifier's accuracy on 70% of the test dataset and your score for this question will be based off of your classifier's accuracy on 100% of the test set.
# 
# *The provided tests check that your predictions are in the correct format, but you must additionally submit to Gradescope to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q10
# points: 3
# -->

# In[29]:


word_freq = train[train['spam'] == 1]['email'].str.extractall(r'(\w+)')
top_300 = pd.Series(word_freq[0].to_numpy()).value_counts().index[0:300]
top_300


# In[30]:


def pipeline(df):
    X = words_in_texts(top_300, df['email'])

    sub_lengths = df['subject'].str.len()
    df['exclamations'] = df['subject'].str.extractall(r"([\W])").groupby(level=0)[0].apply(list).str.len() / sub_lengths
    df['white_space'] = df['subject'].str.extractall(r"(\s)").groupby(level=0)[0].apply(list).str.len() / sub_lengths
    df['e_w_correlation'] = np.where(((df['white_space'] > 0.25) & (df['exclamations'] > 0.25)) 
                                     | ((df['white_space'] < 0.08) & (df['exclamations'] > 0.32)), 1, 0)
    
    X = np.insert(X, 0, df['e_w_correlation'].to_numpy(), axis=1)

    return X   


# In[31]:


#train the model
new_X_train = pipeline(train)
new_model = LogisticRegression(max_iter=1000)
new_model.fit(new_X_train, train['spam'])


# In[32]:


#training data accuracy

train_predictions = new_model.predict(new_X_train)
np.count_nonzero(train_predictions == train['spam']) / len(train)


# In[33]:


#validation data accuracy

val_predictions = new_model.predict(pipeline(val))
np.count_nonzero(val_predictions == val['spam']) / len(val)


# In[34]:


#test data predictions
test_predictions = new_model.predict(pipeline(test))


# In[35]:


grader.check("q10")


# The following cell generates a CSV file with your predictions. **You must submit this CSV file to the "Project 2 Test Predictions" assignment on Gradescope to get credit for this question.**

# In[73]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Gradescope for scoring.')


# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[38]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[39]:


# Save your notebook first, then run this cell to export your submission.
grader.export("proj2.ipynb")


#  
