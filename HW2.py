#!/usr/bin/env python
# coding: utf-8

# # CSCI 544 - Homework 2 - HMMs on part-of- speech tagging

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#reading the dataframe and renaming columns

df = pd.read_csv('data/train', sep='\t', on_bad_lines = 'skip')
df.columns = ['index', 'word', 'postag']
df.loc[-1] = [1, 'Pierre', 'NNP']  # adding a row
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True) 


# In[3]:


# setting threshold as 3

counts = df['word'].value_counts()
idx = counts[counts.lt(3)].index

df.loc[df['word'].isin(idx), 'word'] = '<unk>' 
df2 = df


# In[4]:


df2 = df


# In[5]:


df = df.drop(columns=['index'])


# In[6]:


# getting value counts of each unique word
df['occurrences'] = df.groupby(['word'])['postag'].transform('count')


# In[7]:


df = df.sort_values('occurrences', ascending = False)
df = df.drop(columns=['postag'])


# In[8]:


df = df.drop_duplicates(keep='first')


# In[9]:


df = df.reset_index()
df['index'] = df.index


# In[10]:


df = df[['word', 'index', 'occurrences']]


# In[11]:


df


# In[12]:


vocab_list = df['word'].to_list()


# In[13]:


# shifting results of unk on top

df=df.drop(3)
df.loc[-1] = ['<unk>',3, 32537]  # adding a row
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True) 


# In[14]:


df.to_csv("vocab.txt", sep='\t', index=False)


# In[15]:


df


# In[16]:


#saving the file in the desired format

df["modified"] = df.apply(lambda x: "\\t".join(map(str, x)), axis=1)

# Save the modified data to a .txt file
filename = "vocab.txt"
with open(filename, "w") as file:
    file.write("\n".join(df["modified"]))


# #### What is the selected threshold for unknown words replacement? = 3
# #### What is the total size of your vocabulary? = 16920 unique words
# #### What is the total occurrences of the special token ‘< unk >’ after replacement? = 32537
# 

# ## Task 2: HMM Model Learning 

# In[17]:


df2


# In[18]:


df2_dict = dict(zip(df2.word, df2.postag))


# In[19]:


# list of unique postags and words

words = list((df2["word"]).unique())
postags = list((df2["postag"]).unique())


# ### 2.1 Creating Transition Probability dictionary

# In[20]:


# create combinations including repeats

def postag_combos(list1, list2):
    combinations = []
    for i in list1:
        for j in list2:
            combinations.append((i, j))
    return combinations

postag_postag_combo = postag_combos(postags, postags)


# In[21]:


#create a list of tuples 
postag_dict = {key: 0 for key in postag_postag_combo}
postag_list = list(postag_dict)


# In[22]:


for i in range(1, len(df2)):
    pair = (df2.iloc[i-1]['postag'], df2.iloc[i]['postag'])
    if pair in postag_dict:
        postag_dict[pair] += 1
    


# In[23]:


#calculating transition probab and storing it in a dictionary

transition_probab = {}
for i in range(0, len(postag_list)):
    if postag_list[i] in postag_dict:  
        transition_probab[postag_list[i]] = postag_dict[postag_list[i]] / (df2['postag'] == postag_list[i][0]).sum()
        


# In[24]:


df2['postag'].nunique()


# In[25]:


sum(transition_probab.values())


# ### 2.2 Creating Emission Probability dictionary

# In[26]:


# create unique combinations of words and tags

postag_word_combo = [(x, y) for x in postags for y in words]
postag_word_dict = {key: 0 for key in postag_word_combo}


# In[27]:


# creating a dictionary of tuples and value counts

for i in range(0, len(df2)):
    pair = (df2.iloc[i]['postag'], df2.iloc[i]['word'])
    if pair in postag_word_dict:
        postag_word_dict[pair] += 1


# In[28]:


#value count of every POS tag in the dataset
postag_value_counts = df2['postag'].value_counts()

# convert value counts to a dictionary
value_counts_dict = postag_value_counts.to_dict()


# In[29]:


#calculating emission probab and storing it in a dictionary

emission_probab = {}
for postag1 in postag_word_dict:
    emission_probab[postag1] = postag_word_dict[postag1] / value_counts_dict[postag1[0]]


# In[30]:


sum(emission_probab.values())


# In[31]:


#saving the values in a json

import json 


tp = {str(k): v for k, v in transition_probab.items()}
ep = {str(k): v for k, v in emission_probab.items()}

with open('hmm.json', 'w') as f:
    json.dump(tp, f)
    json.dump(ep, f)


# # Task 3: Greedy Decoding Model

# ### Generate initial probabilities with formula: 
#     - total value counts of each postag / the total length of the dataset
# 

# In[32]:


#Create initial probabilities

initial_probability = df2['postag'].value_counts(ascending=False) / len(df2)
initial_probability = dict(initial_probability) #converting to dictionary
initial_probability = {k: v for k, v in sorted(initial_probability.items(), key=lambda item: list(df2['postag'].unique()).index(item[0]))}
initial_probability = list(initial_probability.values())


# In[33]:


# reading dev data

df_gd = pd.read_csv('data/dev', sep='\t', on_bad_lines = 'skip')
df_gd.columns = ['index', 'word', 'postag']
df_gd.loc[-1] = ['1', 'The', 'DT']  # adding a row
df_gd.index = df_gd.index + 1  # shifting index
df_gd.sort_index(inplace=True)


# In[34]:


df_gd


# In[35]:


len_postags = len(postags)
len_words = len(words)


# ### 3.1 Transition Probability Dataframe

# In[36]:


# converting Transition Probability dictionary to matrix 

trans_prob_matrix = [[0 for j in range(len(postags))] for i in range(len(postags))]

for i in range(len(postags)):
    for j in range(len(postags)):
        trans_prob_matrix[i][j] = transition_probab.get((postags[i], postags[j]), 0)


# In[37]:


#converting Transition Probability matrix to dataframe

transition_df = pd.DataFrame(trans_prob_matrix, columns = list(postags), index=list(postags))


# In[38]:


transition_df


# ### 3.2 Emission Probability Dataframe

# In[39]:


#converting Emission Probability dictionary to matrix 

emission_probab_matrix = [[0 for j in range(len(words))] for i in range(len(postags))]

for i in range(len(postags)):
    for j in range(len(words)):
        emission_probab_matrix[i][j] = emission_probab.get((postags[i],words[j]), 0)


# In[40]:


#converting Emission Probability matrix to dataframe 

emission_df = pd.DataFrame(emission_probab_matrix, columns = list(words), index=list(postags))


# In[41]:


emission_df


# In[42]:


def calculate_nonzero_probs(transition_df, emission_df):
    transition_count = (transition_df != 0).sum().sum()
    emission_count = (emission_df != 0).sum().sum()
    return transition_count, emission_count


# In[43]:


transition_count, emission_count = calculate_nonzero_probs(transition_df, emission_df)
print("Number of non-zero transition probabilities:", transition_count)
print("Number of non-zero emission probabilities:", emission_count)


# ### 3.3 Greedy Decoding - Dev

# In[44]:


# Greedy Decoding - dev

def greedy_decoder(dataframe):
    greedy_predicted_list = []
    first_word = 1 #setting it as true

    for i in dataframe.index:
        dev_word = dataframe['word'][i] #iterate over every word in the dataframe
        if(dev_word not in words): #check if word exists in vocabulary
            dev_word = "<unk>"   

        if(dev_word == "."): # check if a full stop is encountered
            first_word = 1
            greedy_predicted_list.append(dev_word)

        elif(first_word == 1): #check if it is first word
            first_word = 0 #after first word, set flag to 0
            inital_x_emission = initial_probability * emission_df[dev_word] # multiplying initial and emission probabilities
            max_probab = np.argmax(inital_x_emission) #get max probability out of the list
            greedy_predicted_list.append(postags[max_probab])     

        else: # check if it word after first word 
            emission_col = emission_df[dev_word]
            transition_col = transition_df.iloc[max_probab] 
            emission_x_transition = emission_col * transition_col # multiplying transition and emission probabilities
            max_probab = np.argmax(emission_x_transition) # get max probability
            greedy_predicted_list.append(postags[max_probab])
            
    return greedy_predicted_list


# In[45]:


dev_predicted_list = greedy_decoder(df_gd)


# In[46]:


accurate_words = 0

for i in df_gd.index:
    if df_gd['postag'][i] == dev_predicted_list[i]:
        accurate_words += 1
        
accuracy = accurate_words / len(df_gd)     

print("Accuracy of Greedy Decoding model:", accuracy * 100, "%")


# ### 3.4 Generating POS tags for test

# In[47]:


df_test = pd.read_csv('data/test', sep='\t', on_bad_lines = 'skip')
df_test.columns = ['index', 'word']
df_test.loc[-1] = ['1', 'Influential']  # adding a row
df_test.index = df_test.index + 1  # shifting index
df_test.sort_index(inplace=True)


# In[48]:


df_test


# In[49]:


greedy_test_predicted_list = greedy_decoder(df_test)


# In[50]:


df_test['postag'] = greedy_test_predicted_list


# In[51]:


df_test['word'] = np.where(df_test['word'].isin(vocab_list), df_test['word'], '<unk>')


# In[52]:


df_test


# In[53]:


df_test.to_csv('greedy.out', sep='\t', header=None, index=None)


# In[54]:


with open('greedy.out', 'w') as f:
    
    f.write("1" + "\t" + df_test.iloc[0]['word'] + "\t" + str(greedy_test_predicted_list[0]) + "\n" )
    
    count = 2
    for i in range(2, len(df_test)):
        if(df_test.iloc[i]['index'] == 1):
            count = 1
            f.write("\n")
        f.write(str(count) + "\t" + df_test.iloc[i]['word'] + "\t" + str(greedy_test_predicted_list[i]) + "\n" )
        count+=1


# ## Task 4: Viterbi Decoding with HMM 

# In[55]:


# read dev data

df_gd = pd.read_csv('data/dev', sep='\t', on_bad_lines = 'skip')
df_gd.columns = ['index', 'word', 'postag']
df_gd.loc[-1] = ['1', 'The', 'DT']  # adding a row
df_gd.index = df_gd.index + 1  # shifting index
df_gd.sort_index(inplace=True) 


# ### 4.1 Viterbi Decoding - Dev

# In[56]:


# Viterbi Decoding

def viterbi_decoder(dataframe):
    
    final_node_list = []
    first_word = 1
    viterbi_predicted_list = []

    for i in dataframe.index: 
        dev_word = dataframe['word'][i] #iterate
        if(dev_word not in words): #check if word exists in vocabulary
            dev_word = "<unk>"   

        if(first_word == 1):
            first_word  = 0
            prev_probabilities = [-1] * len(initial_probability) #setting all probabilities to -1 for first word
            initial_x_emission = initial_probability * emission_df[dev_word] # multiplying initial and emission probabilities
            prev_probabilities = list(initial_x_emission)
            prev_nodes = [] #creating a list of the indexes of the POS tags with the highest probabilty for that iteration 
            for index, val in enumerate(prev_probabilities):
                if val > 0:
                    prev_nodes.append(index) 
            final_node_list= []

        elif(dev_word != '.'): # check if its the word after first word
            final_probabilities = [-1] * len(initial_probability) #setting all probabilities to -1 for first word
            final_nodes = [-1] * len(initial_probability) #setting all indexes to -1 for first word

            emission_probability = list(emission_df[dev_word])

            for k in prev_nodes: # iterate over the indexes of POS tags

                transition_list = list(transition_df.iloc[k])
                transition_probability = [(itr * prev_probabilities[k]) for itr in transition_list]
                curr_probability = [a*b for a,b in zip(transition_probability, emission_probability)]

                for m in range(len(initial_probability)):
                    if curr_probability[m] > final_probabilities[m]: # if the current probability is greater than the previous probability
                        final_probabilities[m] = curr_probability[m] # replace the specific probability with the new probability
                        final_nodes[m] = k # same for nodes or index of POS tag
            final_node_list.append(final_nodes)
            prev_probabilities = final_probabilities

            prev_nodes = []
            for index, val in enumerate(prev_probabilities):
                if val > 0:
                    prev_nodes.append(index)  
        else: # check if its a full stop
            first_word = 1
            final_probabilities = [-1] * len(initial_probability)
            final_nodes = [-1] * len(initial_probability)
            emission_probability = list(emission_df[dev_word])

            for k in prev_nodes: 
                transition_list = list(transition_df.iloc[:,k])
                transition_probability = [(itr * prev_probabilities[k]) for itr in transition_list]
                curr_probability = [a*b for a,b in zip(transition_probability, emission_probability)] # multiply item by item

                for m in range(len(initial_probability)):
                    if curr_probability[m] > final_probabilities[m]:
                        final_probabilities[m] = curr_probability[m] 
                        final_nodes[m] = k

            final_node_list.append(final_nodes)
            prev_probabilities = final_probabilities

            prev_nodes = []

            for index, val in enumerate(prev_probabilities):
                if val > 0:
                    prev_nodes.append(index) 

            tags_for_this_sentence = []
            first_word  = 1

            tag_for_last_word = np.argmax(prev_probabilities)
            tags_for_this_sentence.append(postags[tag_for_last_word])
            length_sentence = len(final_node_list) - 1

            while(length_sentence >= 0 ): # tracing back nodes to get the highest probability for the sentence
                current_postag = final_node_list[length_sentence][tag_for_last_word]
                length_sentence = length_sentence - 1
                tags_for_this_sentence.append(postags[current_postag])
                tag_for_last_word = current_postag
            tags_for_this_sentence.reverse()

            viterbi_predicted_list.append(tags_for_this_sentence)

    viterbi_predicted_list =  [item for sub_list in viterbi_predicted_list for item in sub_list]
    
    return viterbi_predicted_list


# ### 4.2 Testing on Dev dataset

# In[57]:


#generate list of predicted tags

dev_viterbi_predicted_list = viterbi_decoder(df_gd)


# In[58]:


accurate_words = 0

for i in range(len(dev_viterbi_predicted_list)):
    if df_gd['postag'][i] == dev_viterbi_predicted_list[i]:
        accurate_words += 1
        
accuracy = accurate_words / len(dev_viterbi_predicted_list)     

print("Accuracy of Viterbi model of Dev data:", accuracy * 100, "%")


# ### 4.3 Generating results for test dataset

# In[59]:


#read test dataset

df_test_2 = pd.read_csv('data/test', sep='\t', on_bad_lines = 'skip')
df_test_2.columns = ['index', 'word']
df_test_2.loc[-1] = ['1', 'Influential']  # adding a row
df_test_2.index = df_test_2.index + 1  # shifting index
df_test_2.sort_index(inplace=True)


# In[60]:


#generate list of predicted tags

viterbi_test_predicted_list = viterbi_decoder(df_test_2)


# In[61]:


df_test_2['postag'] = viterbi_test_predicted_list


# In[62]:


df_test_2['word'] = np.where(df_test_2['word'].isin(vocab_list), df_test_2['word'], '<unk>')


# In[63]:


df_test_2


# In[64]:


df_test_2.to_csv('viterbi.out', sep='\t', header=None, index=None)


# In[65]:


with open('viterbi.out', 'w') as f:
    
    f.write("1" + "\t" + df_test_2.iloc[0]['word'] + "\t" + str(viterbi_test_predicted_list[0]) + "\n" )
    
    count = 2
    for i in range(2, len(df_test)):
        if(df_test_2.iloc[i]['index'] == 1):
            count = 1
            f.write("\n")
        f.write(str(count) + "\t" + df_test_2.iloc[i]['word'] + "\t" + str(viterbi_test_predicted_list[i]) + "\n" )
        count+=1


# In[ ]:


ipython nbconvert --to python *.ipynb

