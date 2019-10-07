from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec


#Comments rating below 2.5 are negative and above are positive.
df=pd.read_csv('Ratings.csv')
df['Sent']=df['Rating'].apply(lambda x: 1 if int(x)>2.5 else 0)

##TOPIC MODELLING FOR POSITIVE COMMENTS
#Remove stopwords,lemmatize each word, create corpus and tokenize them.
stops=stopwords.words('english')
lem=WordNetLemmatizer() #Finding the root words which make sense. Slightly different from stemming.
corpus=' '.join(lem.lemmatize(x) for x in df[df['Sent'] == 1]['Review'][:3000]if x not in stops)
tokens=word_tokenize(corpus)

#Term Frequqnecy Inverse doc Frequency(Tfidf) to vectorize the tokens.
vect=TfidfVectorizer()
vect_fit=vect.fit(tokens)

#Latent Dirichlet Allocation
id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vectorized_data=vect_fit.transform(tokens)
gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel=gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)


#Word counts of topic keywords and visualising output
counter=Counter(corpus)
topics=ldamodel.show_topics(formatted=False)
out=[]
for i,topic in topics:
    for word,weight in topic:
        out.append([word,i,weight,counter[word]])

df_1=pd.DataFrame(out,columns=['word','topic_id','importance','word_count'])
fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df_1.loc[df_1.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df_1.loc[df_1.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df_1.loc[df_1.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords for Positive Comments.', fontsize=8, y=1.05)
plt.show()


#TOPIC MODELLING FOR NEGATIVE COMMENTS

stops=stopwords.words('english')
lem=WordNetLemmatizer()
corpus_n=''.join(lem.lemmatize(x)for x in df[df['Sent']==0]['Review'][:3000] if x not in stops)
tokens_n=word_tokenize(corpus_n)
vect=TfidfVectorizer()
vect_fit=vect.fit(tokens_n)
id_map_n=dict((v,k) for k,v in vect.vocabulary_.items())
vectorized_data_n=vect_fit.transform(tokens_n)
gensim_corpus_n=gensim.matutils.Sparse2Corpus(vectorized_data_n,documents_columns=False)
ldamodel_n=gensim.models.ldamodel.LdaModel(gensim_corpus_n,id2word=id_map_n,num_topics=5,random_state=34,passes=25)
counter_n=Counter(corpus_n)
topics_n=ldamodel_n.show_topics(formatted=False)
out_n=[]
for i,topic in topics_n:
    for word,weight in topic:
        out_n.append([word,i,weight,counter_n[word]])

df_2=pd.DataFrame(out_n,columns=['word','topic_id','importance','word_count'])
fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df_2.loc[df_2.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df_2.loc[df_2.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df_1.loc[df_1.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords for Negative Comments.', fontsize=8, y=1.05)
plt.show()


#Reviews in 2D Space

stops=stopwords.words('english')
lem=WordNetLemmatizer()
corpus=[]
for review in tqdm(df['Review'][:10000]):
    words=[]
    for x in word_tokenize(review):
        x=lem.lemmatize(x.lower())
        if x not in stops:
            words.append(x)
    corpus.append(words)

#word2vec to rep each word as a vector
model=word2vec.Word2Vec(corpus,size=100,window=20,min_count=200,workers=4)

#TSNE model
def tsne_model(model):
    labels=[]
    tokens=[]
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tsne_model=TSNE(perplexity=40,n_components=2,init='pca',n_iter=2500,random_state=23)
    new_values=tsne_model.fit_transform(tokens)

    x,y=[],[]
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10,10))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i],y[i]),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.show()

print(tsne_model(model))

#TSNE of adjectives used in positive reviews
positive=df[df['Rating']>3]['Review'][:2000]
negative=df[df['Rating']<2.5]['Review'][:2000]

def return_corpus(df):

    corpus=[]
    for review in df:
        tagged=nltk.pos_tag(word_tokenize(review))
        adj=[]
        for x in tagged:
            if x[1]=='JJ':
                adj.append(x[0])
        corpus.append(adj)
    return corpus

corpus_positive=return_corpus(positive)
model_positive=word2vec.Word2Vec(corpus_positive,size=100,min_count=10,window=20,workers=4)
print(tsne_model(model_positive))

#TSNE of adjectives in negative reviews
corpus_negative=return_corpus(negative)
model_negative=word2vec.Word2Vec(corpus_negative,size=100,min_count=10,window=20,workers=4)
print(tsne_model(model_negative))


