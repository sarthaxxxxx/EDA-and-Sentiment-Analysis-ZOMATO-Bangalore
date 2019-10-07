import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
import seaborn as sns
from wordcloud import WordCloud
import geopy
from geopy.geocoders import Nominatim
from tqdm import tqdm
import re


for i in os.listdir("C:\\Users\\sarth\\Desktop\\Zomato"):
    if i=='zomato.csv':
        data=os.path.join("C:\\Users\\sarth\\Desktop\\Zomato",i)
        df=pd.read_csv(data)
    else: pass

print("Dataset contains {} rows and {} columns.".format(df.shape[0],df.shape[1]))

#Top Restaurant chains in Bangalore
plt.figure(figsize=(10,7))
chains=df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most famous restaurant chains in Bangalore.")
plt.xlabel("No of outlets.")
#plt.show()

#How many of them don't accept online orders?
plt.figure(figsize=(10,7))
x=df['online_order'].value_counts()
colors=['#FEBFB3','#E1396C']
plt.pie(x,labels=x.index,colors=colors,autopct='%1.1f%%',shadow=True)
#handles=[]
#for i,l in enumerate(x.index):
    #handles.append(matplotlib.patches.Patch(color=colors),label=l))
plt.title("Accepting vs non-accepting online orders")
#plt.show()

#Restaurants accepting table booking
tab=df['book_table'].value_counts()
plt.figure(figsize=(10,7))
plt.pie(tab,labels=tab.index,colors=colors,autopct='%1.1f%%',shadow=True)
plt.title("Table Booking.")
plt.show()

#Rating distribution
plt.figure(figsize=(6,5))
rating=df['rate'].dropna().apply(lambda x:float(x.split('/')[0])if (len(x)>3) else np.nan).dropna()
sns.distplot(rating,bins=20)
plt.title("Rating distribution.")
#plt.show()

cost_dist=df[['rate','approx_cost(for two people)','online_order']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x:float(x.split('/')[0])if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x:int(x.replace(',','')))
plt.figure(figsize=(10,7))
sns.scatterplot(x="rate",y="approx_cost(for two people)",hue='online_order',data=cost_dist)
#plt.show()

#Distribution of cost for two people.
plt.figure(figsize=(6,6))
sns.distplot(cost_dist['approx_cost(for two people)'])
#plt.show()

#Votes of restaurants accepting and non-accepting online orders.

#votes_yes=df[['online_order']=="Yes"]['votes']
#trace0=go.Box(y=votes_yes,name='accepting online orders',marker=dict(color='rgb(214,12,140)'))
#votes_no=df[['online_order']=="No"]['votes']
#trace1=go.Box(y=votes_no,name='non-accepting online orders',marker=dict(color='rgb(0,128,128)'))
#layout=go.Layout(title="Box plots of votes",width=800,height=500)
#data=[trace0,trace1]
#fig=go.Figure(data=data,layout=layout)
#plt.plot(fig)

plt.figure(figsize=(10,6))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types.")
plt.xlabel("Count.")
#plt.show()

#Finding the best budget restaurants in any location.
cost_dist=df[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x:float(x.split('/')[0])if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x:int(x.replace(',','')))
def return_budget(location,rest):
    budget=cost_dist[(cost_dist['approx_cost(for two people)']<=400)&(cost_dist['location']==location)&(cost_dist['rate']>=4)&(cost_dist['rest_type']==rest)]
    return (budget['name'].unique())

print("Best options:",return_budget('Indiranagar','Dessert Parlor'))

#Foodie areas
plt.figure(figsize=(10,6))
rest_locs=df['location'].value_counts()[:20]
sns.barplot(rest_locs,rest_locs.index,palette='rocket')
plt.title("Foodie Areas.")
#plt.show()

#Most common cuisines
df_1=df.groupby(['location', 'cuisines']).agg('count')
data=df_1.sort_values(['url'],ascending=False).groupby(['location'],as_index=False).apply(lambda x: x.sort_values(by='url',ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})
print(data.head())

#Extracting location info using GeoPy
locations=pd.DataFrame({"Name":df['location'].unique()})
locations['Name']=locations['Name'].apply(lambda x:"Bangalore"+str(x))
geolocator=Nominatim(user_agent="app")
lat_lon=[]
for location in locations['Name']:
    location=geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:
        geo=(location.latitude,location.latitude)
        lat_lon.append(geo)
locations['geo_loc']=lat_lon
locations.to_csv('locations.csv',index=False)
locations['Name']=locations['Name'].apply(lambda x: x.replace("Bangalore"," ")[0:])
#print(locations.head())

#wordcloud of dishes liked by cuisines
df['dish_liked']=df['dish_liked'].apply(lambda x: x.split(',') if type(x)==str else [''])
rest=df['rest_type'].value_counts()[:9].index
def wordcld(rest):
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus =df[df['rest_type']==r]['dish_liked'].values.tolist()
        corpus =','.join(x for list_words in corpus for x in list_words)
        wordcloud=WordCloud(max_font_size=None,background_color='white',collocations=False,width=1500,height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
       # plt.show()

#print(wordcld(rest))

#Prep the reviews dataframe
all_ratings=[]
for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):
    ratings=eval(ratings)
    for score,doc in ratings:
        if score:
            score=float(score.strip("Rated").strip())
            doc=doc.strip("Rated").strip()
            all_ratings.append([name,score,doc])
rating_Df=pd.DataFrame(all_ratings,columns=['Name','Rating','Review'])
rating_Df['Review']=rating_Df['Review'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]',"",x))
rating_Df.to_csv("Ratings.csv")

#Rating Distribution
plt.figure(figsize=(10,6))
rating=rating_Df['Rating'].value_counts()
sns.barplot(x=rating.index,y=rating)
plt.xlabel("Ratings")
plt.ylabel("Count")
plt.show()






