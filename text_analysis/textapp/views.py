from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd #panda to create dataframe
import tweepy #lib to twitter
from tweepy.streaming import StreamListener #library to listen tweets and fetch
from tweepy import OAuthHandler #authentication for twitter api
from tweepy import Stream
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.sentiment.util import *
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#import pyttsx3

# def index(request):
# 	return render(request,'home.html')

def index(request):
	if request.method=="POST" and request.POST['name']:
		uname=request.POST["name"]
		upass=request.POST["fld"]
		textno=request.POST["nm"]
		textno=str(int(float(textno)))
		print("Number of tweets are: ",textno)
		#response="Name is " +uname+ " and field is  "+upass+" and number of texts is: "+textno+"."

		access_token="956033448577241089-6S1DYfuZdl55BtGnArUHw8ofVWS4Cep"
		access_token_secret="2gwgxEJC9OxsdhSgVRjcprzZY3GxFccb42LvSDSGAJkIK"
		consumer_key="flGGNLwGmqNiav5ooMwgkIklF"
		consumer_secret="FzSANGquLnPSlqAZAutGy7hbjEq0S3FrVFwzkrGEDKsp5OLirW"
		"""using above details establishing connections to twitter api 
		and providing authentication to fetch and search tweets"""
		auth=tweepy.auth.OAuthHandler(consumer_key,consumer_secret)
		auth.set_access_token(access_token,access_token_secret)
		api=tweepy.API(auth)
		"""fetching tweets using tweepy libray for given Choice of name
		field and number to analysing tweets"""
		#fetched_tweets=api.search(q=[input(uname),input(upass)],result_type='recent',lang='en',count=input(textno))
		fetched_tweets=api.search(q=[uname,upass],result_type='recent',lang='en',count=textno)
		def populate_tweet_df(tweets):
			df=pd.DataFrame()
			df['id']=list(map(lambda tweet:tweet.id,tweets))
			df['text']=list(map(lambda tweet:tweet.text,tweets))
			df['retweeted']=list(map(lambda tweet:tweet.retweeted,tweets))
			df['place']=list(map(lambda tweet:tweet.user.location,tweets))
			df['screen_name']=list(map(lambda tweet:tweet.user.screen_name,tweets))
			df['verified_user']=list(map(lambda tweet:tweet.user.verified,tweets))
			df['followers_count']=list(map(lambda tweet:tweet.user.followers_count,tweets))
			df['friends_count']=list(map(lambda tweet:tweet.user.friends_count,tweets))
			return df

		df=populate_tweet_df(fetched_tweets)
		print()

		SIA=SentimentIntensityAnalyzer()
		"""Calculating polarity of each tweet using The Sentiment intensity analyser t
		hat uses Corpus dictionary and bag of word process to Predict 
		the polarity of the Tweets And each postive score ,negative score and nuetral score
		storing each score in diffrent fields of data frame"""


		df['polarity']=df.text.apply(lambda x:SIA.polarity_scores(x)['compound'])
		df['nuetral_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['neu'])
		df['negative_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['neg'])
		df['positive_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['pos'])
		df['sentiment']=''
		#sentiment column
		"""Using the polarity score predicted by the sentiment intensity analyzer
		check the polarity score and predict the sentiment of the tweet"""
		df.loc[df.polarity>0,'sentiment']=1
		df.loc[df.polarity==0,'sentiment']=0
		df.loc[df.polarity<0,'sentiment']=-1
		

		#print(df.head())

		"""Last stage of showing the stats of the analyzed tweets using bar or pie graph"""
		file_path1=os.path.join(BASE_DIR,'static')
		filename1="s.csv"
		final=os.path.join(file_path1,filename1)		
		df.to_csv(final)
		#df.to_csv("f:/s.csv")
		# importing the dataset
		#importing the data from csv file into dataset
		dataset=pd.read_csv("static/s.csv")
		if len(dataset)==0:
			return render(request,'error.html')

		if textno=="1":
			dataset.sentiment.value_counts().plot(kind='bar',title="Sentiment Analysis")
			file_path=os.path.join(BASE_DIR,'static')
			filename="graph.png"
			final=os.path.join(file_path,filename)
			plt.savefig(final, format="png")
			return render(request,'ret.html')

		#dataset=pd.read_csv('f:/s.csv')
		

		# Preprocessing of sentences
		#preprocessing of all the sentences by using library nltk
		import re
		import nltk
		from nltk.corpus import stopwords
		from nltk.stem.porter import PorterStemmer
		corpus=[]   # create a corpus list
		for i in range(len(fetched_tweets)):
			review=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
			review=review.lower()
			review=review.split()
			ps=PorterStemmer()
			review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
			review=' '.join(review)
			corpus.append(review)    # store preprocessed sentences into list
		# using a library CountVectorizer
		#importing CountVectorizer
		from sklearn.feature_extraction.text import CountVectorizer
		cv=CountVectorizer()
		x=cv.fit_transform(corpus).toarray()
		y=dataset.iloc[:,13].values

		# splitting of dataset into training and testing
		from sklearn.cross_validation import train_test_split
		x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

		# using Naive_Bayes classifier
		from sklearn.naive_bayes import GaussianNB
		classifier=GaussianNB()
		classifier.fit(x_train,y_train)

		y_pred=classifier.predict(x_test)

		# creation of Confusion matrix
		from sklearn.metrics import confusion_matrix
		cm=confusion_matrix(y_test,y_pred)
		#print(cm)

		# train accuracy
		#print('**************************************************************************')
		from sklearn.metrics import accuracy_score
		accuracy=accuracy_score(y_train,classifier.predict(x_train))
		#print('Train Accuracy: ',accuracy)

		#print('*************************************************************************')
		# test accuracy
		accuracy1=accuracy_score(y_test,y_pred)
		#print('Test Accuracy: ',accuracy1)

		"""Last stage of showing the stats of the analyzed tweets using bar or pie graph"""
		dataset.sentiment.value_counts().plot(kind='bar',title="Sentiment Analysis")






		#df.sentiment.value_counts().plot(kind='bar',title="Sentiment Analysis")
		#plt.show()

		file_path=os.path.join(BASE_DIR,'static')
		filename="graph.png"
		final=os.path.join(file_path,filename)
		#print(final)
		#if os.path.exists(final):
			#os.remove(final)
		plt.savefig(final, format="png")
		return render(request, 'ret.html')
	return render(request, 'home.html')