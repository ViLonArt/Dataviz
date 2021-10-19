#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:00:14 2021

@author: vincenttran
"""
#imports section
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import missingno as msno
from PIL import Image
#Setting up the streamlit page
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


#Building a function in order to create a dynamic table of content
class TableContent:

    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

col1,col2,col3=st.columns(3)
with col1:
    st.subheader('Vincent Tran - 2021')

with col2:
    image = Image.open('Linkedin.png')
    st.image(image, caption='Linkedin',width=100)
with col3:
    image = Image.open('Github.png')
    st.image(image, caption='Github',width=100)

toc = TableContent()

toc.title('Your Health Data')

image = Image.open('health.png')
st.image(image,width=900)

toc.placeholder()





toc.header('Dataframe')
#first part : Building our Dataframe from the Appel health app data

toc.subheader("Before cleaning")

df=pd.read_csv('HealthAutoExport-2016-08-13-2021-09-20 Data 2.csv')
df['Date']=df['Date'].map(pd.to_datetime)
df['Step Count (count)']=df['Step Count (count)'].astype('int64')

st.dataframe(df)
matrix=msno.matrix(df.set_index('Date'),freq='BQ')
st.pyplot(matrix.figure)


toc.subheader('After cleaning')
st.text("let's get rid of the non-exploitable data")
st.text('Starting with the 2 first columns which don\'t have any values')
df=df.drop(columns=['Apple Stand Hour (count)','Apple Stand Time (min)'])
    
st.text('Then, let\'s fill the NaN values in the other columns based on their means.\nExcept for the Headphone Exposure because '
         'we don\'t necessarily use them everyday')

##############################

#Replacing the blank value by NaN
df['Walking Step Length (cm) '].replace(' ','NaN',inplace=True)
df['Walking Step Length (cm) ']=df['Walking Step Length (cm) '].astype('float64')
#Getting the mean values to fill NaN values
mean_headphoneExp=df['Headphone Audio Exposure (dBASPL)'].mean()
mean_StepC=df['Step Count (count)'].mean()
mean_dist=df['Walking + Running Distance (km)'].mean()
mean_asym=df['Walking Asymmetry Percentage (%)'].mean()
mean_speed=df['Walking Speed (km/hr)'].mean()
mean_stepL=df['Walking Step Length (cm) '].mean()

df['Headphone Audio Exposure (dBASPL)'].fillna(value=0,inplace=True)
df['Step Count (count)'].fillna(value=mean_StepC,inplace=True)
df['Walking + Running Distance (km)'].fillna(value=mean_dist,inplace=True)
df['Walking Asymmetry Percentage (%)'].fillna(value=mean_asym,inplace=True)
df['Walking Speed (km/hr)'].fillna(mean_speed,inplace=True)
df['Walking Step Length (cm) '].fillna(mean_stepL,inplace=True)

st.write(df.tail(10))

matrix=msno.matrix(df.set_index('Date'),freq='BQ')
st.pyplot(matrix.figure)


#############################
#Second Part - Some quick vizualtion of the Data as a whole
#Visualization
toc.header('Visualization')


#st.write('Let\'s visualize some data !')


dfY=df.groupby([df['Date'].dt.year]).agg({'Step Count (count)' : 'sum',
                                                   'Walking + Running Distance (km)':'sum',
                                                   'Headphone Audio Exposure (dBASPL)':'mean',
                                                   'Walking Asymmetry Percentage (%)':'mean',
                                                   'Walking Speed (km/hr)': 'mean',
                                                   'Walking Step Length (cm) ':'mean'})
dfM=df.groupby([df['Date'].dt.month]).agg({'Step Count (count)' : 'mean',
                                                   'Walking + Running Distance (km)':'mean',
                                                   'Headphone Audio Exposure (dBASPL)':'mean',
                                                   'Walking Asymmetry Percentage (%)':'mean',
                                                   'Walking Speed (km/hr)': 'mean',
                                                   'Walking Step Length (cm) ':'mean'})
dfD=df.groupby([df['Date'].dt.day]).agg({'Step Count (count)' : 'sum',
                                                   'Walking + Running Distance (km)':'sum',
                                                   'Headphone Audio Exposure (dBASPL)':'mean',
                                                   'Walking Asymmetry Percentage (%)':'mean',
                                                   'Walking Speed (km/hr)': 'mean',
                                                   'Walking Step Length (cm) ':'mean'})


toc.subheader('Basic visualization of the whole data per year')
fig = px.bar(dfY, barmode='group')
st.plotly_chart(fig)

toc.subheader('Basic visualization of the whole data per month')
fig2 = px.bar(dfM, barmode='group')
st.plotly_chart(fig2)
    
toc.subheader('Basic visualization of the whole data per day')
fig3 = px.bar(dfD, barmode='group')
st.plotly_chart(fig3)

####################################################
####################################################

#A more in depth analysis
toc.header('In depth analysis of the Data')

toc.subheader('Correlation between Step length and Walking speed')

st.write('Is there any correlation between my step length and my speed ?')

model=LinearRegression()
model.fit(dfD[['Walking Speed (km/hr)']],dfD[['Walking Step Length (cm) ']])
y_pred=model.predict(dfD[['Walking Speed (km/hr)']])



fig4=plt.figure()
plt.scatter(dfD[['Walking Speed (km/hr)']],dfD[['Walking Step Length (cm) ']])
plt.plot(dfD[['Walking Speed (km/hr)']], y_pred, c="red")
plt.xlabel('Walking Speed (km/hr)')
plt.ylabel('Walking Step Length (cm)')
st.pyplot(fig4)

####################################################
####################################################
toc.subheader('The impact of COVID19')

#We focus on 2019 and 2020, especially the transition between the two years
mask = (df['Date'] > '2019-07-01') & (df['Date'] <= '2019-12-31')
m1=df[mask][['Walking + Running Distance (km)','Date']]
mask2 = (df['Date'] > '2020-01-01') & (df['Date'] <= '2020-07-01')
m2=df[mask2][['Walking + Running Distance (km)','Date']]

m1m=m1.groupby([m1['Date'].dt.month]).agg({'Walking + Running Distance (km)':'mean'})
m2m=m2.groupby([m2['Date'].dt.month]).agg({'Walking + Running Distance (km)':'mean'})

m1=m1.groupby([m1['Date'].dt.month]).agg({'Walking + Running Distance (km)':'sum'})
m2=m2.groupby([m2['Date'].dt.month]).agg({'Walking + Running Distance (km)':'sum'})


m1m.rename(index={7: 'Jul19',8: 'Aug19',9:'Sep19',10:'Oct19',11:'Nov19',12:'Dec19'},inplace=True)
m2m.rename(index={1:'Jan20',2:'Feb20',3:'Mar20',4:'Apr20',5:'May20',6:'Jun20',7:'Jul20'},inplace=True)

m1.rename(index={7: 'Jul19',8: 'Aug19',9:'Sep19',10:'Oct19',11:'Nov19',12:'Dec19'},inplace=True)
m2.rename(index={1:'Jan20',2:'Feb20',3:'Mar20',4:'Apr20',5:'May20',6:'Jun20',7:'Jul20'},inplace=True)


col1,col2=st.columns(2)
with col1:
    st.text('The total distance per months')
    m3=pd.concat([m1,m2])
    st.write(m3)
    fig = px.bar(m3, barmode='group')
    st.plotly_chart(fig)
with col2:
    st.text('The average distance per months')
    m3m=pd.concat([m1m,m2m])
    st.write(m3m)
    fig = px.bar(m3m, barmode='group')
    st.plotly_chart(fig)

toc.subheader('The impact of Time on the features')
st.write('Impact of the time on the Audio exposure, Walking distance and number of steps ')

dfy=dfY.drop(columns={'Walking Asymmetry Percentage (%)','Walking Speed (km/hr)','Walking Step Length (cm) '})
dfm=dfM.drop(columns={'Walking Asymmetry Percentage (%)','Walking Speed (km/hr)','Walking Step Length (cm) '})
#Let's normalize the data first so we can put the three variables on the same playing field
scaler=MinMaxScaler()
scaler.fit(dfy)
dfYS=scaler.transform(dfy)

scalerM=MinMaxScaler()
scalerM.fit(dfm)
dfMS=scalerM.transform(dfm)

dfYS=pd.DataFrame(data=dfYS,index=['2016','2017','2018','2019','2020','2021'],columns=['Step Count', 'Walking distance','Headphone Audio Exposure (dBASPL)'])
dfMS=pd.DataFrame(data=dfMS,columns=['Step Count', 'Walking distance','Headphone Audio Exposure (dBASPL)'])

col1,col2=st.columns(2)
with col1:
    st.text('Per year')
    fig = px.bar(dfYS, barmode='group')
    st.plotly_chart(fig)
with col2:
    st.text('Per Months')
    fig = px.bar(dfMS, barmode='group')
    st.plotly_chart(fig)


st.write('Impact of the time on the Walking Asymmetry Percentage, Walking speed and step length ')

dfy=dfY.drop(columns={'Step Count (count)','Walking + Running Distance (km)','Headphone Audio Exposure (dBASPL)'})
dfm=dfM.drop(columns={'Step Count (count)','Walking + Running Distance (km)','Headphone Audio Exposure (dBASPL)'})


col1,col2=st.columns(2)
with col1:
    st.text('Per year')
    fig = px.bar(dfy, barmode='group')
    st.plotly_chart(fig)
with col2:
    st.text('Per Months')
    fig = px.bar(dfm, barmode='group')
    st.plotly_chart(fig)

#the table of content is generated at the end
toc.generate()
