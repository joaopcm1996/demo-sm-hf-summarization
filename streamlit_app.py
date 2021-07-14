import streamlit as st
import boto3 
import json
from argparse import ArgumentParser
import os

sm = boto3.client('sagemaker-runtime')

# Define parser to take in 
parser = ArgumentParser()
parser.add_argument('--endpoint_name')
args = parser.parse_args()

def predict(text, len_pen, endpoint_name='summarization-endpoint'):
    endpoint = args.endpoint_name
    response = sm.invoke_endpoint(EndpointName=endpoint,
                                  Body=json.dumps({
                                      'inputs':text,
                                      'parameters':{
                                          'length_penalty':len_pen
                                      }
                                  }),
                                  ContentType='application/json'
                                 )
    response_text = json.loads(response['Body'].read().decode())
    
    return response_text['summary']

st.sidebar.subheader('Choose a shorter or longer summary')
slider = st.sidebar.slider(
    '0.5 - shortest  -------------- longest - 1.5',
    min_value = 0.5,
    max_value = 1.5 
)
fulltext = st.text_area('Enter the text you want to summarize here')

if st.button('Summarize'): 
    with st.spinner():
        summary = predict(fulltext,slider)
        st.success('Done!')
        st.info(summary)


