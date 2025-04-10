import joblib
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from ntscraper import Nitter
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import altair as alt
import re
import streamlit as st
import csv
import subprocess
import torch.nn.functional as F
import joblib
from deep_translator import GoogleTranslator
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain
import altair as alt
from PIL import Image
import numpy as np
import plotly.express as px
from torpy.http.requests import TorRequests
import random
from instagrapi import Client
import time
from datetime import datetime
from transformers import BertModel
from transformers import (AutoTokenizer)
import torch.nn as nn

from scrapy.crawler import CrawlerProcess

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from scrapy_selenium import SeleniumRequest
import time
import random
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait


#######################
# Page configuration
st.set_page_config(
    page_title="Social Media",
    page_icon="\U0001F310",
    layout="wide",
    initial_sidebar_state="expanded")

def my_custom_theme():
    return {
        'config': {
            'background': 'white',
            'view': {'stroke': 'transparent'},  # No border around the chart
            'title': {'fontSize': 20, 'font': 'Arial', 'color': 'black'},
            'axis': {
                'domainColor': 'gray',
                'gridColor': 'lightgray',
                'labelFontSize': 12,
                'titleFontSize': 14,
            },
        }
    }

# Register and enable the custom theme
alt.themes.register('my_custom_theme', my_custom_theme)
alt.themes.enable('my_custom_theme')


#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
 
}

[data-testid="stMetric"] {
    background-color: #DBE2F0;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
  color: #4B4C4E;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class model(nn.Module):
    def __init__(self, checkpoint, freeze=False, device='cpu'):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(checkpoint)
        hidden_sz = self.model.config.hidden_size
        # set device cuda or cpu
        self.device = device
        # freeze model
        if freeze:
            for layer in self.model.parameters():
                layer.requires_grad=False
        
    def forward(self, x, attention_mask=None):
        x = x.to(self.device)
        # pooler_output(seq,dim) 
        with torch.no_grad():
            model_out = self.model(x['input_ids'], x['attention_mask'], return_dict=True)
            
        embds = model_out.last_hidden_state # model_out[0][:,0]
        mean_pool = embds.sum(axis=1)/ x['attention_mask'].sum(axis=1).unsqueeze(axis=1)
        return mean_pool
    
@st.cache_data 
def predict_sentiments(sentences):
    #model load
    model_checkpoint = "indolem/indobert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load saved model
    model = BertClassifier(num_labels=3)
    model.load_state_dict(torch.load("sentiment_model.pth",map_location=torch.device('cpu')))
    model.to(device)

    model.eval()  
    max_length = 256 

    # Tokenize input
    encodings = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask) 
        probabilities = F.softmax(outputs, dim=1)  
        predicted_classes = torch.argmax(probabilities, dim=1)  

    # Map class 
    sentiment_mapping = {1: "positif", 0: "netral", 2: "negatif"}
    predicted_labels = [sentiment_mapping[class_idx.item()] for class_idx in predicted_classes]

    # list score
    scores = probabilities.cpu().numpy()

    return scores,predicted_labels


class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        # model load
        model_checkpoint = "indolem/indobert-base-uncased"
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_labels)

        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state'][:, 0, :]
        x = self.classifier(x)
        return x


@st.cache_data 
def get_topik(kata):
    checkpoint = 'indolem/indobertweet-base-uncased'
    indobert = model(checkpoint, freeze=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model_load = joblib.load('model_topik_new.pkl')


    # dataloader
    final_embeddings =list()
    all_embeddings = []
    final_sentences = kata

    batch_sz = 200 # batch_size
    for idx in range(0, len(final_sentences), batch_sz):
        batch_sentences = final_sentences[idx:idx+batch_sz]
        for sent in batch_sentences:
            tokens = tokenizer(sent ,truncation='longest_first', return_tensors='pt', return_attention_mask=True, padding=True)
            embeddings = indobert(tokens)
            final_embeddings.extend(embeddings)
            all_embeddings = torch.stack(final_embeddings)

    return model_load.predict(pd.DataFrame(all_embeddings))
    

def clean_text(text):
    
    text = re.sub(r'\d+', '', text)
    # Menghapus spasi ekstra
    text = re.sub(r'\s+', ' ', text)
    # Menghapus mention
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # removing mentions
    # Menghapus hashtag
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # removing hastag
    text = re.sub(r'RT[\s]+', '', text)  # removing RT
    text = re.sub(r"http\S+", '', text)  # removing link
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    text = re.sub(r'[^A-Za-z]+', ' ', text)  # removing all character non alphabet

    text = text.replace('\n', ' ')  # replace new line into space
  
    text = GoogleTranslator(target='id').translate(text)
    text = text.lower() 
    text = text.strip(' ')
    
    #text = re.sub(r'http\S+', '', text)  # Remove URL
    #text = re.sub(r'@\w+', '', text)  # Remove mention
    #text = re.sub(r'#\w+', '', text)  # Remove hashtags
    #teks = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special character dan angka
    #teks = teks.lower()  # Convert ke lowercase
    
    return text

# tokenize, remove stop words
def preprocess_text_sastrawi(text):
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()

    tokens = word_tokenize(text)  # Tokenization
    #tokens = [word for word in tokens if word not in stopword_sastrawi]  # Remove stop words
  
    return ' '.join(tokens)

# non-formal ke formal
def replace_slang_word(doc):
    slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
    doc = word_tokenize(doc)
    for index in  range(0,len(doc)-1):
        index_slang = slang_word.slang==doc[index]
        formal = list(set(slang_word[index_slang].formal))
        if len(formal)==1:
            doc[index]=formal[0]
    return ' '.join(doc)

def get_tweets_nitter(hashtag, mode, size):
    scraper = Nitter()
    scraper.instance = "https://nitter.net"
    tweets = scraper.get_tweets(hashtag, mode = mode, number = size, replies=True)
    final_tweets = []

    for tweet in tweets['tweets']:
        data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
        final_tweets.append(data)

    data = pd.DataFrame(final_tweets, columns = ['link', 'text', 'date', 'Likes', 'Comments'])
    return data

@st.cache_data
def get_tweets(keyword, limit):
    #command = f"npx -y tweet-harvest@latest -o 'C:\\Users\\imvla\\ZalidQomalita\\Job\\3. Script\\2024\\Sosmed\\tweets-data\\tweet_lokal.csv' -s '{keyword}' --tab LATEST -l {limit} --token 694ef22c191742028586b7984792517a7e028d1c"
    command = f"npx -y tweet-harvest@latest -o 'tweet_lokal.csv' -s '{keyword}' --tab LATEST -l {limit} --token 694ef22c191742028586b7984792517a7e028d1c"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Cek output atau error dari eksekusi
    
    if result.returncode == 0:
        print("Perintah berhasil dijalankan!")
        print("Output:", result.stdout)
        data = pd.read_csv("C:\\Users\\imvla\\ZalidQomalita\\Job\\3. Script\\2024\\Sosmed\\tweets-data\\'tweet_lokal'.csv",delimiter=',')
        #data = pd.read_csv("tweets-data\\tweet_lokal.csv",delimiter=',')
    else:
        print("Terjadi kesalahan!")
        print("Error:", result.stderr)
        data=pd.DataFrame()
    return data

@st.cache_data
def get_instapost(keyword,limit):
    with TorRequests() as tor_requests:
        with tor_requests.get_session() as sess:
            print(sess.get("http://httpbin.org/ip").json())
            time.sleep(random.randint(5, 10))
            cl = Client()

            # Login 
            USERNAME = "ini.4j0b"
            PASSWORD = "bismillah"

            cl.login(USERNAME, PASSWORD)
            medias = cl.hashtag_medias_recent(keyword, amount=limit)
            print(medias)
            
            list_insta = []
            for media in medias:
                med = media.dict()
                print(med)
                taken_at = med.get('taken_at')
                username = med.get('user', {}).get('username')
                try:
                    location_name = med.get('location', {}).get('name')
                    latitude = med.get('location', {}).get('lat')
                    longitude = med.get('location', {}).get('lng')
                except Exception as e:
                    location_name = '-'
                    latitude = '-'
                    longitude = '-'
                caption = med.get('caption_text')
                usertags = med.get('usertags', [])
                like_count = med.get('like_count')
                comment_count = med.get('comment_count')

                list_insta.append((taken_at, username, location_name, latitude, longitude, caption, usertags, like_count, comment_count))
            
            df_insta = pd.DataFrame(list_insta,columns=['created_at','username','location_name', 'latitude','longitude','full_text','usertags','like_count','comment_count'])
    return df_insta


def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    lexicon_positive = dict()
    with open('lexicon_positive.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    with open('lexicon_negative.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])

    for word_pos in text:
        if (word_pos in lexicon_positive):
            score = score + lexicon_positive[word_pos]
    for word_neg in text:
        if (word_neg in lexicon_negative):
            score = score + lexicon_negative[word_neg]

    polarity=''
    if (score > 0):
        polarity = 'positif'
    elif (score < 0):
        polarity = 'negatif'
    else:
        polarity = 'netral'
    
    return score, polarity


def make_wordcloud(text_cloud):
    mask = np.array(Image.open('C:\\Users\\imvla\\ZalidQomalita\\Job\\3. Script\\2024\\Sosmed\\mask kota bandung.jpg'))
    wordcloud = WordCloud(width=600, height=400, max_words=250,colormap='twilight',collocations=True, contour_width=1, mask=mask,contour_color='grey', background_color='white').generate(text_cloud)
    fig, ax = plt.subplots()
    print(wordcloud)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


def remove_stopword(words):
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()
    stopword_sastrawi.extend(["yg",'yang','biar','ada','enggak', "dg", "rt", "dgn", "ny", "d", 'kl', 'klo','kalau',
                              'kalo', 'amp', 'biar', 'bikin', 'bilang', 'jika','akan','selalu','aku','ke','di','saling',
                              'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'untuk','tidak','tak','tbtb', 'wkwk','wkwkwk','wkwkwkwk','mulu',
                              'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'usah','euy','kang','teh','teteh','akang','mbak','mas','om','tante','bapak','ibu',
                              'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'ya','lho','lo','ajg','anjay','ajng'
                              'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt','aku','kamu','dia','mereka','kalian','&amp', 'yah'])
    return [word for word in words if word not in stopword_sastrawi]



def main():
    with st.sidebar:
        st.title('\U0001F310 Social Media Analysis')
        
        sosmed = ["Twitter","Instagram"]
        selected_sosmed = st.selectbox('Media Sosial', sosmed , index = len(sosmed)-1)
        keyword = st.text_input("Kata kunci pencarian, contoh: kotabandung")
        limit = st.text_input("Maksimum jumlah data ditarik, contoh: 50")
        print("Keyword, Limit : ", keyword,limit)

        analyze = st.button('Analyze')

        
    placeholder = st.empty()
    if analyze:
        with placeholder:
            st.write("Processing...")
            print(selected_sosmed)
            if selected_sosmed=="Twitter":
                #file = get_tweets(keyword,int(limit))
                file = get_tweets_nitter(keyword,mode='hashtag',size=int(limit))
                teks_metriks = 'Tweet'
                nama_kolom = ['Text','Topik','Sentimen','Tanggal Posting','Teks','Jumlah Like','Jumlah Reply','Jumlah Retweet']
            if selected_sosmed=="Instagram":
                file = get_instapost(keyword,int(limit))
                teks_metriks = 'Post'
                nama_kolom = ['Text','Topik','Sentimen','Tanggal Posting','Username','Lokasi', 'Latitude','Longitude','Teks','Usertags','Jumlah Like','Jumlah Comment']

            # Preprocessing Data
            file['full_text'] = file['full_text'].apply(lambda x: clean_text(x))
            processed_text = file['full_text'].apply(lambda x: preprocess_text_sastrawi(x))

            # Text Normalization / Noise Removal
            final_text = processed_text.apply(lambda x : replace_slang_word(x))
            clean_data = final_text
    
            prediksi = get_topik(clean_data)
                
            # Klasifikasikan
            hasil = pd.Series(prediksi).map({1:'ekonomi', 2: 'teknologi', 3:'hukum', 4:'sosial', 5:'kesehatan', 6:'politik', 7:'hiburan',8:'kependudukan', 9:'lingkungan hidup', 10:'infrastruktur', 11:'pendidikan', 12:'administrasi pemerintahan' })
            topik = pd.concat([clean_data, hasil.to_frame()], axis=1)
            print(topik)

            df_class = topik
            df_class.columns = ["teks","topik"]
            
            
            word =  df_class['teks'].apply(word_tokenize)
            print(word)

            # Wordcloud
            word_v = word.apply(remove_stopword)
            combined_list = list(chain(*word_v))
            text_cloud = " ".join(combined_list)
            placeholder.empty()

        col = st.columns((1, 4.5, 2.5), gap='medium')
        with col[0]:
            st.markdown("#### Jumlah/Total")
            st.metric(teks_metriks, len(file))
            st.metric("User", file['username'].nunique())
            if selected_sosmed=="Twitter":
                file['created_at'] = pd.to_datetime(file['created_at'], format="%a %b %d %H:%M:%S %z %Y").dt.date
            if selected_sosmed=="Instagram":
                file['created_at'] =file['created_at'].dt.date

            # Hitung jumlah hari unik
            unique_days_count = file['created_at'].nunique()
            st.metric("Hari", unique_days_count)

        with col[1]:
            st.markdown("#### Pembicaraan Populer")
            make_wordcloud(text_cloud)
            print(text_cloud)

        with col[2]:
            st.markdown('#### Top Topik')
            df_class = df_class[df_class['topik'] != 'pendidikan']
            df_class_count = df_class['topik'].value_counts()
            # Convert ke dataframe
            topik_counts_df = df_class_count.reset_index()
            topik_counts_df.columns = ['topik', 'jumlah']
            topik_counts_df.sort_values(by='jumlah', ascending=False).reset_index(drop=True)

            st.dataframe(topik_counts_df,
                        column_order=("topik", "jumlah"),
                        hide_index=True,
                        width=None,
                        column_config={
                            "topik": st.column_config.TextColumn(
                                "Topik",
                            ),
                            "jumlah": st.column_config.ProgressColumn(
                                "Jumlah",
                                format="%f",
                                min_value=0,
                                max_value=max(topik_counts_df.jumlah),
                            )}
                        )
                
        
        col2 =  st.columns(1)
        with col2[0]:
            st.markdown('#### Proporsi Sentimen')
            #results = word.apply(sentiment_analysis_lexicon_indonesia)
            
            results = predict_sentiments(clean_data.to_list())
            print(results)
            
            polarity_score = pd.DataFrame(results[0], columns=['positif', 'netral', 'negatif'])
            polarity = results[1]
            print(polarity_score)
            print(pd.DataFrame(polarity).value_counts())

            vcount =  pd.DataFrame(polarity).value_counts().reset_index()
            vcount.columns=['polarity','count']

            fig = px.pie(vcount, values='count',height=300, width=200, names='polarity',color="polarity", color_discrete_map={"negatif":"#F36B59",
                                 "netral":"#8CA7C0",
                                 "positif":"#27AE60"})
            fig.update_layout(margin=dict(l=10, r=20, t=30, b=0),)
            st.plotly_chart(fig, use_container_width=True)
            
        col3 =  st.columns(1)
        with col3[0]:
            df_concat = pd.concat([df_class,pd.DataFrame(polarity)], axis=1)
            df_concat.columns = ['teks','topik','polaritas']
            df_concat.drop(columns=['teks'], inplace=True)
            print(df_concat)

            df_concat['topik2'] = df_concat['topik']
            df_concat = df_concat[df_concat['topik'] != 'pendidikan']
            grouped_df = df_concat.groupby(['topik', 'polaritas']).size().reset_index(name='jumlah').fillna(0)

            print(grouped_df)

            st.markdown("#### Sentimen Berdasar Topik")
            c = alt.Chart(grouped_df).mark_bar(size=20).encode(
                x=alt.X('polaritas:N', axis=None),
                y=alt.Y('jumlah:Q',  axis=alt.Axis(grid=True)),
                color=alt.Color('polaritas:N').legend(orient="bottom").scale(domain=['negatif','netral','positif'], range=['#F36B59','#8CA7C0','#27AE60']), 
                column=alt.Column('topik:O'),
                ).configure_header(labelOrient='bottom').configure_scale(bandPaddingInner=0,bandPaddingOuter=0.1,)
            st.altair_chart(c)
            
                    
        col4 = st.columns(1)    
        with col4[0]:
            if selected_sosmed=="Twitter":
                df_output = pd.concat([df_class,pd.DataFrame(polarity),file['created_at'],file['full_text'],file['favorite_count'], file['reply_count'],file['retweet_count']], axis=1)
                df_output.dropna(subset=['full_text'])
            if selected_sosmed=="Instagram":
                df_output = pd.concat([df_class,pd.DataFrame(polarity),file['created_at'],file['username'],file['location_name'], file['latitude'],file['longitude'],file['full_text'],file['usertags'],file['like_count'],file['comment_count']],axis=1)
                df_output.dropna(subset=['full_text'])

            df_output.columns = nama_kolom
            df_output.drop(columns=['Text'],inplace=True)
            st.dataframe(df_output, hide_index=True, use_container_width=True)

    
if __name__ == "__main__":
    main()