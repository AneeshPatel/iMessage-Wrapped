import os
import random
import re
import sqlite3
import string
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from textblob import TextBlob
import emoji
from collections import Counter
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import nltk
from nltk.util import ngrams
from typedstream.stream import TypedStreamReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st
import pytz

# Table of Contents
class Toc:
    def __init__(self):
        self._items = []
        self._expander = st.sidebar.expander("Contents", expanded=False)
        self._placeholder = self._expander.empty()

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
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

# Ensure that the NLTK stopwords are downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))

# Function to decode messages as before
def decode_message_attributedbody(data):
    if not data:
        return None
    for event in TypedStreamReader.from_data(data):
        if type(event) is bytes:
            return event.decode("utf-8")

# Fetch messages based on phone number and cache the result
@st.cache_data(show_spinner=False)
def fetch_messages(phone_number):
    query = f"""
        SELECT * 
        FROM chat 
        JOIN chat_message_join 
            ON chat.ROWID = chat_message_join.chat_id 
        JOIN message 
            ON chat_message_join.message_id = message.ROWID
        WHERE chat.chat_identifier = "{phone_number}"
        """
    db_path = os.path.expanduser("~/Library/Messages/chat.db")
    with sqlite3.connect(db_path) as connection:
        messages_df = pd.read_sql_query(
            sql=query,
            con=connection,
            parse_dates={"datetime": "ISO8601"},
        )
    
    messages_df["text"] = messages_df["text"].fillna(
        messages_df["attributedBody"].apply(decode_message_attributedbody)
    )
    messages_df = messages_df[messages_df["text"] != ""]
    messages_df = messages_df[messages_df["text"].notna()]
    return messages_df

# Function to apply date range on the data
def filter_date(messages_df, start_date, end_date, timezone):
    # Ensure message_date is in datetime format before performing operations
    st.info(messages_df['message_date'].dtype)
    if not pd.api.types.is_datetime64_ns_dtype(messages_df['message_date']):
        offset = 978307200  # Seconds between 1970-01-01 and 2001-01-01
        messages_df['message_date'] = pd.to_datetime(
            messages_df['message_date'] / 1000000000 + offset, unit='s'
        )
        messages_df['message_date'] = messages_df['message_date'].dt.tz_localize('UTC', ambiguous='NaT')
    
    # Localize and convert the timezone for message_date column
    messages_df['message_date'] = messages_df['message_date'].dt.tz_convert(timezone)
    
    # Convert start and end dates to pandas datetime objects
    start_date = pd.to_datetime(start_date).tz_localize(timezone, ambiguous='NaT')
    end_date = pd.to_datetime(end_date).tz_localize(timezone, ambiguous='NaT')

    # Filter messages based on the selected date range
    messages_df = messages_df[(messages_df['message_date'] >= start_date) & 
                              (messages_df['message_date'] <= end_date)]
    
    return messages_df

# Function to remove stopwords from the data
def filter_stopwords(messages_df, stopwords_list):
    # Remove stopwords, while handling None or NaN values
    if stopwords_list:
        def clean_text(text):
            if pd.isna(text):
                return ""
            cleaned_text = ' '.join([word for word in text.split() if word.lower() not in stopwords_list])
            return cleaned_text.strip()  # Ensure leading/trailing spaces are removed
        
        messages_df['text'] = messages_df['text'].apply(clean_text)
        
        # Skip rows with empty text
        messages_df = messages_df[messages_df['text'].str.strip() != ""]

    return messages_df

# Function to analyze message count over time
def analyze_message_count(messages_df, placeholder):
    message_count = messages_df.groupby(messages_df['message_date'].dt.date).size()
    with placeholder.container():
        toc.subheader("Message Stats")
        st.write("Total Messages:", message_count.sum())
        st.write("Average Messages Per Day:", message_count.sum() / len(message_count))
        st.write("Highest in a Day:", message_count.max(), "on", message_count.idxmax())
        st.write("Lowest in a Day:", message_count.min(), "on", message_count.idxmin())

        # Interactive line plot
        toc.subheader("Message Frequency Over Time")
        fig = px.line(x=message_count.index, y=message_count.values, labels={'x': 'Date', 'y': 'Messages Sent'})
        st.plotly_chart(fig)

def remove_outliers(arr, scale=1.5):
    # Calculate the IQR for outlier detection
    Q1 = arr.quantile(0.25)
    Q3 = arr.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - scale * IQR
    upper_bound = Q3 + scale * IQR
    
    # Filter out outliers
    return arr[(arr >= lower_bound) & (arr <= upper_bound)]

# Function for message length analysis with outlier removal
def analyze_message_lengths(messages_df, placeholder):
    # Calculate message lengths
    message_lengths = messages_df['text'].apply(len)
    filtered_message_lengths = remove_outliers(message_lengths)
    
    with placeholder.container():
        toc.subheader("Message Length Stats")
        st.write("Average Message Length:", message_lengths.mean())
        st.write("Longest Message Length:", message_lengths.max())
        st.write("Shortest Message Length:", message_lengths.min())        

        # Interactive histogram for message lengths, excluding outliers
        toc.subheader("Message Length Distribution (Excluding Outliers)")
        fig = px.histogram(filtered_message_lengths, labels={'value': 'Message Length'})
        fig.update_traces(marker_line_width=0.3,marker_line_color="black")
        st.plotly_chart(fig)

# Analyze participant frequencies (i.e., message counts per participant)
def analyze_participant_frequency(messages_df, placeholder):
    # Count the occurrences of each participant's name based on is_from_me
    participant_counts = messages_df['is_from_me'].value_counts()

    # Create a dictionary for participant labels
    participant_labels = {1: "You", 0: "Other Participant"}
    
    # Rename index labels based on is_from_me values (True = "You", False = "Other Participant")
    participant_counts.index = participant_counts.index.map(participant_labels)

    with placeholder.container():
        # Interactive pie chart for participant frequency
        toc.subheader("Participant Message Distribution")
        fig = px.pie(
            names=participant_counts.index, 
            values=participant_counts.values,
            labels={'names': 'Participant', 'values': 'Messages Sent'},
            title=' ',
            color=participant_counts.index,  # Color by participant
            hole=0.4,  # Make the chart a donut
        )

        # Enhance the hover information and text display
        fig.update_traces(
            hoverinfo='label+percent+value',  # Show label, percentage, and value on hover
            textinfo='label+percent+value',  # Show percentage and value inside slices
            textposition='inside',  # Position text inside the slices
            textfont_size=15,
        )

        # Customize layout (optional)
        fig.update_layout(
            title_font_size=24,  # Bigger title font
            title_x=0.5,  # Center title
            showlegend=False,  # Hide legend (self-explanatory with labels)
            margin=dict(t=50, b=50, l=50, r=50)  # Adjust margins for better spacing
        )

        st.plotly_chart(fig)

# Analyzes response times for both you and the other person
def analyze_response_times(messages_df, placeholder):
    # Sort messages by timestamp to ensure they're in order
    messages_df = messages_df.sort_values(by='message_date')

    # Calculate the time differences between consecutive messages
    messages_df['response_time'] = messages_df['message_date'].diff().shift(-1)

    # Separate response times into two categories
    my_response_times = []
    other_response_times = []
    
    for i in range(1, len(messages_df)):
        # If the sender switches, record the response time
        if messages_df['is_from_me'].iloc[i] != messages_df['is_from_me'].iloc[i - 1]:
            response_time = messages_df['response_time'].iloc[i - 1].total_seconds()

            # Check who sent the first message in the pair
            if messages_df['is_from_me'].iloc[i - 1]:  # If the previous message was from me
                other_response_times.append(response_time)
            else:  # If the previous message was from the other person
                my_response_times.append(response_time)

    my_response_times = pd.Series(my_response_times)
    other_response_times = pd.Series(other_response_times)
        
    # Remove outliers
    filtered_my_response_times = remove_outliers(my_response_times)
    filtered_other_response_times = remove_outliers(other_response_times)
    
    with placeholder.container():
        toc.subheader("My Response Time Stats")
        st.write("My Average Response Time:", my_response_times.mean() / 60, "minutes")
        st.write("My Longest Response Time:", my_response_times.max() / 60 / 60, "hours")
        st.write("My Shortest Response Time:", my_response_times.min(), "seconds")

        toc.subheader("Other Person's Response Time Stats")
        st.write("Other Person's Average Response Time:", other_response_times.mean() / 60, "minutes")
        st.write("Other Person's Longest Response Time:", other_response_times.max() / 60 / 60, "hours")
        st.write("Other Person's Shortest Response Time:", other_response_times.min(), "seconds")

        # Interactive histogram for message lengths, excluding outliers
        toc.subheader("Response Time Distribution (Excluding Outliers)")

        fig = ff.create_distplot([filtered_my_response_times, filtered_other_response_times], ["My Responses", "Other Person's Responses"], show_hist=False, show_rug=False)
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Density"
        )

        st.plotly_chart(fig)

# Analyzes time of day of messages
def analyze_time_of_day(messages_df, placeholder):
    # Extract the hour from the local time (you can also extract minute or second if you want more granularity)
    messages_df['hour'] = messages_df['message_date'].dt.hour

    # Count the number of messages per hour
    message_counts_per_hour = messages_df['hour'].value_counts().sort_index().reset_index()
    message_counts_per_hour.columns = ['Hour', 'Message Count']

    # Plot the distribution using Matplotlib (or Plotly if preferred)
    with placeholder.container():
        st.subheader("Message Distribution Throughout the Day")
        
        fig = px.bar(
            message_counts_per_hour, 
            x='Hour', 
            y='Message Count',
            labels={'Hour': 'Hour of the Day', 'Message Count': 'Number of Messages'},
            template='plotly',
            color='Hour',  # Color by hour for a nice gradient effect
            color_continuous_scale='Viridis'  # Use a color scale for the hour values
        )

        # Customize the layout for better visualization
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),  # Show every hour
            xaxis_title="Hour of the Day",
            yaxis_title="Number of Messages",
            showlegend=False,  # No need to show legend
            title_x=0.5,  # Center the title
            title_font_size=20,
            title_font_family="Arial",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

# Function to create a word-cloud-like visualization
def generate_word_cloud(messages_df, n_words, stopwords_list, placeholder):
    # Step 1: Tokenize text and clean up the words
    all_text = ' '.join(messages_df['text'].dropna())  # Concatenate all messages into a single string
    all_text = all_text.lower()  # Convert to lowercase
    
    # Remove punctuation
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 2: Split into words and remove stopwords
    words = all_text.split()
    words = [word for word in words if word not in stopwords_list]
    
    # Step 3: Count word frequencies
    word_counts = Counter(words)

    # Step 4: Create a word cloud-like plot using matplotlib
    with placeholder.container():
        # Get the most common words and their frequencies
        most_common_words = word_counts.most_common(n_words)

        # Prepare data for plotting
        words, counts = zip(*most_common_words)

        # Normalize the word frequencies to determine font sizes
        max_count = max(counts)
        font_sizes = [100 * (count / max_count) for count in counts]

        # Create the plot with random word positions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')  # Hide the axes

        # Set the plot limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Set the background to transparent
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')

        # Randomly place words on the plot
        positions = []
        for i, word in enumerate(words):
            placed = False
            while not placed:
                # Generate random coordinates for each word
                x = random.uniform(0, 1)
                y = random.uniform(0, 1)

                # Check if the position is too close to others
                too_close = any(abs(x - px) < 0.1 and abs(y - py) < 0.1 for px, py in positions)
                if not too_close:
                    positions.append((x, y))
                    ax.text(x, y, word, ha='center', va='center', fontsize=font_sizes[i], color='white', alpha=0.9, fontweight='bold')
                    placed = True

        # Display the word cloud-like plot
        st.pyplot(fig)

# Function to calculate sentiment
def calculate_sentiment(text):
    if pd.isna(text):
        return None, None  # In case of missing text
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Function to analyze sentiment
def analyze_sentiment(messages_df, placeholder):
    messages_df['sentiment_polarity'], messages_df['sentiment_subjectivity'] = zip(*messages_df['text'].apply(calculate_sentiment))
    with placeholder.container():
        toc.subheader("Sentiment Stats")
        st.write("Average Sentiment Polarity:", messages_df['sentiment_polarity'].mean())
        st.write("Average Sentiment Subjectivity:", messages_df['sentiment_subjectivity'].mean())

        # Interactive histogram for sentiment
        toc.subheader("Sentiment Polarity Distribution")
        fig = px.histogram(messages_df, x='sentiment_polarity', nbins=30,
                        labels={'sentiment_polarity': 'Polarity (Positive to Negative)'})
        # fig.update_layout(yaxis_type="log")
        st.plotly_chart(fig)

        st.write("""
        **Polarity Description**:
        - **Positive**: Indicates a happy or good sentiment.
        - **Negative**: Indicates a sad or bad sentiment.
        - **Neutral**: Indicates a neutral or indifferent sentiment.
        """)

# Extract and count emojis
def extract_emojis(text):
    if not text:
        return ''
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

# Analyze emojis
def analyze_emojis(messages_df, placeholder):
    messages_df['emojis'] = messages_df['text'].apply(extract_emojis)
    all_emojis = [emoji for sublist in messages_df['emojis'] for emoji in sublist]
    emoji_count = Counter(all_emojis)

    top_emojis = emoji_count.most_common(10)
    with placeholder.container():
        if top_emojis:
            top_emoji_labels, top_emoji_values = zip(*top_emojis)

            # Create a plotly bar chart for emojis
            fig = px.bar(x=top_emoji_labels, y=top_emoji_values, labels={'x': 'Emojis', 'y': 'Frequency'})
            st.plotly_chart(fig)
        else:
            st.write("No emojis found.")

# Function to extract n-grams (phrases)
def get_ngrams(text, n=2, words_to_exclude=[]):
    if pd.isna(text):
        return []
    words = text.split()
    words = [word.lower() for word in words if len(word) > 2 and word.lower() not in words_to_exclude]
    return list(ngrams(words, n))

# Function to analyze top phrases (bigrams, trigrams, etc.)
def analyze_top_phrases(messages_df, words_to_exclude, phrase_size, placeholder):
    top_phrases = extract_top_phrases(messages_df, n=phrase_size, words_to_exclude=words_to_exclude, top_n=10)
    
    top_phrase_labels = [' '.join(phrase) for phrase, _ in top_phrases]
    top_phrase_values = [count for _, count in top_phrases]
    
    fig = px.bar(x=top_phrase_labels, y=top_phrase_values, labels={'x': 'Phrases', 'y': 'Frequency'})
    placeholder.plotly_chart(fig)

# Function to extract and count top phrases (bigrams, trigrams)
def extract_top_phrases(messages_df, n=2, words_to_exclude=[], top_n=10):
    all_phrases = []
    for text in messages_df['text']:
        all_phrases.extend(get_ngrams(text, n, words_to_exclude))
    
    phrase_count = Counter(all_phrases)
    return phrase_count.most_common(top_n)

# Function to determine the optimal number of clusters using the Elbow Method
def optimal_clusters(X):
    inertias = []
    for i in range(1, 11):  # Try clusters from 1 to 10
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Calculate the differences in inertia values
    inertia_diff = np.diff(inertias)  # First derivative (difference between consecutive inertias)
    inertia_diff2 = np.diff(inertia_diff)  # Second derivative (change in inertia difference)

    # Find the index of the "elbow" by looking for the largest change in the second derivative
    elbow_point = np.argmax(inertia_diff2) + 2  # Adding 2 because diff2 is one step ahead
    
    # Plot the inertia curve and mark the elbow point
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertias, marker='o', label="Inertia")
    plt.axvline(x=elbow_point, linestyle='--', color='r', label=f"Elbow Point ({elbow_point} clusters)")
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.legend()
    st.pyplot(fig)

    return elbow_point  # Return the number of clusters corresponding to the elbow point

# Function to generate more complex titles for clusters based on TextRank
def generate_cluster_titles(messages_df, n_clusters, X, kmeans, top_n_words=5):
    cluster_titles = []

    # For each cluster, extract the texts belonging to it
    for cluster_idx in range(n_clusters):
        # Get the messages in the current cluster
        cluster_messages = messages_df[messages_df['cluster'] == cluster_idx]
        texts = cluster_messages['text'].tolist()
        texts = [text for text in texts if text.strip()]  # Remove empty or blank strings

        # Get the most common words (remove stopwords if needed)
        word_counter = Counter(texts)
        common_words = word_counter.most_common(top_n_words)

        # Form a title based on the top words
        title = " | ".join([word for word, _ in common_words])
        cluster_titles.append(title)

    return cluster_titles

# Function to perform topic modeling and plot clusters
def topic_modeling(messages_df, stopwords_list, n_clusters=3, pca_components=2, placeholder=None):
    # Ensure that there are no empty messages before vectorizing
    messages_df = messages_df[messages_df['text'].str.strip() != ""]

    # Vectorize the messages using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords_list, max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(messages_df['text'])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Add cluster labels to DataFrame
    messages_df['cluster'] = kmeans.labels_

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=pca_components)
    reduced_X = pca.fit_transform(X.toarray())

    if pca_components == 3:
        fig = px.scatter_3d(
            x=reduced_X[:, 0],
            y=reduced_X[:, 1],
            z=reduced_X[:, 2],
            color=labels.astype(str),  # Color by cluster label
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'},
        )
        
        # Add cluster centers to the 3D plot
        centers = pca.transform(kmeans.cluster_centers_)
        fig.add_scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode='markers',
            marker=dict(size=5, color='white', symbol='x'),
            name='Cluster Centers'
        )
        
        # Update layout to increase figure size
        fig.update_layout(
            width=450,  # Width of the figure
            height=450,  # Height of the figure
            margin=dict(l=0, r=0, b=0, t=40),  # Adjust the margins for better visibility
        )

        # Display the 3D plot in Streamlit
        placeholder.plotly_chart(fig)

    else:
        # If PCA componens are not 3, fall back to 2D visualization
        fig = px.scatter(
            x=reduced_X[:, 0],
            y=reduced_X[:, 1],
            color=labels.astype(str),  # Color by cluster label
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        )

        # Add cluster centers to the plot
        centers = pca.transform(kmeans.cluster_centers_)
        fig.add_scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(size=10, color='white', symbol='x'),
            name='Cluster Centers'
        )

        # Update layout to increase figure size
        fig.update_layout(
            width=500,  # Width of the figure
            height=500,  # Height of the figure
            margin=dict(l=0, r=0, b=0, t=40),  # Adjust the margins for better visibility
        )

        # Display the 3D plot in Streamlit
        placeholder.plotly_chart(fig)

    # Explanation of clusters
    st.write("""
    The above plot shows how the messages are grouped into different topics (clusters). Each point represents a message,
    and the color indicates the cluster to which it belongs. The white 'X' markers represent the central point of each topic.
    
    - **Cluster Centers**: The white 'X' markers represent the centroids of each cluster. These are the average positions of all messages in a particular topic.
    - **Interactive Visualization**: You can rotate and zoom into the plot to explore the clusters in more detail.
    """)

    cluster_titles = generate_cluster_titles(messages_df, n_clusters, X, kmeans)
    st.write("### Top Keywords for Each Topic")
    for idx, title in enumerate(cluster_titles):
        st.write(f"**Topic {idx + 1}:** {title}")
    

# Streamlit UI
st.title("iMessage Wrapped")

st.header("Introduction")
st.write("""
Welcome to the **iMessage Wrapped** app! This tool allows you to analyze and visualize your iMessage conversations in fun and insightful ways. 

The idea for this app started as a project to replicate the popular concept of **Spotify Wrapped** but for iMessage. Inspired by how users get an overview of their listening habits, the goal was to provide an overview of texting habits for a specific year. 

Initially, the app was designed to pull and analyze iMessage data for a given year, but as the project evolved, I added more detailed analyses, such as message frequency, sentiment analysis, emoji usage, and topic modeling for any specified time range. The result is a tool that gives you a deep dive into your messaging history.

### Key Features:
- **Message Count Analysis**: Track how many messages you sent and received over time.
- **Message Length Analysis**: Analyze the average length of your messages.
- **Participant Frequency**: See who texts the most.
- **Response Time Analysis**: Analyze how quickly you and the other person respond to messages.
- **Time of Day Analysis**: Visualize how your message activity is distributed throughout the day.
- **Word Cloud**: Visualize the most common words in your messages.
- **Sentiment Analysis**: Gauge the overall sentiment of your messages.
- **Emoji Analysis**: Discover your most-used emojis.
- **Phrase (n-grams) Analysis**: Find the most common phrases in your messages.
- **Topic Modeling**: Analyze topics and themes in your messages using unsupervised machine learning.

### Privacy Assurance:
This app **does not store or transmit any data**. Everything runs locally on your machine. No data is sent to the cloud, and your messages remain private. You can analyze your iMessage history offline, ensuring full privacy. All data is stored in memory temporarily during analysis and is cleared after the session ends.
""")

# Inputs for the Streamlit app
phone_number = st.text_input("Phone Number", "+19999999999")

# Button to fetch data
fetch_data_button = st.button("Fetch Data")

# Initialize session state variables if not already present
if 'messages_df' not in st.session_state:
    st.session_state['messages_df'] = None

# Fetch data if the phone number is provided
if fetch_data_button and phone_number:
    # Check if the data for the specific phone number exists in session state
    if phone_number not in st.session_state:
        with st.spinner("Fetching data, please wait..."):
            messages_df = fetch_messages(phone_number)
            st.session_state[phone_number] = messages_df  # Store the data in session state for this phone number
            st.session_state['messages_df'] = messages_df
            st.success("Data fetched successfully!")
            data_obtained = True
    else:
        st.session_state['messages_df'] = st.session_state[phone_number]  # Retrieve the cached data from session state

# Filters for analysis
if st.session_state['messages_df'] is not None and phone_number in st.session_state:
    messages_df = st.session_state['messages_df']  # Retrieve messages_df from session state
    
    # Table of Contents Section
    st.sidebar.header("Table of Contents")
    toc = Toc()

    # Filters Section
    st.sidebar.header("Filters")
    # Create a searchable dropdown using st.selectbox
    default_timezone = "US/Pacific"
    timezone = st.sidebar.selectbox(
        "Select Your Timezone",
        options=pytz.common_timezones,
        index=pytz.common_timezones.index(default_timezone),
        help="Select your timezone from the dropdown"
    )

    # Display the selected timezone
    st.write(f"Selected timezone: {timezone}")

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
    stopwords_input = st.sidebar.text_area("Words to Exclude (comma-separated)", ", ".join(nltk_stopwords), height=170)
    stopwords_list = [word.strip().lower() for word in stopwords_input.split(",")]

    date_filtered_df = filter_date(messages_df, start_date, end_date, timezone)
    stopwords_filtered_df = filter_stopwords(date_filtered_df, stopwords_list)

    # --- Message Count Analysis ---
    toc.header("1. Message Count Analysis")
    message_count_placeholder = st.empty()  # Placeholder for message count analysis
    # if st.button("Run Message Count Analysis"):
    with st.spinner("Analyzing message count..."):
        analyze_message_count(date_filtered_df, message_count_placeholder)
    toc.generate()

    # --- Message Length Analysis ---
    toc.header("2. Message Length Analysis")
    message_length_placeholder = st.empty()  # Placeholder for message length analysis
    with st.spinner("Analyzing message lengths..."):
        analyze_message_lengths(date_filtered_df, message_length_placeholder)
    toc.generate()

    # --- Participant Frequency Analysis ---
    toc.header("3. Participant Frequency Analysis")
    participant_frequency_placeholder = st.empty()  # Placeholder for participant frequency analysis
    with st.spinner("Analyzing participant frequencies..."):
        analyze_participant_frequency(date_filtered_df, participant_frequency_placeholder)
    toc.generate()

    # --- Response Time Analysis ---
    toc.header("4. Response Time Analysis")
    response_time_placeholder = st.empty()  # Placeholder for response time analysis
    with st.spinner("Analyzing response times..."):
        analyze_response_times(date_filtered_df, response_time_placeholder)
    toc.generate()

    # --- Time of Day Analysis ---
    toc.header("5. Time of Day Analysis")
    time_of_day_placeholder = st.empty()  # Placeholder for time of day analysis
    with st.spinner("Analyzing time distribution..."):
        analyze_time_of_day(date_filtered_df, time_of_day_placeholder)
    toc.generate()

    # --- Word Cloud ---
    toc.header("6. Word Cloud")
    toc.subheader("Word Cloud Visualization")
    n_words = st.slider(
        "Choose Number of Words", 
        min_value=5, 
        max_value=100, 
        value=40,  # Default to 40 words
        step=5
    )
    word_cloud_placeholder = st.empty()  # Placeholder for word cloud
    with st.spinner("Analyzing texts..."):
        generate_word_cloud(stopwords_filtered_df, n_words, stopwords_list, word_cloud_placeholder)
    toc.generate()

    # --- Sentiment Analysis ---
    toc.header("7. Sentiment Analysis")
    sentiment_placeholder = st.empty()  # Placeholder for sentiment analysis
    with st.spinner("Analyzing sentiment..."):
        analyze_sentiment(stopwords_filtered_df, sentiment_placeholder)
    toc.generate()

    # --- Emoji Analysis ---
    toc.header("8. Emojis Analysis")
    toc.subheader("Top 10 Emojis Sent")
    emoji_placeholder = st.empty()  # Placeholder for emoji analysis
    # if st.button("Run Emoji Analysis"):
    with st.spinner("Analyzing emojis..."):
        analyze_emojis(date_filtered_df, emoji_placeholder)
    toc.generate()

    # --- Phrase (n-grams) Analysis ---
    toc.header("9. Top Phrases Analysis")
    toc.subheader(f"Top 10 Phrases")
    phrase_size = st.slider(
        "Choose Phase Size (1 to 10 words)", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1
    )

    # Placeholder for n-grams analysis
    phrase_placeholder = st.empty()

    # Run n-gram analysis immediately as the slider changes
    with st.spinner("Analyzing top phrases, please wait..."):
        phrase_result = analyze_top_phrases(stopwords_filtered_df, stopwords_list, phrase_size, phrase_placeholder)
    toc.generate()

    # --- Topic Modeling ---
    toc.header("10. Topic Modeling")
    toc.subheader(f"Topic Graph")
    n_clusters = st.slider(
        "Choose Number of Clusters", 
        min_value=1, 
        max_value=10, 
        value=4,  # Default to 4 clusters
        step=1
    )

    # Radio button to choose number of PCA components (2 or 3)
    pca_components = st.radio(
        "Number of PCA Components", 
        options=[2, 3], 
        index=1  # Default to 3 components
    )

    # Placeholder for topic modeling results
    topic_placeholder = st.empty()

    # Run Topic Modeling immediately as the slider changes
    with st.spinner("Analyzing topics, please wait..."):
        topic_result = topic_modeling(stopwords_filtered_df, stopwords_list, n_clusters=n_clusters, pca_components=pca_components, placeholder=topic_placeholder)
    toc.generate()
