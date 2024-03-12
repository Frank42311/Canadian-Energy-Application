import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# Ensure necessary NLTK downloads
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')


# Load the dataset
def load_dataset(filepath):
    return pd.read_csv(filepath, encoding='utf-8')


# Preprocess a single document
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.isalnum() and not word in stop_words]
    return ' '.join(filtered_text)


# Apply preprocessing to all documents
def preprocess_documents(documents):
    return [preprocess(doc) for doc in documents]


# Build the TF-IDF matrix
def build_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(documents)


# Save data using pickle
def save_data(filepath, data):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


# Main function to orchestrate the processing pipeline
def main():
    download_nltk_data()

    dataset_path = 'data/final.csv'
    local_doc = load_dataset(dataset_path)

    # Initialize documents list
    documents = local_doc['company_summary'].tolist()

    # Add sector summaries and trends
    documents += aggregate_sector_data(local_doc)

    # Preprocess documents
    preprocessed_documents = preprocess_documents(documents)

    # Build TF-IDF matrix and get vectorizer
    vectorizer, tfidf_matrix = build_tfidf_matrix(preprocessed_documents)

    # Prepare data for saving
    data_to_save = {
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'documents': documents  # Include original documents for reference
    }

    # Save data
    save_data('data/tfidf_vectorizer.pkl', data_to_save)
    print("Data successfully saved to data/tfidf_vectorizer.pkl")

# Helper function to aggregate sector summaries and trends
def aggregate_sector_data(local_doc):
    sector_documents = []

    # Add sector summaries by year and sector
    grouped_by_year = local_doc.groupby('year')
    for _, group in grouped_by_year:
        sector_summaries = group.groupby('sector_name')['sector_summary'].first().tolist()
        sector_documents.extend(sector_summaries)

    # Add first sector trend for each sector, independent of year
    sector_trends = local_doc.groupby('sector_name')['sector_trend'].first().tolist()
    sector_documents.extend(sector_trends)

    return sector_documents


if __name__ == '__main__':
    main()
