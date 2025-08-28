import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    data['Employment_Type'] = LabelEncoder().fit_transform(data['Employment_Type'])
    data['Approval_Status'] = data['Approval_Status'].map({'Approved': 1, 'Rejected': 0})
    data = data.drop('Application_ID', axis=1)
    features = data.drop('Approval_Status', axis=1)
    labels = data['Approval_Status'].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels

def split_clients(features, labels, n_clients=4):
    split_size = len(features) // n_clients
    client_data = []
    for i in range(n_clients):
        start = i * split_size
        end = (i+1) * split_size if i < n_clients-1 else len(features)
        client_data.append((features[start:end], labels[start:end]))
    return client_data
