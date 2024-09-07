
import numpy as np

# Function to generate synthetic data for events
def generate_synthetic_event_data(num_events, num_features):
    # Randomly generate features for each event
    features = np.random.rand(num_events, num_features)
    # Generate labels (0 for background, 1 for signal)
    labels = np.random.randint(0, 2, num_events)
    return features, labels

num_events = 10000
num_features = 30  # Number of features per event
features, labels = generate_synthetic_event_data(num_events, num_features)
