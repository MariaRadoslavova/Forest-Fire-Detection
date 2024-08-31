import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

def plot_class_distribution(annotations):
    class_counts = Counter(ann['category_id'] for ann in annotations)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Category ID')
    plt.ylabel('Number of Annotations')
    plt.title('Class Distribution')
    plt.show()

def generate_statistics_report(annotations):
    df = pd.DataFrame(annotations)
    df['width'] = df['bbox'].apply(lambda bbox: bbox[2])
    df['height'] = df['bbox'].apply(lambda bbox: bbox[3])
    df['aspect_ratio'] = df['width'] / df['height']
    print(df[['width', 'height', 'aspect_ratio']].describe())
    df[['width', 'height']].plot(kind='hist', bins=30, alpha=0.5)
    plt.title('Bounding Box Size Distribution')
    plt.show()
