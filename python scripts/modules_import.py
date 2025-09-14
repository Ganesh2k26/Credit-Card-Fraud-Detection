import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  # Faster than SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('d:\GANESH\MINI PROJECTS\Document from GANESH.csv')
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['Class'].value_counts())

