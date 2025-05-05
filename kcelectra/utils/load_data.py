import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(file_path):
    
    abs_path = os.path.abspath(file_path)
    print(f"Trying to load file at: {abs_path}")
    print(f"File exists? {os.path.exists(abs_path)}")
    
    df = pd.read_excel(abs_path)
    
    df = df[['comment', 'label']]  # 'comment'와 레이블이 있는 컬럼만 선택
    df.columns = ['text', 'label']  # 컬럼 이름 변경
    
    df = df.dropna()  # 결측값 제거
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels