# utils/early_stopping.py
import numpy as np

class CustomEarlyStopping:
    def __init__(self, patience=2, threshold=0.01):
        self.patience = patience  # 개선이 없는 에폭 수
        self.threshold = threshold  # 성능 개선 기준 (예: loss 변화)
        self.best_loss = np.inf  # 최상의 손실 값
        self.counter = 0  # 개선되지 않은 에폭 수

    def check(self, current_loss):
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.counter = 0
            return False  # 학습 계속 진행
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 학습 종료
            return False  # 학습 계속 진행
