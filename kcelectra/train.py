from utils.early_stopping import CustomEarlyStopping
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model import create_model
from utils.early_stopping import CustomEarlyStopping
from custom_dataset import CustomDataset
from utils.load_data import load_data

file_path = "./data/final_comment.xlsx"

train_texts, val_texts, train_labels, val_labels = load_data(file_path)

# 모델 학습 설정
# evaluation_strategy가 계속 error

model, tokenizer = create_model()

train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",  # 에폭마다 평가
    # save_strategy="epoch",  # 에폭마다 저장
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    # load_best_model_at_end=True,  # 가장 좋은 모델을 훈련 후 저장
    no_cuda=True,  # CPU 사용
)

# EarlyStoppingCallback 설정 (검증 손실이 개선되지 않으면 학습을 중단)
# early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # callbacks=[early_stopping_callback],  # EarlyStoppingCallback 추가
)

# CustomEarlyStopping 객체 생성
early_stopping = CustomEarlyStopping(patience=2)

# 모델 학습
for epoch in range(training_args.num_train_epochs):
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train()
    
    # 평가 수행 (매 에폭마다)
    eval_result = trainer.evaluate()
    current_loss = eval_result["eval_loss"]
    
    # 조기 종료 체크
    if early_stopping.check(current_loss):
        print("조기 종료 실행!")
        break
