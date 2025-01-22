import random
import time
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW, BertConfig

from tqdm import tqdm
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

import plotly.graph_objects as go
from collections import defaultdict


## -------------------------------------------------
## 1. Data Loading and Preprocessing
## -------------------------------------------------

def load_data(filename):
    """
    파일에서 한 줄씩 읽어 문장과 라벨을 분리하는 함수.
    각 줄은 '문장<TAB>라벨' 형태로 구성됨.

    Args:
        filename (str): 데이터를 로드할 파일의 경로

    Returns:
        tuple:
            - sentences (list): 문장 리스트
            - labels (list): 라벨 리스트
            - int: 문장 수
    """
    sentences = []
    labels = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            sen, lab = line.strip().split('\t')
            sentences.append(sen)
            labels.append(lab)

    return sentences, labels, len(sentences)


def preprocess(original_sentences, new_label):
    """
    특정 토큰('[MASK]')을 새 라벨(여기서는 '起来')로 대체하는 전처리 함수.

    Args:
        original_sentences (list): 원본 문장 리스트
        new_label (str): 대체할 문자열

    Returns:
        list: 전처리된 문장 리스트
    """
    preprocessed_sentences = []
    for sentence in original_sentences:
        preprocessed_sentences.append(sentence.replace('[MASK]', new_label))  # '起来'
    return preprocessed_sentences


def tokenize_and_pad(sentences, tokenizer):
    """
    주어진 문장 리스트를 토크나이징하고, pad_sequences를 통해 길이를 맞추는 함수.

    Args:
        sentences (list): 문장 리스트
        tokenizer (BertTokenizer): BERT 토크나이저 (또는 호환되는 토크나이저)

    Returns:
        tuple:
            - token_ids (list of list): 패딩 적용된 토큰 ID
            - masks (list of list): 어텐션 마스크
    """
    # 토크나이징
    tokenized = [tokenizer.tokenize(sen) for sen in sentences]
    # ID 변환 + 패딩
    token_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(tkn) for tkn in tokenized],
        dtype='long',
        truncating='post',
        padding='post'
    )
    # 어텐션 마스크 생성
    masks = create_attention_masks(token_ids)
    return token_ids, masks


def create_attention_masks(token_ids):
    """
    토큰 ID 배열에서 실제 토큰(0이 아닌 부분)에 대해 1, 0인 부분에 대해서는 0을 부여하는 어텐션 마스크를 생성.

    Args:
        token_ids (list of list): 패딩 처리된 토큰 ID 리스트

    Returns:
        list of list: 어텐션 마스크 리스트
    """
    return [[float(token_id > 0) for token_id in seq] for seq in token_ids]


def create_dataloader(token_ids, masks, labels, batch_size, sampler_type='random'):
    """
    토큰 ID, 마스크, 라벨 텐서를 Dataset으로 묶고,
    Sampler와 함께 DataLoader를 생성하는 함수.

    Args:
        token_ids (list of list): 패딩 완료된 토큰 ID
        masks (list of list): 어텐션 마스크
        labels (list or np.ndarray): 라벨
        batch_size (int): 배치 크기
        sampler_type (str): 'random' or 'sequential' 선택

    Returns:
        DataLoader: 해당 데이터셋을 로드하는 PyTorch DataLoader
    """
    datas_tensor = torch.tensor(token_ids)
    masks_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(datas_tensor, masks_tensor, labels_tensor)

    if sampler_type == 'random':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


## -------------------------------------------------
## 2. Training & Validation Steps
## -------------------------------------------------

def format_time(elapsed):
    """
    걸린 시간을 'hh:mm:ss' 형식으로 변환하여 반환.

    Args:
        elapsed (float): 경과 시간 (초)

    Returns:
        str: "HH:MM:SS" 포맷의 문자열
    """
    return str(time.strftime("%H:%M:%S", time.gmtime(elapsed)))


def flat_accuracy(preds, labels):
    """
    모델 예측값(preds)과 실제 라벨(labels)을 비교하여 정확도를 계산.

    Args:
        preds (numpy.ndarray): 모델의 로짓/확률 예측 (배치 단위)
        labels (numpy.ndarray): 실제 라벨

    Returns:
        float: 예측 정확도 (0.0 ~ 1.0)
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_one_epoch(model, train_dataloader, optimizer, scheduler, device):
    """
    한 에포크 동안 모델을 훈련하는 함수.

    Args:
        model (torch.nn.Module): 훈련할 모델
        train_dataloader (DataLoader): 훈련 데이터 로더
        optimizer (torch.optim.Optimizer): 옵티마이저
        scheduler (transformers.get_linear_schedule_with_warmup): 학습률 스케줄러
        device (torch.device): 모델이 학습될 디바이스 (GPU / CPU)

    Returns:
        float: 에포크당 평균 훈련 손실
    """
    t0 = time.time()
    total_loss = 0.0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and step > 0:
            elapsed = format_time(time.time() - t0)
            print(f'   Batch {step:,} of {len(train_dataloader):,}. Elapsed: {elapsed}.')

        batch_token_ids, batch_mask, batch_labels = [t.to(device) for t in batch]

        outputs = model(batch_token_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss


def validate(model, validation_dataloader, device, validation_sentences=None):
    """
    검증 데이터로 모델을 평가하고 정확도와 임베딩 등을 추출하는 함수.

    Args:
        model (torch.nn.Module): 평가할 모델
        validation_dataloader (DataLoader): 검증 데이터 로더
        device (torch.device): 모델이 올라가 있는 디바이스
        validation_sentences (list, optional): 검증 문장 리스트 (TSNE 시각화용)

    Returns:
        tuple:
            - float: 검증 정확도
            - list: CLS 임베딩 리스트
            - list: 검증 라벨 리스트
            - list: 검증 문장 텍스트 리스트
    """
    t0 = time.time()
    model.eval()

    eval_accuracy = 0
    now_epoch_embedding = []
    now_epoch_label = []
    now_epoch_text = []

    for batch in validation_dataloader:
        batch_token_ids, batch_mask, batch_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(batch_token_ids, attention_mask=batch_mask, output_hidden_states=True)

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        eval_accuracy += flat_accuracy(logits, label_ids)

        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            cls_embeddings = hidden_states[-1][:, 0, :].cpu().numpy()  # [CLS] 토큰 임베딩
            now_epoch_embedding.extend(cls_embeddings)
            now_epoch_label.extend(label_ids)

            # 문장 인덱스만큼 텍스트를 저장 (툴팁에 쓰이도록)
            if validation_sentences:
                # 배치가 순차적으로 들어오므로, 현재 배치 크기에 해당하는 문장 인덱스를 할당
                now_epoch_text.extend(
                    [validation_sentences[idx] for idx in range(len(now_epoch_embedding) - len(cls_embeddings),
                                                                len(now_epoch_embedding))]
                )
            else:
                now_epoch_text.extend([f"Sentence {i}" for i in range(len(label_ids))])
        else:
            print("hidden_states is None. Make sure output_hidden_states=True is set.")

    avg_eval_accuracy = eval_accuracy / len(validation_dataloader)
    return avg_eval_accuracy, now_epoch_embedding, now_epoch_label, now_epoch_text


def compute_tsne(sentence_embeddings_per_epoch):
    """
    각 에포크별 CLS 임베딩에 대해 t-SNE를 수행하여 2차원 좌표로 축소.

    Args:
        sentence_embeddings_per_epoch (list): 에포크별 CLS 임베딩 리스트

    Returns:
        list: 에포크별로 축소된 t-SNE 임베딩 리스트
    """
    tsne_embeddings_per_epoch = []
    for epoch_embeddings in sentence_embeddings_per_epoch:
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(epoch_embeddings)
        tsne_embeddings_per_epoch.append(reduced_embeddings)
    return tsne_embeddings_per_epoch


def plot_animation(tsne_embeddings_per_epoch, all_epochs_labels, validation_sentences,
                   epochs, label_meaning, title="t-SNE Sentence Embeddings by Label with Animation"):
    """
    t-SNE 축소 결과를 Plotly로 시각화하고 에포크별 애니메이션을 구현하는 함수.

    Args:
        tsne_embeddings_per_epoch (list): 에포크별 t-SNE 임베딩 결과
        all_epochs_labels (list): 에포크별 라벨
        validation_sentences (list): 검증 문장 리스트
        epochs (int): 총 에포크 수
        label_meaning (dict): 라벨 번호와 의미를 매핑한 딕셔너리
        title (str): 그래프 제목
    """
    # 라벨별 색상 정의
    label_colors = [
        '#636EFA',  # 라벨 0: 파란색
        '#EF553B',  # 라벨 1: 주황/빨강
        '#00CC96',  # 라벨 2: 녹색
        '#AB63FA',  # 라벨 3: 보라색
        '#FFA15A'   # 라벨 4: 밝은 주황
    ]

    # Plotly 시각화 객체 생성
    fig = go.Figure()

    # === 첫 번째 에포크 시각화 ===
    for label in label_meaning.keys():
        mask = np.array(all_epochs_labels[0]) == label
        scatter = go.Scatter(
            x=tsne_embeddings_per_epoch[0][mask, 0],
            y=tsne_embeddings_per_epoch[0][mask, 1],
            mode='markers',
            marker=dict(size=8, color=label_colors[label], showscale=False),
            text=[validation_sentences[idx] for idx in range(len(tsne_embeddings_per_epoch[0])) if mask[idx]],
            hoverinfo='text',
            name=f"{label_meaning.get(label)}",
            visible=True
        )
        fig.add_trace(scatter)

    # === 애니메이션 프레임 구성 ===
    frames = []
    for epoch in range(epochs):
        frame_data = []
        for label in label_meaning.keys():
            mask = np.array(all_epochs_labels[epoch]) == label
            frame_data.append(go.Scatter(
                x=tsne_embeddings_per_epoch[epoch][mask, 0],
                y=tsne_embeddings_per_epoch[epoch][mask, 1],
                mode='markers',
                marker=dict(size=4, color=label_colors[label], showscale=False),
                text=[validation_sentences[idx] for idx in range(len(tsne_embeddings_per_epoch[epoch])) if mask[idx]],
                hoverinfo='text',
                name=f"{label_meaning.get(label)}",
            ))
        frames.append(go.Frame(data=frame_data, name=f'Epoch {epoch + 1}'))

    sliders = [dict(
        steps=[dict(method='animate',
                    args=[[f'Epoch {k + 1}'],
                          dict(mode='immediate',
                               frame=dict(duration=1500, redraw=True),
                               transition=dict(duration=1000))],
                    label=f'Epoch {k + 1}') for k in range(epochs)],
        active=0,
        transition=dict(duration=1000),
        x=0.1, y=0, len=0.9,
        currentvalue=dict(font=dict(size=20), prefix="Epoch: ", visible=True, xanchor='center')
    )]

    fig.update_layout(
        title=title,
        updatemenus=[dict(type="buttons",
                          showactive=False,
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, dict(frame=dict(duration=1500, redraw=True),
                                                         fromcurrent=True,
                                                         mode='immediate')]),
                                   dict(label="Pause",
                                        method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode='immediate')])
                                   ])],
        sliders=sliders,
        legend_title="Labels",
        showlegend=True,
        xaxis=dict(range=[-50, 50]),
        yaxis=dict(range=[-50, 50])
    )

    fig.frames = frames
    fig.show()


## -------------------------------------------------
## 3. Main Entry Point
## -------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 라벨 의미 정의
    label_meaning = {
        0: 'Directional',  # 방향성 라벨
        1: 'Resultative',  # 결과 라벨
        2: 'Completive',   # 완료 라벨
        3: 'Inchoative',   # 시작 라벨
        4: 'Discourse'     # 담화 라벨
    }

    # 데이터 파일 경로
    train_path = 'qilai_train_data.txt'
    test_path = 'qilai_test_data.txt'

    # 1) 훈련 세트 로드
    train_sentences, train_labels, train_len = load_data(train_path)
    # 2) 테스트 세트 로드
    test_sentences, test_labels, test_len = load_data(test_path)

    # 라벨을 0~4 범위로 변환
    train_labels_to_int = [int(label) - 1 for label in train_labels]
    test_labels_to_int = [int(label) - 1 for label in test_labels]

    # '[MASK]' --> '起来'
    train_sentences_mask_replaced = preprocess(train_sentences, '起来')
    test_sentences_mask_replaced = preprocess(test_sentences, '起来')

    # 훈련 세트를 9:1 비율로 분할 (훈련: 90%, 검증: 10%)
    train_sen_split, val_sen, train_lab_split, val_lab = train_test_split(
        train_sentences_mask_replaced, train_labels_to_int,
        test_size=0.1,
        random_state=42
    )

    # 모델 설정 (MacBERT Large)
    MODEL = 'hfl/chinese-macbert-large'
    tokenizer = BertTokenizer.from_pretrained(MODEL)

    # === 토크나이징 & 패딩 & 어텐션 마스크 생성 (훈련, 검증, 테스트) ===
    train_token_ids, train_masks = tokenize_and_pad(train_sen_split, tokenizer)
    val_token_ids, val_masks = tokenize_and_pad(val_sen, tokenizer)
    test_token_ids, test_masks = tokenize_and_pad(test_sentences_mask_replaced, tokenizer)

    # === DataLoader 생성 ===
    batch_size = 10
    train_dataloader = create_dataloader(train_token_ids, train_masks, train_lab_split, batch_size, sampler_type='random')
    validation_dataloader = create_dataloader(val_token_ids, val_masks, val_lab, batch_size, sampler_type='sequential')
    test_dataloader = create_dataloader(test_token_ids, test_masks, test_labels_to_int, batch_size, sampler_type='sequential')

    # 모델 로드
    model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=5)
    model.to(device)

    # 옵티마이저, 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 10
    total_batch = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_batch
    )

    # 재현성 보장
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 학습 및 검증
    loss_accuracy_metrics = defaultdict(list)
    sentence_embeddings_per_epoch = []
    all_epochs_labels = []
    all_epochs_sentences = []

    # === 메인 루프: 각 Epoch별로 훈련 & 검증 ===
    for epoch_i in range(epochs):
        print(f'\n======= Epoch {epoch_i + 1} / {epochs} =======')
        print("Training...")
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        loss_accuracy_metrics['train_loss'].append(avg_train_loss)
        print(f"  Average training loss: {avg_train_loss:.2f}")

        print("\nRunning Validation...")
        avg_eval_accuracy, now_epoch_embedding, now_epoch_label, now_epoch_text = validate(
            model, validation_dataloader, device, validation_sentences=val_sen
        )
        loss_accuracy_metrics['val_accuracy'].append(avg_eval_accuracy)
        print(f"  Accuracy: {avg_eval_accuracy:.2f}")

        # 현재 에포크의 CLS 임베딩, 라벨, 텍스트 저장
        sentence_embeddings_per_epoch.append(np.array(now_epoch_embedding))
        all_epochs_labels.append(np.array(now_epoch_label))
        all_epochs_sentences.append(now_epoch_text)

        print("Training complete!")

    # t-SNE 수행
    tsne_embeddings_per_epoch = compute_tsne(sentence_embeddings_per_epoch)

    # Plotly로 에포크별 애니메이션 시각화
    plot_animation(
        tsne_embeddings_per_epoch,
        all_epochs_labels,
        val_sen,  # 검증에 사용된 문장 리스트
        epochs,
        label_meaning
    )
