import ijson
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm

# Tokenizer 및 Pretrained Model 초기화
# BERT는 문제 설명용, CodeBERT는 코드용
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer_codebert = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_codebert = AutoModel.from_pretrained("microsoft/codebert-base")
model_bert.eval()
model_codebert.eval()

# 임베딩 후 projection layer 정의 (논문)
# 모델 임베딩 → 평균 풀링 → proj layer (768 → 256)
proj_dim = 256
proj_bert = nn.Linear(768, proj_dim)
proj_codebert = nn.Linear(768, proj_dim)

# 타입 문자열을 고유한 숫자 ID로 매핑
type2id = {}
type_id_counter = 0

def map_types_to_ids(token2type: dict) -> list:
    """
    {"K": "int", "ad": "long"} → ["0", "1"]
    타입마다 고유 ID 부여 후 숫자 시퀀스로 반환
    """
    global type2id, type_id_counter
    id_list = []
    for token, typename in token2type.items():
        typename = typename.strip()
        if typename not in type2id:
            type2id[typename] = type_id_counter
            type_id_counter += 1
        id_list.append(str(type2id[typename]))
    return id_list

# CosineEmbeddingLoss 사용
loss_fn = nn.CosineEmbeddingLoss()

# 메인 학습 함수
def run_contrastive(json_path: str):
    losses = []

    # json 파일을 streaming 방식으로 읽음
    with open(json_path, "r") as f:
        for problem_name, entry in tqdm(ijson.kvitems(f, ""), desc="Processing problems"):
            solutions = entry.get("solutions", [])
            incorrects = entry.get("incorrect_solutions", [])
            text = problem_name  # 문제명 → BERT 입력

            for sol in solutions:
                for inc in incorrects:
                    #  정답/오답 token2type → 타입 ID 시퀀스로 변환
                    tokens_pos = map_types_to_ids(sol.get("token2type", {}))
                    tokens_neg = map_types_to_ids(inc.get("token2type", {}))
                    if not tokens_pos or not tokens_neg:
                        continue

                    # 타입 ID 시퀀스를 문자열로 변환
                    code_pos = " ".join(tokens_pos)
                    code_neg = " ".join(tokens_neg)

                    # 입력 텐서 생성
                    input_text = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True)
                    input_pos = tokenizer_codebert(code_pos, return_tensors="pt", truncation=True, padding=True)
                    input_neg = tokenizer_codebert(code_neg, return_tensors="pt", truncation=True, padding=True)

                    with torch.no_grad():
                        # BERT → 평균 풀링 → projection
                        emb_text = model_bert(**input_text).last_hidden_state.mean(dim=1)
                        emb_text = proj_bert(emb_text)

                        # CodeBERT → 평균 풀링 → projection
                        emb_pos = model_codebert(**input_pos).last_hidden_state.mean(dim=1)
                        emb_pos = proj_codebert(emb_pos)
                        emb_neg = model_codebert(**input_neg).last_hidden_state.mean(dim=1)
                        emb_neg = proj_codebert(emb_neg)

                    # Positive / Negative 쌍에 대해 loss 계산
                    loss_pos = loss_fn(emb_text, emb_pos, torch.tensor([1.0]))
                    loss_neg = loss_fn(emb_text, emb_neg, torch.tensor([-1.0]))

                    # 평균 loss 기록
                    losses.append(((loss_pos + loss_neg) / 2).item())

    # 결과 출력
    print(f" 평균 Contrastive 손실: {sum(losses)/len(losses):.4f}")
    print(f" 총 타입 수: {len(type2id)}개")
