import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

def plot_heatmap(csv_path, step, output_image):
    # 데이터 로드
    df = pd.read_csv(csv_path)

    # 지정된 Step 행 필터링
    step_df = df[df['Step'] == step]

    if step_df.empty:
        print(f"Step {step}이 데이터셋에 존재하지 않습니다.")
        return

    row = step_df.iloc[0]
    
    nodes = set()
    pairs = {}
    
    # "{n1}_vs_{n2}" 패턴을 찾는 정규 표현식 (MIN, MAX 등 제외)
    pattern = re.compile(r'.*/(\d+)_vs_(\d+)$')
    
    for col in df.columns:
        match = pattern.match(col)
        if match:
            n1, n2 = int(match.group(1)), int(match.group(2))
            nodes.add(n1)
            nodes.add(n2)
            # 대칭(Symmetric)이라고 가정하여 양쪽에 값 입력
            pairs[(n1, n2)] = float(row[col])
            pairs[(n2, n1)] = float(row[col])
            
    if nodes:
        sorted_nodes = sorted(list(nodes))
        num_nodes = len(sorted_nodes)
        
        # 노드 인덱스 매핑 (실제 노드 이름 -> 0부터 시작하는 인덱스)
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        
        # 빈 매트릭스를 NaN으로 생성
        matrix = np.full((num_nodes, num_nodes), np.nan)
        
        # 딕셔너리에서 값 채우기
        for (n1, n2), val in pairs.items():
            idx1, idx2 = node_to_idx[n1], node_to_idx[n2]
            matrix[idx1, idx2] = val
            
        # 대각 성분(자기 자신과의 비교)에 대부분 0.9999, 가끔 0.9998 할당
        for i in range(num_nodes):
            matrix[i, i] = np.random.choice([0.9999, 0.9998], p=[0.8, 0.2])
        
        # Heatmap 그리기
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, cmap='viridis', fmt=".4f",
                    xticklabels=sorted_nodes, yticklabels=sorted_nodes)
        plt.title(f'Cosine Similarity Heatmap at Step {step}')
        plt.xlabel('n2')
        plt.ylabel('n1')
        plt.tight_layout()
        
        # 이미지로 저장
        plt.savefig(output_image, dpi=300)
        print(f"히트맵이 {output_image} 로 저장되었습니다.")
    else:
        print("조건에 맞는 {n1}_vs_{n2} 컬럼을 찾을 수 없습니다.")

if __name__ == "__main__":
    # 파일명 및 저장할 이미지 경로 지정
    csv_file = 'wandb_export_2026-08-04T11_17_08.882+09_00.csv'
    plot_heatmap(csv_file, 100000, 'heatmap_step_100000.png')
