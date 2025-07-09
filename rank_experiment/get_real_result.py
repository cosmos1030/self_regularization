import numpy as np
import matplotlib.pyplot as plt
import os

# --- 사용자 설정 ---

# 1. 정규화된 정확도 CSV 파일이 저장된 디렉토리 경로를 지정하세요.
# 예: "runs/alex_seed100_..._recon_expt"
csv_directory_path = "/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/rank_experiment/experiment_regularized_svd" 

# 2. 각 시나리오의 Full Rank (최대 랭크) 테스트 정확도를 입력하세요.
# 제공해주신 값을 기반으로 작성되었습니다.
full_rank_accuracies = {
    "0 CNN Layers": 0.5250,
    "1 CNN Layer":  0.6523,
    "2 CNN Layers": 0.7082,
    "3 CNN Layers": 0.7268,
}

# 3. 각 시나리오에 해당하는 CSV 파일 이름을 매핑하세요.
#    이전 코드의 명명 규칙을 따르는 경우 'recon_{scenario_name}_acc_norm_epoch{epoch}.csv' 형식이 됩니다.
#    아래는 예시이며, 실제 파일 이름으로 수정해야 합니다.
#    'cnn_0', 'cnn_1' 등 파일 이름에 사용된 키워드로 수정하세요.
scenario_to_filename = {
    # 예시: "0 CNN Layers" 시나리오의 파일 이름이 'recon_cnn_0_acc_norm.csv'인 경우
    "0 CNN Layers": "0_CNN_Layers_norm_acc_regularized.csv",
    "1 CNN Layer":  "1_CNN_Layer_norm_acc_regularized.csv",
    "2 CNN Layers":"2_CNN_Layers_norm_acc_regularized.csv",
    "3 CNN Layers":"3_CNN_Layers_norm_acc_regularized.csv",
}

# 4. 저장될 그래프 파일의 이름을 지정하세요.
output_plot_filename = "reconstruction_raw_accuracy.png"

# --- 코드 실행 ---

print("실제 정확도 그래프 생성을 시작합니다.")

# 그래프 준비
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# 각 시나리오에 대해 그래프 그리기
for scenario_name, baseline_accuracy in full_rank_accuracies.items():
    
    # 시나리오에 해당하는 파일 이름 가져오기
    if scenario_name not in scenario_to_filename:
        print(f"경고: '{scenario_name}'에 대한 파일 이름이 'scenario_to_filename' 딕셔너리에 없습니다. 건너뜁니다.")
        continue
    
    csv_filename = scenario_to_filename[scenario_name]
    file_path = os.path.join(csv_directory_path, csv_filename)

    # CSV 파일 불러오기
    try:
        # 정규화된 정확도 데이터를 로드
        normalized_data = np.loadtxt(file_path)
        
        # 실제 정확도로 변환 (Un-normalization)
        raw_accuracy_data = normalized_data * baseline_accuracy
        
        # x축 데이터 생성 (랭크 k = 1, 2, 3, ...)
        ranks = np.arange(1, len(raw_accuracy_data) + 1)
        
        # 그래프에 플롯
        ax.plot(ranks, raw_accuracy_data, marker='o', linestyle='-', markersize=4, label=f'{scenario_name} (Full Rank Acc: {baseline_accuracy:.4f})')
        print(f"'{scenario_name}' 시나리오의 데이터를 처리했습니다.")

    except FileNotFoundError:
        print(f"경고: 파일 '{file_path}'을(를) 찾을 수 없습니다. 이 시나리오는 건너뜁니다.")
    except Exception as e:
        print(f"오류: '{file_path}' 처리 중 오류 발생: {e}")

# 그래프 속성 설정
ax.set_title('Reconstruction Raw Accuracy vs. Number of Components', fontsize=16)
ax.set_xlabel('Number of Singular Components (Rank k)', fontsize=12)
ax.set_ylabel('Raw Test Accuracy', fontsize=12)
ax.legend(title="Scenarios")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()

# 그래프 저장 및 출력
try:
    plt.savefig(output_plot_filename)
    print(f"\n그래프를 '{output_plot_filename}' 파일로 성공적으로 저장했습니다.")
except Exception as e:
    print(f"\n오류: 그래프 저장 실패: {e}")

plt.show()