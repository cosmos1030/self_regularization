import os
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
base_path = 'multiseed_output'
optimizers = {
    'adam': 'alex_batch64_adam_lr0.0001_epochs100',
    'sgd': 'alex_batch64_sgd_lr0.0001_epochs100'
}
fc_files = ['fc1.csv', 'fc2.csv', 'fc3.csv']
save_dir = 'visualization'
os.makedirs(save_dir, exist_ok=True)

# 각 fc 파일마다 작업 수행
for fc_file in fc_files:
    combined_df = pd.DataFrame()

    # adam과 sgd의 데이터프레임 불러와서 레이블 붙이기
    for opt_name, opt_folder in optimizers.items():
        file_path = os.path.join(base_path, opt_folder, fc_file)
        df = pd.read_csv(file_path)
        df['optimizer'] = opt_name
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 'seed' 열 제외한 열에 대해 박스플롯 생성
    for column in combined_df.columns:
        if column in ['seed', 'optimizer']:
            continue

        plt.figure(figsize=(8, 6))
        # boxplot 생성
        combined_df.boxplot(column=column, by='optimizer')
        plt.title(f'{fc_file} - {column} Comparison')
        plt.suptitle('')
        plt.xlabel('Optimizer')
        plt.ylabel(column)
        plt.grid(True)
        
        # 파일 저장
        save_path = os.path.join(save_dir, f'{fc_file.replace(".csv", "")}_{column}_boxplot.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

print(f"Every boxplot has been created and saved in '{save_dir}'.")
