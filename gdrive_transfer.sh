#!/bin/bash

echo "[INFO] Google Drive File Transfer Script Started..."

# 📌 환경 변수 설정
SERVICE_ACCOUNT_JSON="./service_account.json"
GDRIVE_FOLDER_ID="1AFJ6SzuS552YuTCb1bZJtNGh3abb5SvR"
GDRIVE_REMOTE="google_drive"
LOCAL_DIR="./local_files"

# 📌 JSON 파일 존재 확인
if [[ ! -f "$SERVICE_ACCOUNT_JSON" ]]; then
    echo "[ERROR] Service account JSON file not found at: $SERVICE_ACCOUNT_JSON"
    exit 1
fi

# 1️⃣ rclone 설치 확인
if ! command -v rclone &> /dev/null; then
    echo "[INFO] Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
else
    echo "[INFO] rclone is already installed."
fi

# 2️⃣ rclone 설정 파일 생성
echo "[INFO] Configuring rclone..."
mkdir -p ~/.config/rclone
cat <<EOF > ~/.config/rclone/rclone.conf
[$GDRIVE_REMOTE]
type = drive
scope = drive
service_account_file = $SERVICE_ACCOUNT_JSON
root_folder_id = $GDRIVE_FOLDER_ID
EOF

# 3️⃣ 작업 실행
COMMAND=$1
FILE_PATH=$2

# 파일 경로 앞의 ./ 제거
FILE_NAME=$(basename "$FILE_PATH")

case "$COMMAND" in
    upload)
        if [[ -z "$FILE_PATH" ]]; then
            echo "[ERROR] Usage: $0 upload <local_file_or_folder>"
            exit 1
        fi
        if [[ -d "$FILE_PATH" ]]; then
            echo "[INFO] Uploading folder: $FILE_PATH to Google Drive..."
            rclone copy "$FILE_PATH" "$GDRIVE_REMOTE:/$FILE_NAME/" --progress
        else
            echo "[INFO] Uploading file: $FILE_PATH to Google Drive..."
            rclone copy "$FILE_PATH" "$GDRIVE_REMOTE:/"
        fi
        echo "[SUCCESS] Upload completed!"
        ;;
    
    download)
        if [[ -z "$FILE_PATH" ]]; then
            echo "[ERROR] Usage: $0 download <remote_file_or_folder>"
            exit 1
        fi
        echo "[INFO] Checking available files in Google Drive..."
        rclone lsf "$GDRIVE_REMOTE:/" || { echo "[ERROR] Failed to list files!"; exit 1; }

        echo "[INFO] Downloading $FILE_NAME from Google Drive..."
        rclone copy "$GDRIVE_REMOTE:/$FILE_NAME" "$LOCAL_DIR/" --progress || { echo "[ERROR] File/Folder not found in Google Drive!"; exit 1; }
        echo "[SUCCESS] Download completed!"
        ;;
    
    list)
        echo "[INFO] Listing files in Google Drive (Folder ID: $GDRIVE_FOLDER_ID)..."
        rclone lsf "$GDRIVE_REMOTE:/"
        ;;
    
    delete)
        if [[ -z "$FILE_PATH" ]]; then
            echo "[ERROR] Usage: $0 delete <remote_file_or_folder>"
            exit 1
        fi
        echo "[INFO] Deleting $FILE_NAME from Google Drive..."
        rclone delete "$GDRIVE_REMOTE:/$FILE_NAME" || { echo "[ERROR] File not found!"; exit 1; }
        echo "[SUCCESS] File deleted!"
        ;;
    
    *)
        echo "[ERROR] Invalid command! Usage:"
        echo "  $0 upload <local_file_or_folder>   # Upload file or folder to Google Drive"
        echo "  $0 download <remote_file_or_folder> # Download file or folder from Google Drive"
        echo "  $0 list                            # List files in Google Drive"
        echo "  $0 delete <remote_file_or_folder>  # Delete a file or folder from Google Drive"
        exit 1
        ;;
esac
