#!/bin/bash
# 백업 파일 생성 및 정리 스크립트

BACKUP_DIR="backup"
PROJECT_ROOT="/Users/pphj19116/Library/CloudStorage/OneDrive-Personal/dev_onedrive/rl"

# 백업 디렉토리가 없으면 생성
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    echo "✅ backup 디렉토리 생성됨"
fi

# 사용법 함수
show_usage() {
    echo "사용법: $0 [옵션] <파일명>"
    echo ""
    echo "옵션:"
    echo "  -c, --create    백업 파일 생성"
    echo "  -m, --move      기존 백업 파일들을 backup 폴더로 이동"
    echo "  -l, --list      현재 백업 파일들 목록 보기"
    echo "  -h, --help      도움말 보기"
    echo ""
    echo "예제:"
    echo "  $0 -c models.py              # models.py의 백업 생성"
    echo "  $0 -m                        # 모든 백업 파일들을 backup 폴더로 이동"
    echo "  $0 -l                        # 백업 파일 목록 보기"
}

# 백업 파일 생성 함수
create_backup() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        echo "❌ 오류: 파일 '$file'이 존재하지 않습니다."
        return 1
    fi
    
    # 백업 파일명 생성 (확장자 고려)
    local filename=$(basename "$file")
    local name="${filename%.*}"
    local ext="${filename##*.}"
    
    if [ "$name" = "$ext" ]; then
        # 확장자가 없는 경우
        local backup_name="${filename}_backup"
    else
        # 확장자가 있는 경우
        local backup_name="${name}_backup.${ext}"
    fi
    
    local backup_path="$BACKUP_DIR/$backup_name"
    
    # 백업 생성
    cp "$file" "$backup_path"
    echo "✅ 백업 생성됨: $file → $backup_path"
}

# 기존 백업 파일들을 backup 폴더로 이동
move_existing_backups() {
    echo "🔍 기존 백업 파일들 검색 중..."
    
    # *_backup* 패턴의 파일들 찾기
    local backup_files=$(find . -maxdepth 1 -name "*backup*" -type f)
    
    if [ -z "$backup_files" ]; then
        echo "📝 이동할 백업 파일이 없습니다."
        return 0
    fi
    
    echo "📦 발견된 백업 파일들:"
    for file in $backup_files; do
        echo "  - $file"
    done
    
    echo ""
    echo "🚚 backup 폴더로 이동 중..."
    
    for file in $backup_files; do
        local filename=$(basename "$file")
        mv "$file" "$BACKUP_DIR/$filename"
        echo "  ✅ $file → $BACKUP_DIR/$filename"
    done
    
    echo "🎉 모든 백업 파일 이동 완료!"
}

# 백업 파일 목록 보기
list_backups() {
    echo "📋 현재 백업 파일 목록:"
    echo ""
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
        echo "  (백업 파일이 없습니다)"
        return 0
    fi
    
    # 백업 파일들 크기와 함께 표시
    echo "  크기      날짜           파일명"
    echo "  ----      ----           ------"
    ls -lah "$BACKUP_DIR" | grep -v "^total" | grep -v "^d" | while read -r line; do
        size=$(echo "$line" | awk '{print $5}')
        date=$(echo "$line" | awk '{print $6, $7, $8}')
        name=$(echo "$line" | awk '{print $9}')
        printf "  %-8s  %-12s  %s\n" "$size" "$date" "$name"
    done
}

# 메인 로직
case "$1" in
    -c|--create)
        if [ -z "$2" ]; then
            echo "❌ 오류: 백업할 파일명을 지정해주세요."
            show_usage
            exit 1
        fi
        create_backup "$2"
        ;;
    -m|--move)
        move_existing_backups
        ;;
    -l|--list)
        list_backups
        ;;
    -h|--help|"")
        show_usage
        ;;
    *)
        echo "❌ 오류: 알 수 없는 옵션 '$1'"
        show_usage
        exit 1
        ;;
esac
