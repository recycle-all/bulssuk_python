# 베이스 이미지
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app
RUN apt-get update && apt-get install -y python3-pip && pip3 install --upgrade pip
# .DS_Store 삭제 명령어 추가 (호스트에서 빌드 시 삭제)
RUN find . -name ".DS_Store" -delete || true

# 필요한 파일들을 컨테이너로 복사
COPY main/ /app/main/
COPY font/ /app/font/
COPY image/ /app/image/

# 의존성 설치를 위해 requirements.txt 복사 (필요한 경우)
COPY main/requirements.txt /app/

# Python 패키지 설치
RUN pip install --no-cache-dir -r /app/requirements.txt

# 컨테이너에서 사용할 포트를 명시적으로 선언
EXPOSE 5002

# 실행할 Python 스크립트 지정
CMD ["python", "/app/main/wordCloud_tree.py"]
