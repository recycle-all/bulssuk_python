# Python 3.9-slim 이미지 사용
FROM python:3.9-slim

WORKDIR /app

# 필수 라이브러리 설치 및 pip 업그레이드
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 라이브러리 경로 갱신
RUN ldconfig

# .DS_Store 삭제 명령어 추가 (호스트에서 빌드 시 삭제)
RUN find . -name ".DS_Store" -delete || true

# Python 의존성 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# 컨테이너 포트 노출
EXPOSE 8765

# 앱 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
