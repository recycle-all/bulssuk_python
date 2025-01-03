# Python slim 이미지를 기반으로 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# .DS_Store 파일 무시를 위한 명령어 추가 (빌드 중 삭제)
RUN find . -name ".DS_Store" -delete || true

RUN apt-get update && apt-get install -y python3-pip && pip3 install --upgrade pip

# 필요한 파일 복사
COPY requirements.txt ./
COPY app.py ./

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 내부에서 사용할 포트 지정 (필요한 경우 수정)
EXPOSE 5005

# 애플리케이션 실행 명령어
CMD ["python", "app.py"]
