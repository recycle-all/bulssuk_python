# Python 3.9-slim 이미지 사용
FROM python:3.9-slim

WORKDIR /app


# .DS_Store 삭제 명령어 추가 (호스트에서 빌드 시 삭제)
RUN find . -name ".DS_Store" -delete || true

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
