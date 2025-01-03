# # YOLO 모델 로드
# from ultralytics import YOLO
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
import torch
# from PIL import Image
# import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import traceback
import pandas as pd

# app = FastAPI()


# # 모델 로드
# model = YOLO('/Users/shimgeon-u/Downloads/last.pt')

# @app.post("/analyze")
# async def analyze_image(file: UploadFile = File(...)):
#     try:
#         # 업로드된 이미지 읽기
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes))

#         # YOLO 모델로 예측
#         results = model(image)
#         predictions = results.pandas().xyxy[0].to_dict(orient='records')  # JSON 변환

#         return JSONResponse(content={"predictions": predictions}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

app = FastAPI()

# YOLO 모델 로드
try:
    # model = YOLO('/path/to/best.pt')  # 정확한 모델 경로로 수정
    model = YOLO('/Users/shimgeon-u/Downloads/bulssuk.pt')

    print("YOLO 모델 로드 성공!")
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")
    traceback.print_exc()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 업로드된 파일 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        print("이미지 읽기 성공!")

        # YOLO 모델로 추론
        results = model(image)
        print("YOLO 모델 추론 성공!")

        # YOLO 결과 처리
        predictions = []
        for result in results:
            for box in result.boxes.data.tolist():  # box 데이터 추출
                x_min, y_min, x_max, y_max, confidence, class_id = box
                predictions.append({
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max,
                    "confidence": confidence,
                    "class_id": int(class_id),
                    "name": model.names[int(class_id)]  # 클래스 이름
                })

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        print(f"이미지 분석 중 오류 발생: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

