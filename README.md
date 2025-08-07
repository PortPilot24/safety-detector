# safety-detector
YOLOv8 기반 하역장 안전보호구 미착용 감지 시스템


## 🚀 실행 방법

1. **필요 라이브러리 설치**

```bash
pip install -r requirements.txt
```

**2. 가상 환경 생성 및 활성화**

프로젝트의 의존성을 관리하기 위해 가상 환경을 사용하는 것을 권장합니다.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. **FastAPI 실행**

```bash
uvicorn main:app --reload
```
