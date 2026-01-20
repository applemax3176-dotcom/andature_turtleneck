import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import math

# --- 페이지 설정 ---
st.set_page_config(page_title="거북목 정밀 진단 (Re-Set)", page_icon="🐢")

st.title("🐢 안다쳐랩 : 거북목 부하량 측정")
st.markdown("""
**[사용자 맞춤 재설정 값 적용]**
- **어깨(견봉):** 관절 중심에서 **위로 4.5cm, 앞으로 4.0cm**
- **귀(포인트):** 원래 귀 위치에서 **뒤로 5.0cm**
""")

# --- 메인 로직 ---

uploaded_file = st.file_uploader("측면 사진 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # MediaPipe Pose 설정
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

    with st.spinner("강력해진 보정값 적용 중..."):
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 1. AI 초기 좌표 (Raw Data)
        nose = lm[mp_pose.PoseLandmark.NOSE]
        ear_raw = lm[mp_pose.PoseLandmark.LEFT_EAR]      # 귀 (원본)
        shoulder_raw = lm[mp_pose.PoseLandmark.LEFT_SHOULDER] # 어깨 (관절중심)

        # 픽셀 변환
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        ear_x_raw, ear_y_raw = int(ear_raw.x * w), int(ear_raw.y * h)
        sh_x_raw, sh_y_raw = int(shoulder_raw.x * w), int(shoulder_raw.y * h)

        # -----------------------------------------------------------
        # [STEP 1] 스케일링 (코~귀 12cm 기준)
        # -----------------------------------------------------------
        pixel_dist_nose_ear = math.sqrt((nose_x - ear_x_raw)**2 + (nose_y - ear_y_raw)**2)
        
        if pixel_dist_nose_ear < 30:
            st.error("얼굴 인식이 명확하지 않습니다.")
            st.stop()
            
        cm_per_pixel = 12.0 / pixel_dist_nose_ear

        # -----------------------------------------------------------
        # [STEP 2] 해부학적 보정 (Correction) - 강화됨!
        # -----------------------------------------------------------
        
        # 방향 판단 (코가 귀보다 오른쪽이면 -> 오른쪽 보는 중)
        looking_right = nose_x > ear_x_raw
        
        # --- A. 견봉(Acromion) 보정 ---
        # 상완골두에서: 위로 4.5cm(유지) / 앞으로 4.0cm(2cm 추가)
        ACROMION_UP_CM = 4.5
        ACROMION_FRONT_CM = 4.0 

        acromion_up_px = int(ACROMION_UP_CM / cm_per_pixel)
        acromion_front_px = int(ACROMION_FRONT_CM / cm_per_pixel)
        
        sh_y = sh_y_raw - acromion_up_px # 위로 이동
        
        # --- B. 귀(Ear) 보정 ---
        # 요청사항: 원래 위치에서 "뒤로 5.0cm" (기존3 + 추가2)
        EAR_BACK_CM = 5.0
        ear_back_px = int(EAR_BACK_CM / cm_per_pixel)
        
        # Y축(높이)은 원본 유지
        ear_y = ear_y_raw 
        
        # 좌우(앞뒤) 이동 적용
        if looking_right:
            # 오른쪽 보는 중
            sh_x = sh_x_raw + acromion_front_px # 어깨는 앞(우)으로 4cm
            ear_x = ear_x_raw - ear_back_px     # 귀는 뒤(좌)로 5cm!
            
            # FHD 계산 (귀X - 어깨X)
            fhd_pixel = ear_x - sh_x
            
        else:
            # 왼쪽 보는 중
            sh_x = sh_x_raw - acromion_front_px # 어깨는 앞(좌)으로 4cm
            ear_x = ear_x_raw + ear_back_px     # 귀는 뒤(우)로 5cm!
            
            # FHD 계산 (어깨X - 귀X)
            fhd_pixel = sh_x - ear_x

        # -----------------------------------------------------------
        # [STEP 3] 결과 계산
        # -----------------------------------------------------------
        # 귀가 어깨보다 뒤에 있으면 0 처리
        if fhd_pixel < 0: fhd_pixel = 0 
        
        fhd_cm = fhd_pixel * cm_per_pixel
        neck_load_kg = 5.0 + (fhd_cm * 3.0)

        # 진단 등급
        if fhd_cm <= 2.5:
            status = "정상 (Normal)"
            bg_color = "#d4edda"
            msg_color = "#155724"
        elif fhd_cm < 5.0:
            status = "초기 거북목 (Mild)"
            bg_color = "#fff3cd"
            msg_color = "#856404"
        else:
            status = "심각 (Severe)"
            bg_color = "#f8d7da"
            msg_color = "#721c24"

        # -----------------------------------------------------------
        # [STEP 4] 시각화
        # -----------------------------------------------------------
        annotated_image = image_np.copy()
        
        # 견봉 (빨강)
        cv2.line(annotated_image, (sh_x, sh_y - 200), (sh_x, sh_y + 200), (0, 0, 255), 2)
        cv2.circle(annotated_image, (sh_x, sh_y), 6, (0, 0, 255), -1)
        cv2.putText(annotated_image, "Acromion(+4cm)", (sh_x - 60, sh_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 귀 포인트 (초록 - 뒤로 5cm 이동됨)
        cv2.line(annotated_image, (ear_x, ear_y - 200), (ear_x, ear_y + 200), (0, 255, 0), 2)
        cv2.circle(annotated_image, (ear_x, ear_y), 6, (0, 255, 0), -1)
        cv2.putText(annotated_image, "Point(-5cm)", (ear_x - 40, ear_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 거리선 (파랑)
        mid_y = int((ear_y + sh_y) / 2)
        cv2.line(annotated_image, (sh_x, mid_y), (ear_x, mid_y), (255, 0, 0), 4)
        cv2.putText(annotated_image, f"{fhd_cm:.1f}cm", (int((sh_x+ear_x)/2)-30, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        st.image(annotated_image, caption="분석 결과: 대폭 수정된 보정값 적용", use_column_width=True)

        st.divider()
        st.subheader(f"진단 결과: {status}")
        
        comment = f"보정된 귀 포인트가 견봉보다 **{fhd_cm:.1f}cm** 앞에 있습니다.<br>목 하중 예측: **{neck_load_kg:.1f}kg**"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 1px solid {msg_color};">
            <h3 style="color: {msg_color}; margin:0;">{comment}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("현재 적용된 강력 보정값"):
            st.write(f"- **견봉:** 관절 중심에서 위로 {ACROMION_UP_CM}cm, **앞으로 {ACROMION_FRONT_CM}cm**")
            st.write(f"- **귀:** AI 원본 위치에서 **뒤로 {EAR_BACK_CM}cm**")

    else:
        st.error("사람을 찾지 못했습니다.")
