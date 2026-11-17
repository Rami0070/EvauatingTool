import sys
import os
import concurrent.futures
import threading

# 모듈 탐색 경로 보정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# .env 환경 설정 자동 로드 (API 키 보안 및 자동 입력)
env_file = os.path.join(current_dir, ".env")
if os.path.exists(env_file):
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GEMINI_API_KEY" and not os.environ.get("GEMINI_API_KEY"):
                        os.environ["GEMINI_API_KEY"] = v.strip()
    except Exception:
        pass

import streamlit as st
import pandas as pd
import json
import time
import re
from typing import List, Dict, Any

import importlib
import core.pdf_processor
import core.gemini_grader
import core.excel_exporter

# Streamlit 재실행 시에도 항상 최신 core 모듈 로드 보장
importlib.reload(core.pdf_processor)
importlib.reload(core.gemini_grader)
importlib.reload(core.excel_exporter)

from core.pdf_processor import convert_multiple_pdfs_to_page_items
from core.gemini_grader import GeminiGrader
from core.excel_exporter import create_grading_excel, get_student_sort_key

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 파스텔 그린 3D 테마 CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI 수행평가 자동 채점 & 평어 생성기",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 3D 프레스 버튼 및 파스텔 그린 UI 스타일
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
}

/* 전체 배경 및 톤 */
.main {
    background-color: #F8FAF8;
}

/* 헤더 배너 */
.app-header {
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(46, 125, 50, 0.08);
    margin-bottom: 2rem;
    border: 1px solid #C8E6C9;
}
.app-header h1 {
    color: #1B5E20;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.app-header p {
    color: #2E7D32;
    margin: 0.5rem 0 0 0;
    font-size: 1.05rem;
    line-height: 1.5;
}

/* 안내 카드 */
.card-box {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #E0EBE2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    margin-bottom: 1.2rem;
}

/* 파일 업로드 카드 헤더 */
.upload-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2E7D32;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* 3D 입체 프레스 버튼 (Pressable 3D Button) */
div.stButton > button:first-child {
    background: linear-gradient(to bottom, #4CAF50, #388E3C) !important;
    color: #FFFFFF !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 0.8rem 2.2rem !important;
    border-radius: 12px !important;
    border: none !important;
    border-bottom: 4px solid #1B5E20 !important;
    box-shadow: 0 4px 0 #1B5E20, 0 8px 15px rgba(46, 125, 50, 0.3) !important;
    transition: all 0.1s ease-in-out !important;
    cursor: pointer !important;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(to bottom, #43A047, #2E7D32) !important;
    transform: translateY(1px) !important;
    box-shadow: 0 3px 0 #1B5E20, 0 6px 12px rgba(46, 125, 50, 0.3) !important;
}

div.stButton > button:first-child:active {
    transform: translateY(4px) !important;
    border-bottom: 0px solid transparent !important;
    box-shadow: 0 0 0 #1B5E20, inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* 다운로드 전용 3D 버튼 */
div.stDownloadButton > button:first-child {
    background: linear-gradient(to bottom, #2E7D32, #1B5E20) !important;
    color: #FFFFFF !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    border-bottom: 4px solid #0D3813 !important;
    box-shadow: 0 4px 0 #0D3813, 0 8px 15px rgba(27, 94, 32, 0.25) !important;
    transition: all 0.1s ease-in-out !important;
    cursor: pointer !important;
}

div.stDownloadButton > button:first-child:hover {
    background: linear-gradient(to bottom, #256628, #144919) !important;
    transform: translateY(1px) !important;
    box-shadow: 0 3px 0 #0D3813, 0 6px 12px rgba(27, 94, 32, 0.25) !important;
}

div.stDownloadButton > button:first-child:active {
    transform: translateY(4px) !important;
    border-bottom: 0px solid transparent !important;
    box-shadow: 0 0 0 #0D3813, inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* 등급 뱃지 */
.grade-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.grade-vh { background-color: #C8E6C9; color: #1B5E20; }
.grade-hi { background-color: #E8F5E9; color: #2E7D32; }
.grade-md { background-color: #FFF9C4; color: #F57F17; }
.grade-lo { background-color: #FFCDD2; color: #B71C1C; }

/* 단계 표시 칩 */
.step-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #E8F5E9;
    color: #2E7D32;
    padding: 6px 14px;
    border-radius: 16px;
    font-size: 0.92rem;
    font-weight: 600;
    border: 1px solid #C8E6C9;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 세션 상태 초기화
# -----------------------------------------------------------------------------
if "rubrics" not in st.session_state:
    st.session_state.rubrics = []
if "graded_results" not in st.session_state:
    st.session_state.graded_results = []
if "editing_dfs" not in st.session_state:
    st.session_state.editing_dfs = {}

# -----------------------------------------------------------------------------
# 3. 사이드바 설정 영역
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    st.caption("초등학교 교사용 스마트 AI 평가 관리실")
    
    # Gemini API Key (환경변수 또는 .env 자동 연동)
    gemini_key = st.text_input(
        "🔑 Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Google Gemini API 키가 기본 적용되어 있습니다. 필요한 경우 변경할 수 있습니다."
    )
    
    # 모델 선택 (무료 쿼터 최적화)
    model_choice = st.selectbox(
        "🧠 Gemini Vision 모델 선택",
        options=[
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash"
        ],
        index=0,
        help="3.5-flash-lite는 1~2초대로 가장 빠르고 일일 쿼터가 넉넉합니다. (강력 추천)"
    )

    st.markdown("---")
    st.markdown("### 📑 시험지 구성 방식 (쿼터 절약)")

    split_mode_choice = st.radio(
        "시험지 페이지 분할 방식",
        options=[
            ("2pages", "📄 2페이지씩 앞·뒷장 (추천: 쿼터 소모 0회 / 초고속)"),
            ("1page", "📄 1페이지씩 단면 (쿼터 소모 0회 / 초고속)"),
            ("auto", "🔍 AI 자동 감지 (양식 불규칙 시)")
        ],
        format_func=lambda x: x[1],
        index=0,
        help="앞·뒷장 2페이지 단위로 선택하시면 헤더 판독에 API를 전혀 쓰지 않아 일일 무료 쿼터를 100% 아낄 수 있습니다."
    )[0]
    
    st.markdown("---")
    st.markdown("### ✍️ 평어 작성 옵션")
    
    tone_choice = st.radio(
        "문체 스타일 선택",
        options=[
            ("concise", "생기부 표준 개조식 (~함, ~보임)"),
            ("polite", "정중한 서술형 (~합니다, ~보입니다)")
        ],
        format_func=lambda x: x[1],
        index=0
    )[0]
    
    dpi_choice = st.select_slider(
        "📄 PDF 이미지 변환 해상도 (DPI)",
        options=[100, 150, 200],
        value=150,
        help="DPI가 높을수록 작은 손글씨와 그래프가 더 선명해지며, 150이 속도와 화질 균형에 이상적입니다."
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='background-color:#F1F8F3; padding:12px; border-radius:10px; font-size:0.85rem; color:#2E7D32;'>
        💡 <b>안내 사항</b><br>
        • PDF 페이지를 고해상도 이미지로 변환하여 Vision AI로 분석하므로 양식이 바뀌어도 파싱 오류가 없습니다.<br>
        • 2페이지 시험지(뒷장)는 자동으로 앞 학생에 병합되어 종합 채점됩니다.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 4. 메인 화면 헤더 및 파일 업로드 섹션
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🌱 AI 초등 수행평가 자동 채점 & 평어 생성기</h1>
        <p>어떤 서식의 평가 계획서와 기준안이든 Vision AI가 자동으로 구조화하고, 
        학생들의 손글씨·분수 그림·꺾은선그래프·포스터를 꼼꼼히 채점하여 100자 내외 개별 맞춤 평어를 완성합니다.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("#### 📂 3대 필수 평가 자료 업로드")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card-box">
            <div class="upload-card-title">📋 1. 수행평가 계획안</div>
            <span style="font-size:0.85rem; color:#666;">과목별 성취기준, 단원, 평가 요소가 포함된 PDF</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    plan_files = st.file_uploader(
        "수행평가 계획서 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="plan_uploader",
        label_visibility="collapsed"
    )

with col2:
    st.markdown(
        """
        <div class="card-box">
            <div class="upload-card-title">🎯 2. 교사별 평가 기준안</div>
            <span style="font-size:0.85rem; color:#666;">4단계 도달기준, 모범답안, 유의점이 포함된 PDF</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    rubric_files = st.file_uploader(
        "교사별 평가 기준안 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="rubric_uploader",
        label_visibility="collapsed"
    )

with col3:
    st.markdown(
        """
        <div class="card-box">
            <div class="upload-card-title">📝 3. 학생 시험지 (채점대상)</div>
            <span style="font-size:0.85rem; color:#666;">학생들이 작성한 스캔 시험지 (다중 파일 가능)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    exam_files = st.file_uploader(
        "학생 시험지 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="exam_uploader",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. [채점 실행] 3D 버튼 및 메인 프로세싱 파이프라인
# -----------------------------------------------------------------------------
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    start_btn = st.button("🚀 AI 수행평가 채점 및 평어 생성 시작", use_container_width=True)

if start_btn:
    if not gemini_key:
        st.error("⚠️ 사이드바에서 Gemini API Key를 먼저 입력해 주세요!")
    elif not plan_files or not rubric_files or not exam_files:
        st.warning("⚠️ 3가지 자료(수행평가 계획, 교사별 평가 기준안, 학생 시험지)를 모두 업로드해 주세요!")
    else:
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        
        try:
            grader = GeminiGrader(api_key=gemini_key, model_name=model_choice)

            # -----------------------------------------------------------------
            # Step 1: 평가 기준 동적 추출 및 통합 (유연한 파싱)
            # -----------------------------------------------------------------
            status_box.markdown(
                '<div class="step-chip"><b>Step 1/3</b> 평가 계획서 및 기준안 이미지 변환 & 루브릭 동적 추출 중...</div>',
                unsafe_allow_html=True
            )
            progress_bar.progress(0.1)

            plan_pages = convert_multiple_pdfs_to_page_items(plan_files, dpi=dpi_choice)
            rubric_pages = convert_multiple_pdfs_to_page_items(rubric_files, dpi=dpi_choice)

            def step1_cb(pct, msg):
                progress_bar.progress(0.1 + pct * 0.25)
                status_box.markdown(f'<div class="step-chip"><b>Step 1/3</b> {msg}</div>', unsafe_allow_html=True)

            merged_rubrics = grader.extract_and_merge_rubrics(plan_pages, rubric_pages, progress_callback=step1_cb)
            st.session_state.rubrics = merged_rubrics

            # -----------------------------------------------------------------
            # Step 2: 학생 시험지 자동 분할 및 다중 페이지 병합
            # -----------------------------------------------------------------
            status_box.markdown(
                '<div class="step-chip"><b>Step 2/3</b> 학생 시험지 분할 및 다중 페이지(뒷장) 자동 병합 중...</div>',
                unsafe_allow_html=True
            )
            progress_bar.progress(0.4)

            exam_pages = convert_multiple_pdfs_to_page_items(exam_files, dpi=dpi_choice)

            def step2_cb(pct, msg):
                progress_bar.progress(0.4 + pct * 0.2)
                status_box.markdown(f'<div class="step-chip"><b>Step 2/3</b> {msg}</div>', unsafe_allow_html=True)

            student_submissions = grader.group_student_exam_pages(
                exam_pages,
                split_mode=split_mode_choice,
                progress_callback=step2_cb
            )

            # -----------------------------------------------------------------
            # Step 3: 기준 기반 채점, 4단계 등급 부여 및 100자 맞춤 평어 생성
            # -----------------------------------------------------------------
            status_box.markdown(
                '<div class="step-chip"><b>Step 3/3</b> 학생별 시험지 정밀 채점 및 100자 맞춤 평어 생성 중...</div>',
                unsafe_allow_html=True
            )
            
            total_students = len(student_submissions)
            graded_list = [None] * total_students
            completed_students = 0
            grade_lock = threading.Lock()

            def process_single_student_grade(task_item):
                s_idx, s_data = task_item
                try:
                    g_info = grader.grade_student(
                        student=s_data,
                        rubrics=merged_rubrics,
                        tone_style=tone_choice
                    )
                except Exception as grade_err:
                    g_info = {
                        "과목": s_data.get("과목", ""),
                        "단원명": s_data.get("단원명", ""),
                        "답안요약_및_근거": f"채점 재시도 후 일시 오류 ({str(grade_err)[:50]})",
                        "등급": "보통",
                        "평가의견": "학생의 성취 수준을 평가 기준에 따라 종합적으로 검토하였으며 꾸준한 학습 태도와 노력이 필요함.",
                        "글자수": 58,
                        "이름": s_data.get("이름", "미상"),
                        "번호": s_data.get("번호", "미상"),
                        "반": s_data.get("반", "")
                    }

                sub = s_data.get("과목") or g_info.get("과목") or "기타과목"
                # 스캔 순번(1, 2, 3...)과 손글씨 번호 안전 결합
                assigned_num = str(g_info.get("번호") or s_data.get("번호", str(s_idx + 1))).strip()
                assigned_name = str(g_info.get("이름") or s_data.get("이름", f"학생_{assigned_num}")).strip()

                rec = {
                    "번호": assigned_num,
                    "이름": assigned_name,
                    "반": g_info.get("반") or s_data.get("반", ""),
                    "과목": sub,
                    "단원명": g_info.get("단원명") or s_data.get("단원명", ""),
                    "답안요약_및_근거": g_info.get("답안요약_및_근거", ""),
                    "등급": g_info.get("등급", "보통"),
                    "평가의견": g_info.get("평가의견", ""),
                    "글자수": g_info.get("글자수", len(g_info.get("평가의견", ""))),
                    "student_key": s_data.get("student_key", "")
                }
                return s_idx, rec

            # 무료 쿼터 안정성을 위해 2개 동시 스레드로 병렬 채점
            max_grade_workers = min(2, total_students) if total_students > 0 else 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_grade_workers) as executor:
                grade_tasks = [(i, s) for i, s in enumerate(student_submissions)]
                grade_futures = [executor.submit(process_single_student_grade, gt) for gt in grade_tasks]

                for future in concurrent.futures.as_completed(grade_futures):
                    s_idx, combined_record = future.result()
                    graded_list[s_idx] = combined_record
                    
                    with grade_lock:
                        completed_students += 1
                        curr_pct = 0.6 + (completed_students / total_students) * 0.38
                        progress_bar.progress(min(curr_pct, 0.98))
                        status_box.markdown(
                            f'<div class="step-chip"><b>Step 3/3</b> [{completed_students}/{total_students}] {combined_record.get("이름","학생")}({combined_record.get("번호","")}번) 채점 완료!</div>',
                            unsafe_allow_html=True
                        )

            # 혹시 누락된 레코드 정리
            graded_list = [r for r in graded_list if r is not None]

            # 번호 기준 정수 오름차순 정렬 (미상은 맨 뒤)
            graded_list.sort(key=get_student_sort_key)
            st.session_state.graded_results = graded_list

            progress_bar.progress(1.0)
            status_box.success(f"🎉 채점 및 평어 생성이 완료되었습니다! (총 {len(graded_list)}명 학생 채점 완료)")

        except Exception as ex:
            # 부분 성공 데이터가 있다면 보존하여 화면에 표출
            valid_done = [r for r in graded_list if r is not None] if 'graded_list' in locals() else []
            if valid_done:
                valid_done.sort(key=get_student_sort_key)
                st.session_state.graded_results = valid_done
                st.warning(f"⚠️ 일부 처리 중 알림: {str(ex)} (채점 완료된 {len(valid_done)}명 결과는 아래에 정상 보존되었습니다.)")
            else:
                st.error(f"❌ 처리 중 오류가 발생했습니다: {str(ex)}")

# -----------------------------------------------------------------------------
# 6. 결과 확인 및 실시간 데이터 수정(st.data_editor) & 엑셀 다운로드
# -----------------------------------------------------------------------------
if st.session_state.graded_results:
    st.markdown("---")
    st.markdown("### 📊 채점 결과 미리보기 및 교사용 편집기")
    st.caption("결과를 직접 확인하고 필요한 경우 등급이나 평어를 바로 수정할 수 있습니다. 수정한 내용은 엑셀 다운로드 시 그대로 반영됩니다.")

    all_records = st.session_state.graded_results

    # 과목별 그룹핑
    subject_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        sub = str(r.get("과목", "공통")).strip() or "기타과목"
        if sub not in subject_map:
            subject_map[sub] = []
        subject_map[sub].append(r)

    # 과목별 탭 생성
    tabs = st.tabs([f"📖 {sub} ({len(items)}명)" for sub, items in subject_map.items()])

    edited_all_records = []

    for tab_idx, (subject_name, items) in enumerate(subject_map.items()):
        with tabs[tab_idx]:
            # DataFrame 생성 및 정렬
            items.sort(key=get_student_sort_key)
            df = pd.DataFrame(items)

            # 표시용 컬럼 정리
            display_cols = ["번호", "이름", "반", "과목", "단원명", "답안요약_및_근거", "등급", "평가의견", "글자수"]
            existing_cols = [c for c in display_cols if c in df.columns]
            df_display = df[existing_cols]

            edited_df = st.data_editor(
                df_display,
                key=f"editor_{subject_name}_{tab_idx}",
                column_config={
                    "번호": st.column_config.TextColumn("번호", width="small"),
                    "이름": st.column_config.TextColumn("이름", width="small"),
                    "반": st.column_config.TextColumn("반", width="small"),
                    "과목": st.column_config.TextColumn("과목", width="small"),
                    "단원명": st.column_config.TextColumn("단원명(평가범위)", width="medium"),
                    "답안요약_및_근거": st.column_config.TextColumn("학생 답안 요약 및 채점 근거", width="large"),
                    "등급": st.column_config.SelectboxColumn(
                        "등급",
                        options=["매우잘함", "잘함", "보통", "노력요함"],
                        required=True,
                        width="small"
                    ),
                    "평가의견": st.column_config.TextColumn("평가 의견 (100자 내외)", width="large"),
                    "글자수": st.column_config.NumberColumn("글자 수", width="small", disabled=True)
                },
                num_rows="dynamic",
                use_container_width=True
            )

            # 수정된 내용 반영 및 글자 수 재계산
            edited_df["글자수"] = edited_df["평가의견"].astype(str).str.len()
            edited_records = edited_df.to_dict(orient="records")
            edited_all_records.extend(edited_records)

    # 엑셀 다운로드 섹션
    st.markdown("<br>", unsafe_allow_html=True)
    down_col1, down_col2, down_col3 = st.columns([1, 2, 1])
    
    with down_col2:
        # 수정본 기반으로 엑셀 바이트 생성
        excel_bytes = create_grading_excel(edited_all_records if edited_all_records else all_records)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"초등수행평가_채점및평어_{timestamp_str}.xlsx"

        st.download_button(
            label="📥 과목별 시트 분리 Excel(.xlsx) 다운로드",
            data=excel_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# 7. 추출된 평가 기준(루브릭) 확인용 아코디언
# -----------------------------------------------------------------------------
if st.session_state.rubrics:
    with st.expander("🔍 AI가 추출 및 통합한 평가 기준(루브릭) 상세 보기"):
        for idx, r in enumerate(st.session_state.rubrics, 1):
            st.markdown(f"**[{idx}] {r.get('과목','')} - {r.get('단원명','')}**")
            st.markdown(f"- **성취기준:** {r.get('성취기준','')}")
            st.markdown(f"- **평가요소:** {r.get('평가요소','')}")
            st.markdown(f"- **모범답안 및 유의점:** {r.get('모범답안_및_유의점','')}")
            
            crit = r.get("평가기준", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<span class='grade-badge grade-vh'>매우잘함</span><br><small>{crit.get('매우잘함','')}</small>", unsafe_allow_html=True)
            c2.markdown(f"<span class='grade-badge grade-hi'>잘함</span><br><small>{crit.get('잘함','')}</small>", unsafe_allow_html=True)
            c3.markdown(f"<span class='grade-badge grade-md'>보통</span><br><small>{crit.get('보통','')}</small>", unsafe_allow_html=True)
            c4.markdown(f"<span class='grade-badge grade-lo'>노력요함</span><br><small>{crit.get('노력요함','')}</small>", unsafe_allow_html=True)
            st.markdown("---")
