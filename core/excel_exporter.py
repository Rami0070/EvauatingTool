import io
import re
from typing import List, Dict, Any
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

def clean_sheet_name(raw_name: str, existing_names: List[str] = None) -> str:
    """
    Excel 시트명 제한 사항 처리:
    1. 특수문자 [ ] : * ? / \\ 제거
    2. 최대 31자 초과 시 절삭
    3. 중복 방지 (필요시 _1, _2 접미사)
    """
    if existing_names is None:
        existing_names = []
        
    cleaned = re.sub(r'[\\/*?:\[\]]', '', raw_name).strip()
    if not cleaned:
        cleaned = "수행평가결과"
    
    cleaned = cleaned[:28]  # 중복 접미사 공간 확보
    
    final_name = cleaned
    counter = 1
    while final_name in existing_names:
        final_name = f"{cleaned}_{counter}"
        counter += 1
        
    return final_name[:31]

def get_student_sort_key(item: Dict[str, Any]):
    """
    학생 번호 기준 정렬 키:
    - 순수 숫자 또는 숫자 포함 문자열 -> (0, 정수값)
    - 미상 또는 번호 부재 -> (1, 미상번호 또는 이름)
    이를 통해 1, 2, 3, 10, 11 순으로 자연스럽게 오름차순 정렬되고
    미상 학생은 맨 뒤로 배치됩니다.
    """
    raw_num = str(item.get("번호", "")).strip()
    digits = re.findall(r'\d+', raw_num)
    if "미상" in raw_num or not digits:
        return (1, raw_num)
    else:
        return (0, int(digits[0]))

def create_grading_excel(results: List[Dict[str, Any]]) -> bytes:
    """
    채점 결과를 받아 과목별(단원별) 시트로 분리된 고품질 Excel 바이트 데이터를 생성합니다.
    파스텔 그린 테마 서식 및 열 너비, 자동 줄바꿈을 완벽히 적용합니다.
    """
    wb = openpyxl.Workbook()
    # 기본 생성된 기본 시트 제거 준비
    default_sheet = wb.active

    # 과목별(또는 과목_단원별) 그룹핑
    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        subject = str(r.get("과목", "공통")).strip() or "기타과목"
        unit = str(r.get("단원명", "")).strip()
        
        # 탭 그룹 키: 과목_단원 요약
        short_unit = re.sub(r'^\d+\.\s*', '', unit)
        group_key = f"{subject}_{short_unit}" if short_unit else subject
        
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(r)

    # 파스텔 그린 스타일 정의
    header_fill = PatternFill(start_color="D6EADF", end_color="D6EADF", fill_type="solid")
    header_font = Font(name="맑은 고딕", size=11, bold=True, color="1F4E38")
    content_font = Font(name="맑은 고딕", size=10, color="2C3E50")
    
    # 등급별 하이라이트 색상 (파스텔 톤)
    grade_colors = {
        "매우잘함": PatternFill(start_color="E8F5E9", fill_type="solid"),
        "잘함": PatternFill(start_color="F1F8E9", fill_type="solid"),
        "보통": PatternFill(start_color="FFFDE7", fill_type="solid"),
        "노력요함": PatternFill(start_color="FFEBEE", fill_type="solid")
    }

    thin_border_side = Side(style="thin", color="D0D7DE")
    grid_border = Border(
        left=thin_border_side,
        right=thin_border_side,
        top=thin_border_side,
        bottom=thin_border_side
    )

    columns = [
        ("번호", 8, "center"),
        ("이름", 12, "center"),
        ("반", 8, "center"),
        ("과목", 12, "center"),
        ("단원명(평가범위)", 22, "left"),
        ("학생 답안 요약 및 채점 근거", 35, "left"),
        ("등급", 12, "center"),
        ("평가 의견 (150자 내외)", 55, "left"),
        ("글자 수", 10, "center")
    ]

    created_sheet_names = []

    for group_key, items in grouped_data.items():
        # 정렬: 학생 번호 기준 정수 오름차순
        items.sort(key=get_student_sort_key)

        sheet_title = clean_sheet_name(group_key, created_sheet_names)
        created_sheet_names.append(sheet_title)
        
        ws = wb.create_sheet(title=sheet_title)

        # 1. 헤더 행 작성
        for col_idx, (col_name, col_width, align) in enumerate(columns, 1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = grid_border
            ws.column_dimensions[get_column_letter(col_idx)].width = col_width

        ws.row_dimensions[1].height = 28

        # 2. 데이터 행 작성
        for row_idx, item in enumerate(items, 2):
            grade_val = str(item.get("등급", "보통")).strip()
            comment_val = str(item.get("평가의견", "")).strip()
            char_count = item.get("글자수", len(comment_val))

            row_values = [
                item.get("번호", ""),
                item.get("이름", ""),
                item.get("반", ""),
                item.get("과목", ""),
                item.get("단원명", ""),
                item.get("답안요약_및_근거", ""),
                grade_val,
                comment_val,
                char_count
            ]

            for col_idx, val in enumerate(row_values, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.font = content_font
                cell.border = grid_border
                
                align_type = columns[col_idx-1][2]
                wrap = (col_idx in (6, 8)) # 답안 요약과 평어는 자동 줄바꿈 필수
                cell.alignment = Alignment(horizontal=align_type, vertical="center", wrap_text=wrap)

                # 등급 열에 부드러운 파스텔 하이라이트 부여
                if col_idx == 7 and grade_val in grade_colors:
                    cell.fill = grade_colors[grade_val]
                    cell.font = Font(name="맑은 고딕", size=10, bold=True, color="2C3E50")

            ws.row_dimensions[row_idx].height = 55 # 넉넉한 가독성 확보

    # 처음 기본 시트 삭제
    if default_sheet in wb.worksheets and len(wb.worksheets) > 1:
        wb.remove(default_sheet)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()
