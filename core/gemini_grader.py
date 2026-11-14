import os
import time
import json
import random
import re
from typing import List, Dict, Any, Optional
from PIL import Image

class GeminiGrader:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self._init_client()

    def _init_client(self):
        """google.genai 최신 SDK를 우선 사용하고 실패 시 google.generativeai로 fallback"""
        self.client_type = "genai"
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self.client_type = "legacy"
            self.legacy_genai = legacy_genai

    def _call_with_retry(self, contents: list, json_mode: bool = True, max_retries: int = 5) -> str:
        """API Rate Limit 및 일시적 오류에 대응하는 지수 백오프 재시도 함수"""
        delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.client_type == "genai":
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json" if json_mode else "text/plain"
                    )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    )
                    return response.text
                else:
                    generation_config = {
                        "temperature": 0.1,
                    }
                    if json_mode:
                        generation_config["response_mime_type"] = "application/json"
                    
                    model = self.legacy_genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=generation_config
                    )
                    response = model.generate_content(contents)
                    return response.text

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # 429 Resource Exhausted / Rate limit / Quota exceeded 또는 일시적 서버 오류 감지
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str or "503" in err_str or "overloaded" in err_str:
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = delay + jitter
                    time.sleep(sleep_time)
                    delay *= 2.0
                else:
                    time.sleep(1.5)

        raise RuntimeError(f"Gemini API 호출 실패 (재시도 {max_retries}회 초과): {last_error}")

    def _clean_json_text(self, text: str) -> str:
        """응답에서 마크다운 코드블록(```json ... ```) 제거 및 순수 JSON 문자열 추출"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    # =========================================================================
    # Step 1: 평가 기준 동적 추출 및 통합 (유연한 파싱)
    # =========================================================================
    def extract_and_merge_rubrics(
        self,
        plan_images: List[Dict[str, Any]],
        rubric_images: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        '수행평가 계획서 PDF'와 '교사별 평가 기준안 PDF'의 이미지들을 Vision API로 분석하여
        과목별/단원별 평가 루브릭을 단일 JSON 구조화 리스트로 병합합니다.
        단원명이 다소 상이하더라도 성취기준과 평가요소가 일치하면 하나로 통합합니다.
        """
        if progress_callback:
            progress_callback(0.1, "평가 계획서 및 교사별 기준안 이미지 분석 준비 중...")

        prompt = """
당신은 대한민국 초등학교 교육과정 및 학생 평가 전문가입니다.
제공된 이미지들은 초등학교 '수행평가 계획안' 및 '교사별 평가 기준안' 문서 페이지들입니다.
문서의 서식이나 표 양식에 구애받지 않고, 포함된 모든 과목 및 단원별 평가 기준을 파싱하여 다음 JSON 스키마의 배열로 추출해 주세요.

[중요 통합 및 매칭 규칙]
1. 계획서와 교사별 기준안의 단원명 표기가 약간 다르더라도(예: 계획서의 '5. 의견이 드러나게 글을 써요' vs 기준안의 '3. 의견을 모아서') 성취기준과 평가요소의 내용 및 맥락이 같으면 동일한 하나의 평가 단원으로 스마트하게 병합하세요.
2. '모범답안_및_유의점'에는 교사별 기준안에 제시된 예시 답안, 문항별 정답, 채점 시 유의사항 등을 핵심 요약하여 포함하세요.
3. 4단계 평가기준(매우잘함, 잘함, 보통, 노력요함)의 문장을 누락 없이 명확히 추출하세요.

[반환 JSON 스키마]
[
  {
    "과목": "국어 / 수학 / 사회 / 과학 / 도덕 / 음악 / 미술 / 체육 / 영어 / 창체 등",
    "단원명": "단원 번호 및 명칭 (예: 3. 의견을 모아서)",
    "성취기준": "교육과정 성취기준 텍스트",
    "평가요소": "평가 관점 및 평가 요소",
    "평가기준": {
      "매우잘함": "매우 잘함 도달 기준",
      "잘함": "잘함 도달 기준",
      "보통": "보통 도달 기준",
      "노력요함": "노력 요함 도달 기준"
    },
    "모범답안_및_유의점": "예시 답안, 정답 및 유의사항 요약"
  }
]
반드시 유효한 JSON 형식의 배열로만 답변해 주세요.
"""
        contents = [prompt]
        # 계획서 이미지 추가
        for idx, item in enumerate(plan_images):
            contents.append(f"[수행평가 계획서 파일: {item['filename']} p.{item['page_num']}]")
            contents.append(item["image"])

        # 교사별 기준안 이미지 추가
        for idx, item in enumerate(rubric_images):
            contents.append(f"[교사별 평가 기준안 파일: {item['filename']} p.{item['page_num']}]")
            contents.append(item["image"])

        if progress_callback:
            progress_callback(0.4, "Vision AI를 통해 루브릭 및 모범답안 동적 구조화(JSON) 중...")

        response_text = self._call_with_retry(contents, json_mode=True)
        cleaned_json = self._clean_json_text(response_text)
        
        try:
            rubrics = json.loads(cleaned_json)
            if isinstance(rubrics, dict) and "rubrics" in rubrics:
                rubrics = rubrics["rubrics"]
        except Exception as parse_err:
            # 보조 정규식 추출
            match = re.search(r'\[.*\]', cleaned_json, re.DOTALL)
            if match:
                rubrics = json.loads(match.group(0))
            else:
                raise ValueError(f"루브릭 JSON 파싱 실패: {parse_err}\n원본 응답:\n{response_text[:300]}")

        if progress_callback:
            progress_callback(1.0, f"평가 기준 추출 완료 (총 {len(rubrics)}개 과목/단원 분석됨)")

        return rubrics

    # =========================================================================
    # Step 2: 학생 시험지 자동 분할 및 다중 페이지 병합
    # =========================================================================
    def group_student_exam_pages(
        self,
        exam_pages: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        시험지 각 페이지의 상단 헤더를 인식하고,
        인적사항(반/번호/이름) 칸이 없고 문항이 앞장과 이어지는 뒷장의 경우
        직전 학생에게 자동 병합하여 다중 페이지 시험지를 형성합니다.
        번호 미기재 시 '미상_1', '미상_2' 등으로 자동 분류합니다.
        """
        total = len(exam_pages)
        if total == 0:
            return []

        # 1차로 각 페이지의 헤더 정보를 Vision AI로 빠르게 판독
        header_prompt = """
시험지 페이지의 상단 헤더 영역을 판독하여 다음 정보를 JSON 객체로 반환해 주세요.

[판독 항목]
1. is_new_student_header: 이 페이지가 새로운 학생의 첫 페이지(상단에 반, 번호, 이름 기재란이 명확히 존재하는 경우)이면 true, 
   인적사항 기재란이 없고 문항 번호가 이어지거나 앞장의 뒷장/이음페이지이면 false.
2. 학년: 숫자 (예: 4)
3. 반: 숫자 (예: 7)
4. 번호: 숫자 또는 미상 (예: 1, 15, 미상)
5. 이름: 학생 이름 (예: 홍길동, 빈칸인 경우 '미상')
6. 과목: 국어 / 수학 / 사회 / 과학 / 도덕 / 음악 / 미술 등
7. 단원명: 표기된 단원명 또는 평가범위 (예: 3. 의견을 모아서)

[응답 JSON 스키마]
{
  "is_new_student_header": true,
  "학년": 4,
  "반": 7,
  "번호": "1",
  "이름": "김철수",
  "과목": "수학",
  "단원명": "1. 분수의 덧셈과 뺄셈"
}
"""
        page_analyses = []
        unknown_counter = 1

        for idx, p_item in enumerate(exam_pages):
            if progress_callback:
                pct = 0.1 + (idx / total) * 0.4
                progress_callback(pct, f"시험지 페이지 헤더 판독 중... ({idx+1}/{total} 페이지)")

            contents = [
                header_prompt,
                f"파일명: {p_item['filename']}, 페이지: {p_item['page_num']}",
                p_item["image"]
            ]
            resp = self._call_with_retry(contents, json_mode=True)
            cleaned = self._clean_json_text(resp)
            try:
                header_data = json.loads(cleaned)
            except Exception:
                header_data = {
                    "is_new_student_header": True,
                    "학년": "",
                    "반": "",
                    "번호": "미상",
                    "이름": "미상",
                    "과목": "",
                    "단원명": ""
                }
            
            page_analyses.append({
                "page_info": p_item,
                "header": header_data
            })

        # 다중 페이지 병합 처리
        student_submissions = []
        current_student = None

        for idx, item in enumerate(page_analyses):
            header = item["header"]
            p_img = item["page_info"]["image"]
            page_meta = {
                "filename": item["page_info"]["filename"],
                "page_num": item["page_info"]["page_num"],
                "image": p_img
            }

            is_new = header.get("is_new_student_header", True)

            # 첫 페이지이거나 새로운 학생의 헤더가 명확한 경우
            if is_new or current_student is None:
                # 번호 처리
                raw_num = str(header.get("번호", "")).strip()
                if not raw_num or raw_num == "None" or raw_num == "미상" or not any(char.isdigit() for char in raw_num):
                    assigned_num = f"미상_{unknown_counter}"
                    unknown_counter += 1
                else:
                    digits = re.findall(r'\d+', raw_num)
                    assigned_num = digits[0] if digits else f"미상_{unknown_counter}"

                name = str(header.get("이름", "")).strip()
                if not name or name == "None" or name == "빈칸":
                    name = "미상"

                current_student = {
                    "student_key": f"{header.get('반','')}_{assigned_num}_{name}",
                    "학년": header.get("학년", ""),
                    "반": header.get("반", ""),
                    "번호": assigned_num,
                    "이름": name,
                    "과목": header.get("과목", ""),
                    "단원명": header.get("단원명", ""),
                    "pages": [page_meta]
                }
                student_submissions.append(current_student)
            else:
                # 뒷장이므로 직전 학생의 pages 리스트에 추가 병합!
                current_student["pages"].append(page_meta)
                # 만약 과목이나 단원명이 첫 장에 비어있었고 뒷장에 있다면 보완
                if not current_student.get("과목") and header.get("과목"):
                    current_student["과목"] = header.get("과목")
                if not current_student.get("단원명") and header.get("단원명"):
                    current_student["단원명"] = header.get("단원명")

        if progress_callback:
            progress_callback(0.6, f"시험지 분할 및 다중 페이지 병합 완료: 총 {len(student_submissions)}명 학생 식별됨")

        return student_submissions

    # =========================================================================
    # Step 3: 기준 기반 채점, 4단계 등급 부여 및 150자 평어 생성
    # =========================================================================
    def grade_student(
        self,
        student: Dict[str, Any],
        rubrics: List[Dict[str, Any]],
        tone_style: str = "concise"  # 'concise': ~함, ~보임 | 'polite': ~합니다, ~보입니다
    ) -> Dict[str, Any]:
        """
        학생의 시험지(단일/다중 페이지) 이미지를 분석하여
        해당 과목/단원 평가기준 및 모범답안을 바탕으로 4단계 등급 판정 및
        150자 내외(130~170자) 생기부 평어를 생성합니다.
        """
        # 가장 적절한 루브릭 매칭 (과목 + 단원명)
        target_subject = str(student.get("과목", "")).strip()
        target_unit = str(student.get("단원명", "")).strip()

        matched_rubric = None
        # 1. 과목 & 단원명 동시 매칭 시도
        for r in rubrics:
            r_sub = str(r.get("과목", "")).strip()
            r_unit = str(r.get("단원명", "")).strip()
            if target_subject and r_sub in target_subject or target_subject in r_sub:
                if target_unit and (r_unit in target_unit or target_unit in r_unit):
                    matched_rubric = r
                    break

        # 2. 과목만이라도 매칭 시도
        if not matched_rubric and target_subject:
            for r in rubrics:
                if target_subject in str(r.get("과목", "")):
                    matched_rubric = r
                    break

        # 3. 매칭 안 될 경우 첫 번째 루브릭 참고
        if not matched_rubric and rubrics:
            matched_rubric = rubrics[0]

        tone_instruction = (
            "종합 평어는 학교생활기록부/통지표 기재 형식에 맞추어 '~함.', '~보임.', '~나타냄.'과 같은 간결한 명사형/개조식 문체로 작성하세요."
            if tone_style == "concise"
            else "종합 평어는 학부모 안내 및 격려용으로 알맞게 '~합니다.', '~보입니다.'와 같은 정중하고 따뜻한 해요/하십시오체로 작성하세요."
        )

        rubric_summary = json.dumps(matched_rubric, ensure_ascii=False, indent=2) if matched_rubric else "평가 기준 없음"

        grading_prompt = f"""
당신은 대한민국 초등학교 교사로서 학생의 수행평가 시험지를 엄정하고 따뜻하게 채점하고 평어를 작성하는 AI입니다.

[적용 평가 기준 및 모범 답안]
{rubric_summary}

[학생 정보]
- 학년/반/번호: {student.get('학년','')}학년 {student.get('반','')}반 {student.get('번호','')}번
- 이름: {student.get('이름','')}
- 과목: {student.get('과목','')}
- 단원명: {student.get('단원명','')}

[채점 지침]
1. 학생의 시험지 이미지를 면밀히 확인하세요. 손글씨 서술형 답안, 수학 풀이식 및 분수 색칠 그림, 꺾은선그래프 점 찍기 및 선 연결, 사회 홍보물 포스터/그림 및 슬로건 문구 등 시각적인 모든 요소를 정확하게 파독하세요.
2. 학생의 답안을 모범답안 및 채점기준과 비교하여 정답 여부와 성취 수준을 종합적으로 분석하세요.
3. [등급 산출]: 반드시 '매우잘함', '잘함', '보통', '노력요함' 4개 등급 중 단 하나를 선택하여 부여하세요.
4. [답안 요약 및 채점 근거]: 학생이 쓴 답안 내용과 잘된 점, 오류나 보완할 점을 핵심적으로 기술하세요.
5. [평가 의견 (150자 평어)]:
   - 학생이 실제로 작성한 답안 내용의 구체적 사례, 성취수준, 그리고 칭찬과 발전 과제를 균형 있게 담아 작성하세요.
   - 글자 수는 공백 포함 **반드시 130자 ~ 170자 사이 (150자 내외)** 로 정밀하게 작성하세요.
   - {tone_instruction}

[응답 JSON 스키마]
{{
  "과목": "{student.get('과목','') or (matched_rubric.get('과목','') if matched_rubric else '')}",
  "단원명": "{student.get('단원명','') or (matched_rubric.get('단원명','') if matched_rubric else '')}",
  "답안요약_및_근거": "학생이 작성한 주요 답안 요약 및 기준 대비 채점 이유",
  "등급": "매우잘함 / 잘함 / 보통 / 노력요함 중 하나",
  "평가의견": "공백 포함 130~170자 내외의 종합 평어 문장"
}}
"""

        contents = [grading_prompt]
        for p_idx, page in enumerate(student["pages"]):
            contents.append(f"[학생 시험지 {p_idx+1}페이지 (파일명: {page['filename']} p.{page['page_num']})]")
            contents.append(page["image"])

        response_text = self._call_with_retry(contents, json_mode=True)
        cleaned = self._clean_json_text(response_text)
        
        try:
            grade_result = json.loads(cleaned)
        except Exception as e:
            # 보조 정규식 파싱
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                grade_result = json.loads(match.group(0))
            else:
                grade_result = {
                    "과목": student.get("과목", ""),
                    "단원명": student.get("단원명", ""),
                    "답안요약_및_근거": "채점 분석 중 일시적 파싱 오류 발생",
                    "등급": "보통",
                    "평가의견": "학생의 성취 수준을 평가 기준에 따라 종합적으로 검토하였으며 지속적인 관심과 노력이 요구됨."
                }

        # 4단계 등급 유효성 보정
        valid_grades = ["매우잘함", "잘함", "보통", "노력요함"]
        raw_grade = str(grade_result.get("등급", "")).replace(" ", "")
        final_grade = "보통"
        for vg in valid_grades:
            if vg in raw_grade:
                final_grade = vg
                break
        grade_result["등급"] = final_grade

        # 평가의견 및 글자 수 계산
        comment = str(grade_result.get("평가의견", "")).strip()
        grade_result["평가의견"] = comment
        grade_result["글자수"] = len(comment)

        return grade_result
