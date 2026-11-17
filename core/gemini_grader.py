import os
import time
import json
import random
import re
import concurrent.futures
import threading
from typing import List, Dict, Any, Optional
from PIL import Image

# .env 환경 설정 자동 로드 (API 키 보안 및 자동 입력)
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_file = os.path.join(_root_dir, ".env")
if os.path.exists(_env_file):
    try:
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() == "GEMINI_API_KEY" and not os.environ.get("GEMINI_API_KEY"):
                        os.environ["GEMINI_API_KEY"] = _v.strip()
    except Exception:
        pass

class GeminiGrader:
    FALLBACK_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash"
    ]

    def __init__(self, api_key: str = "", model_name: str = "gemini-3.5-flash-lite"):
        self.api_key = (api_key.strip() if api_key and api_key.strip() else os.getenv("GEMINI_API_KEY", "")).strip()
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

    def _call_with_retry(self, contents: list, json_mode: bool = True, max_retries: int = 6) -> str:
        """API Rate Limit 및 일시적 오류에 대응하며, Quota 소진 시 자동 모델 전환(Failover) 지원"""
        delay = 1.5
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
                
                # 429 RESOURCE_EXHAUSTED / Quota 초과 감지 시 다른 모델로 즉시 자동 Failover 전환
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    curr_idx = -1
                    for idx, m in enumerate(self.FALLBACK_MODELS):
                        if m in self.model_name:
                            curr_idx = idx
                            break
                    next_idx = (curr_idx + 1) % len(self.FALLBACK_MODELS)
                    next_model = self.FALLBACK_MODELS[next_idx]
                    
                    if next_model != self.model_name:
                        print(f"⚠️ [스마트 자동 전환] {self.model_name} 쿼터 제한 도달 -> {next_model} 로 자동 전환하여 계속 진행합니다.")
                        self.model_name = next_model
                        time.sleep(1.0)
                        continue

                # 503 Overloaded 또는 일시적 서버 오류 감지
                if "503" in err_str or "overloaded" in err_str:
                    jitter = random.uniform(0.5, 1.5)
                    time.sleep(delay + jitter)
                    delay *= 1.5
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

[중요 통합 및 모범답안 상세 추출 규칙]
1. 계획서와 교사별 기준안의 단원명 표기가 약간 다르더라도 성취기준과 평가요소의 맥락이 같으면 동일한 하나의 평가 단원으로 스마트하게 병합하세요.
2. [문항별 기준 정답 및 모범답안 상세 추출]:
   - '교사별 평가 기준안'에 기재된 각 문항별 정답(선다형/단답형 정답, 서술형 모범답안, 수학 계산식 및 정답 분수/소수, 도표/그래프 그리기 기준, 채점 유의점 등)을 절대로 요약하거나 축약하지 마십시오!
   - 문항 번호별(예: 1번 정답: ..., 2번 정답: ...)로 원문 그대로 상세하게 추출하여 '모범답안_및_유의점' 항목에 모두 담아주세요. 이 정답이 학생 시험지 채점의 절대적 기준 답안이 됩니다.
3. 4단계 평가기준(매우잘함, 잘함, 보통, 노력요함)의 도달 기준 문장을 누락 없이 명확히 추출하세요.

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
    "모범답안_및_유의점": "문항 1번 정답: ..., 문항 2번 정답: ... 등 문항별 상세 기준 정답 및 모범답안 전체"
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
        split_mode: str = "2pages",  # "2pages" (2장 앞/뒤), "1page" (1장), "auto" (AI 자동 판독)
        progress_callback=None,
        *args,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        학생 시험지를 분할 및 병합합니다.
        - "2pages": 앞·뒷장 2페이지 단위로 즉시 분할 (API 호출 0회, 쿼터 소모 0)
        - "1page": 1페이지 단위로 즉시 분할 (API 호출 0회, 쿼터 소모 0)
        - "auto": Vision AI로 헤더를 판독하여 자동 병합
        """
        if "split_mode" in kwargs:
            split_mode = kwargs["split_mode"]
        elif args and len(args) >= 1:
            split_mode = args[0]
        if "progress_callback" in kwargs and progress_callback is None:
            progress_callback = kwargs["progress_callback"]

        total = len(exam_pages)
        if total == 0:
            return []

        # 1. 파일별로 페이지들을 분리 (예: 국어 파일 페이지들과 수학 파일 페이지들을 절대 섞지 않음)
        files_dict = {}
        for p in exam_pages:
            fn = p.get("filename", "시험지.pdf")
            files_dict.setdefault(fn, []).append(p)

        def detect_subject_from_filename(filename_str: str) -> str:
            sub_candidates = ["국어", "수학", "사회", "과학", "영어", "도덕", "음악", "미술", "체육", "실과"]
            for s in sub_candidates:
                if s in filename_str:
                    return s
            return ""

        student_submissions = []

        # 2. 2페이지 단위 즉시 분할 모드 (초고속, 쿼터 소모 0회, 파일별 독립 분할)
        if split_mode == "2pages":
            for fn, f_pages in files_dict.items():
                f_total = len(f_pages)
                inferred_sub = detect_subject_from_filename(fn)
                for i in range(0, f_total, 2):
                    s_num = (i // 2) + 1
                    pages_chunk = []
                    for p_idx in range(i, min(i + 2, f_total)):
                        p_item = f_pages[p_idx]
                        pages_chunk.append({
                            "filename": p_item.get("filename", ""),
                            "page_num": p_item.get("page_num", 1),
                            "image": p_item.get("image")
                        })
                    student_submissions.append({
                        "student_key": f"{fn}_{s_num}",
                        "source_file": fn,
                        "학년": "",
                        "반": "",
                        "번호": str(s_num),
                        "이름": f"학생_{s_num}",
                        "과목": inferred_sub,
                        "단원명": "",
                        "pages": pages_chunk
                    })
            if progress_callback:
                progress_callback(0.4, f"파일별 2페이지 단위로 총 {len(student_submissions)}명 학생 시험지를 분할했습니다. (쿼터 소모 0)")
            return student_submissions

        # 3. 1페이지 단위 즉시 분할 모드 (초고속, 쿼터 소모 0회, 파일별 독립 분할)
        if split_mode == "1page":
            for fn, f_pages in files_dict.items():
                inferred_sub = detect_subject_from_filename(fn)
                for i, p_item in enumerate(f_pages):
                    s_num = i + 1
                    student_submissions.append({
                        "student_key": f"{fn}_{s_num}",
                        "source_file": fn,
                        "학년": "",
                        "반": "",
                        "번호": str(s_num),
                        "이름": f"학생_{s_num}",
                        "과목": inferred_sub,
                        "단원명": "",
                        "pages": [{
                            "filename": p_item.get("filename", ""),
                            "page_num": p_item.get("page_num", 1),
                            "image": p_item.get("image")
                        }]
                    })
            if progress_callback:
                progress_callback(0.4, f"파일별 1페이지 단위로 총 {len(student_submissions)}명 학생 시험지를 분할했습니다. (쿼터 소모 0)")
            return student_submissions

        # 3. Vision AI 자동 판독 모드
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
        page_analyses = [None] * total
        unknown_counter = 1
        completed_lock = threading.Lock()
        completed_count = 0

        def process_single_header(task_args):
            idx, p_item = task_args
            p_img = p_item["image"]
            
            # 헤더 판독 최적화: 시험지 상단 28% 영역만 잘라서 전송 (용량 1/4 감소 & 처리속도 급상승)
            try:
                w, h = p_img.size
                header_crop = p_img.crop((0, 0, w, int(h * 0.28)))
            except Exception:
                header_crop = p_img

            contents = [
                header_prompt,
                f"파일명: {p_item['filename']}, 페이지: {p_item['page_num']}",
                header_crop
            ]
            
            try:
                resp = self._call_with_retry(contents, json_mode=True)
                cleaned = self._clean_json_text(resp)
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

            return idx, {
                "page_info": p_item,
                "header": header_data
            }

        # 4개 동시 스레드로 병렬 판독 (88페이지도 30~50초 내에 초고속 완료)
        max_workers = min(4, total) if total > 0 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(i, p) for i, p in enumerate(exam_pages)]
            futures = [executor.submit(process_single_header, t) for t in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    page_analyses[idx] = result
                except Exception as ex:
                    pass

                with completed_lock:
                    completed_count += 1
                    if progress_callback:
                        pct = 0.1 + (completed_count / total) * 0.4
                        progress_callback(pct, f"시험지 페이지 헤더 판독 중... ({completed_count}/{total} 페이지)")

        # 다중 페이지 병합 처리
        student_submissions = []
        current_student = None

        for idx, item in enumerate(page_analyses):
            if item is None:
                item = {
                    "page_info": exam_pages[idx],
                    "header": {"is_new_student_header": False, "번호": "미상", "이름": "미상"}
                }
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

    def _recognize_name_from_crop(self, header_img: Image.Image) -> str:
        """인적사항 크롭 이미지에서 손글씨 학생 이름만 정밀 핀포인트 추출 (2차 안전장치)"""
        try:
            name_prompt = (
                "이 이미지는 초등학교 시험지 상단의 [반, 번호, 이름] 기재란입니다. "
                "연필이나 볼펜으로 적힌 학생의 손글씨 이름(보통 2~4글자 한글 이름, 예: 김민수, 이서연 등)만 정확히 단답형으로 출력하세요. "
                "설명, 따옴표, 접두어 없이 이름만 한글로 출력하세요. "
                "손글씨 이름이 전혀 없거나 빈칸이면 '미상'이라고만 출력하세요."
            )
            resp = self._call_with_retry([name_prompt, header_img], json_mode=False, max_retries=3)
            cleaned = resp.strip().replace('"', '').replace("'", "").replace(".", "").strip()
            # 한글 2~4글자 이름 추출
            hangul_match = re.search(r'[가-힣]{2,4}', cleaned)
            if hangul_match:
                return hangul_match.group(0)
            return cleaned if cleaned and cleaned != "미상" else ""
        except Exception:
            return ""

    def _clean_comment(self, comment: str, student_name: str = "") -> str:
        """
        초등 생기부 공식 규격에 맞게 평어에서 '학생' 지칭 주어와 '문제 번호 및 선택지 기호'를 완벽히 제거 및 정제합니다.
        """
        if not comment:
            return ""
        text = comment.strip()

        # 1. 특정 학생 이름 주어 제거
        if student_name and len(student_name) >= 2 and student_name not in ["미상", "학생"]:
            name_esc = re.escape(student_name)
            text = re.sub(rf'^{name_esc}\s*(학생은|학생이|학생의|학생도|학생에게|학생|은|는|이가|이)?\s*', '', text)
            text = re.sub(rf'\b{name_esc}\s*(학생은|학생이|학생의|학생도|학생에게|학생|은|는|이가|이)?\s*', '', text)

        # 2. 일반 학생 지칭 표현 제거 (문두 및 문중)
        student_patterns = [
            r'^(위\s*학생은|본\s*학생은|이\s*학생은|해당\s*학생은|학생은|위\s*학생의|본\s*학생의|학생이|학생도|학생에게)\s*',
            r'(위\s*학생은|본\s*학생은|이\s*학생은|해당\s*학생은)\s*',
            r'\b학생은\s*',
            r'\b학생이\s*',
            r'\b학생의\s*',
            r'\b학생에게\s*',
            r'\b학생도\s*',
            r'\b학생\s*',
        ]
        for pat in student_patterns:
            text = re.sub(pat, '', text)

        # 3. 조사 목록 확장: 에서도, 으로, 로, 에서, 의, 을, 를, 은, 는, 에, 과, 와, 도
        postpositions = r'(에서도|으로|로|에서|의|을|를|은|는|에|과|와|도)?\s*'
        text = re.sub(rf'\b\d+\s*번\s*(문항|문제)?\s*{postpositions}', '', text)
        text = re.sub(rf'\b(문항|문제)\s*\d+\s*{postpositions}', '', text)
        text = re.sub(rf'\b제\s*\d+\s*(문항|문제)\s*{postpositions}', '', text)

        # 4. 선택지 및 기호 제거: ①~⑮, ㉠~㉭, ㉮~㉻
        text = re.sub(rf'[①-⑮㉠-㉭㉮-㉻]\s*(번)?\s*(문항|문제|선택지|보기|기호)?\s*{postpositions}', '', text)
        text = re.sub(rf'\b(선택지|보기|기호)\s*([①-⑮㉠-㉭㉮-㉻A-Da-d1-5번]|에서도|에서|의|를|을|에)?\s*', '', text)

        # 5. 문두 찌꺼기 및 기호 정제
        text = re.sub(r'^[,\s\-\.\·\:\;\~\^]+', '', text)
        text = re.sub(r'^(또한|그리고|하지만|그러나|따라서|이에|이와\s*함께)\s*,\s*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # =========================================================================
    # Step 3: 기준 기반 채점, 4단계 등급 부여 및 100자 내외 개별 맞춤 평어 생성
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
        100자 내외(90~110자) 생기부 맞춤 평어를 생성합니다.
        """
        target_subject = str(student.get("과목", "")).strip()
        target_unit = str(student.get("단원명", "")).strip()
        scan_order_num = str(student.get("번호", "1")).strip()

        # 1. 학생의 과목에 일치하는 모든 교사별 평가 기준안 수집
        relevant_rubrics = []
        if target_subject:
            for r in rubrics:
                r_sub = str(r.get("과목", "")).strip()
                if target_subject in r_sub or r_sub in target_subject:
                    relevant_rubrics.append(r)

        # 2. 과목 매칭이 없으면 전체 기준안 제공
        if not relevant_rubrics:
            relevant_rubrics = rubrics

        tone_instruction = (
            "종합 평어는 학교생활기록부/통지표 기재 형식에 맞추어 '~함.', '~보임.', '~나타냄.'과 같은 간결한 명사형/개조식 문체로 작성하세요."
            if tone_style == "concise"
            else "종합 평어는 학부모 안내 및 격려용으로 알맞게 '~합니다.', '~보입니다.'와 같은 정중하고 따뜻한 해요/하십시오체로 작성하세요."
        )

        rubric_summary = json.dumps(relevant_rubrics, ensure_ascii=False, indent=2) if relevant_rubrics else "평가 기준 없음"

        # 3. 시험지 1페이지 상단 인적사항 영역 선명 크롭 (이름/번호 핀포인트 인식용)
        first_page_img = student["pages"][0]["image"] if student.get("pages") else None
        header_crop = None
        if first_page_img:
            try:
                w, h = first_page_img.size
                header_crop = first_page_img.crop((0, 0, w, int(h * 0.22)))
            except Exception:
                header_crop = None

        grading_prompt = f"""
당신은 대한민국 초등학교 교사로서 학생의 수행평가 시험지를 엄정하고 따뜻하게 채점하고, 학생마다 고유한 100자 맞춤 평어를 작성하는 AI입니다.

[교사별 평가 기준안 (문항별 기준 정답 및 모범답안)]
{rubric_summary}

[학생 기본 정보]
- 과목: {target_subject or '시험지 표기 과목'}
- 스캔 순번: {scan_order_num}번 (※ 시험지는 1번부터 차례대로 정리되어 있습니다)

[채점 및 판독 핵심 지침]
★ 절대 준수 사항 1: [교사별 평가 기준안 기준 1:1 대조 채점]
- 반드시 위 [교사별 평가 기준안]에 명시된 문항별 정답, 계산식, 서술형 모범답안을 절대적인 채점 기준으로 삼으세요.
- 학생 답안을 위 기준안의 정답과 1:1로 직접 대조하여 정오답을 엄정하게 판정하고, 4단계 성취수준(매우잘함, 잘함, 보통, 노력요함) 중 하나를 부여하세요.
- '답안요약_및_근거'란에는 기준안의 정답 대비 학생이 맞힌 문항과 틀린 문항, 구체적인 오류 원인을 상세히 기술하세요.

★ 절대 준수 사항 2: [학생 인적사항(이름/번호) 정확 판독 - 지문 속 인물 혼동 절대 금지]
- 제공된 [이미지 1]은 시험지 1페이지 상단의 '반, 번호, 이름' 기재란 확대본입니다.
- '이름: ( )' 란에 학생이 연필/볼펜으로 쓴 실제 손글씨 이름을 정확하게 읽어 응답 JSON의 '이름'에 적으세요. (초등학생 이름은 보통 2~4글자 한글)
- ★ 주의: 시험 문제나 지문 속 이야기의 등장인물(소설 주인공, 역사 인물, 지문 속 대화 상대 등)을 절대로 학생 이름으로 쓰지 마십시오! 오직 상단 '이름:' 란에 손글씨로 쓴 이름만 적어야 합니다.
- '번호: ( )' 란의 손글씨 번호도 읽으세요. (스캔 순서상 기본 번호는 {scan_order_num}번입니다.)

★ 절대 준수 사항 3: [100자 내외 개별화 평어 작성 - '학생' 언급 및 '문제 번호/선택지 기호' 일체 언급 금지]
- 분량 준수: 공백 포함 **반드시 90자 ~ 110자 사이 (100자 내외)** 로 정밀하게 작성하세요. (85자 미만 또는 115자 초과 엄격히 금지)
- [금지 1: 학생 지칭 주어 절대 사용 금지 (초등 학교생활기록부 공식 기재 규정)]
  * '학생은', '위 학생은', '본 학생은', '이 학생은', '해당 학생은', 'OO 학생은', 학생 이름('민준이는' 등) 등 학생을 지칭하는 단어를 절대로 사용하지 마십시오!
  * 평어 문장 전체에서 '학생'이라는 단어 자체를 일절 쓰지 마십시오.
  * 반드시 주어를 생략하고 구체적인 교과 학습 행동이나 성취 내용으로 문장을 곧바로 시작하십시오.
    - [잘못된 예]: "위 학생은 분수의 덧셈을 바르게 계산하고..." (X)
    - [잘못된 예]: "김민준 학생은 중심 문장을 잘 찾았으나..." (X)
    - [올바른 예]: "분수의 덧셈과 뺄셈 원리를 명확하게 이해하고 통분 과정을 차근차근 전개하여 계산을 정확히 해결함." (O)
- [금지 2: 문제 번호 및 선택지 기호 언급 절대 금지]
  * '1번 문항', '2번 문제', '3번에서', '4번 답안', '문항 1' 등 시험지 문제 번호를 절대로 언급하지 마십시오.
  * '①번', '②', '㉠', '㉡', '보기' 등 선택지 기호를 절대로 언급하지 마십시오.
  * 문제 번호나 기호 대신 학생이 수행한 구체적인 '교과 학습 내용 및 핵심 개념 명칭'으로 서술하십시오.
    - [잘못된 예]: "1번 문항에서 통분을 잘하고 3번 문제에서 오답이 나옴." (X)
    - [올바른 예]: "진분수의 덧셈 과정에서 공통분모를 정확히 찾아 계산하였으나, 대분수를 가분수로 변환하는 연산 과정에서 실수가 나타남." (O)
- [학생별 고유성(개별화) 절대 필수]:
  * 모든 학생의 평어가 판에 박힌 듯 비슷하거나 똑같아서는 절대 안 됩니다.
  * 학생이 실제로 시험지에 적은 손글씨 답안의 구체적인 어휘, 표현, 풀이식이나 선택한 소재를 평어 문장에 직접 인용하거나 반영하여 학생마다 전혀 다른 고유한 평어를 작성하세요.
    (예: 국어의 경우 학생이 적은 구체적 근거나 제안 내용 인용, 수학의 경우 학생이 푼 계산 과정 및 오류 지점 지목)
- [상투적 정형 문구 사용 금지]:
  * "성취기준에 잘 도달함", "지속적인 관심과 노력이 요구됨", "훌륭한 성취를 보임", "앞으로의 발전이 기대됨" 같은 뻔하고 모든 학생에게 복사 가능한 문구는 일절 사용하지 마십시오.
- 구성: 학생의 구체적 강점 1가지 + 보완할 구체적 학습 포인트 1가지를 자연스럽게 연결하여 작성하세요.
- {tone_instruction}

[응답 JSON 스키마]
{{
  "반": "시험지 1쪽 상단에 적힌 반 숫자",
  "번호": "{scan_order_num}",
  "이름": "시험지 1쪽 상단에 손글씨로 적힌 실제 학생 이름 (한글 2~4자)",
  "과목": "{target_subject or '시험지 과목'}",
  "단원명": "시험지 단원 번호 및 명칭",
  "답안요약_및_근거": "기준안의 정답 대비 학생이 맞힌 문항 및 오답 사유 구체적 기술",
  "등급": "매우잘함 / 잘함 / 보통 / 노력요함 중 하나",
  "평가의견": "공백 포함 90~110자 내외의 학생 언급과 문항 번호가 배제된 개별 맞춤 평어 문장"
}}
"""

        contents = [grading_prompt]
        if header_crop:
            contents.append("[★ [이미지 1] 학생 인적사항 기재란 확대 이미지 - 학생의 실제 손글씨 이름과 번호는 반드시 여기서 정확히 판독하세요]")
            contents.append(header_crop)

        for p_idx, page in enumerate(student["pages"]):
            contents.append(f"[학생 시험지 {p_idx+1}페이지 전체 (파일명: {page['filename']} p.{page['page_num']})]")
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
                    "과목": target_subject,
                    "단원명": "",
                    "답안요약_및_근거": "채점 분석 중 일시적 파싱 오류 발생",
                    "등급": "보통",
                    "평가의견": "평가 기준에 따른 주요 성취 요소를 충족하였으나 문제 해결 과정에서 세부적인 보완이 필요함."
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

        # 번호 보정: 시험지는 항상 번호 순서대로 정리되어 있으므로 스캔 순번(1, 2, 3...)을 최우선 기준으로 유지
        parsed_num = str(grade_result.get("번호", "")).strip()
        digits = re.findall(r'\d+', parsed_num)
        if digits and abs(int(digits[0]) - int(scan_order_num)) <= 2:
            grade_result["번호"] = digits[0]
        else:
            grade_result["번호"] = scan_order_num

        # 이름 보정: AI가 읽은 이름 검증 및 2차 전담 핀포인트 판독
        parsed_name = str(grade_result.get("이름", "")).strip()
        # 유효한 한글 이름인지 검증 (2~4자)
        is_valid_name = bool(re.match(r'^[가-힣]{2,4}$', parsed_name)) and parsed_name not in ["미상", "학생", "None", "null"]

        if not is_valid_name and header_crop:
            # 2차 핀포인트 전담 판독 실행 (속도가 조금 느려지더라도 확실히 인식!)
            refined_name = self._recognize_name_from_crop(header_crop)
            if refined_name and re.match(r'^[가-힣]{2,4}$', refined_name):
                grade_result["이름"] = refined_name
            else:
                grade_result["이름"] = parsed_name if parsed_name and parsed_name not in ["None", "null"] else f"학생_{scan_order_num}"
        else:
            grade_result["이름"] = parsed_name if is_valid_name else f"학생_{scan_order_num}"

        # 평어 정제 및 글자 수 재계산 (학생 지칭 주어 및 문제 번호/선택지 기호 완벽 제거)
        raw_comment = str(grade_result.get("평가의견", "")).strip()
        final_comment = self._clean_comment(raw_comment, grade_result.get("이름", ""))
        grade_result["평가의견"] = final_comment
        grade_result["글자수"] = len(final_comment)

        parsed_class = str(grade_result.get("반", "")).strip()
        if parsed_class and parsed_class not in ["None", "null", ""]:
            grade_result["반"] = parsed_class
        else:
            grade_result["반"] = student.get("반", "")

        return grade_result
