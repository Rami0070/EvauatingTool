import os
import sys

# Windows 콘솔 cp949 인코딩 오류 원천 방지
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import threading
import time
import webbrowser

def get_base_dir():
    """PyInstaller 번들 내부 임시 폴더(_MEIPASS) 또는 스크립트 실행 폴더 반환"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def open_browser(url: str, delay: float = 2.0):
    """서버가 기동될 시간을 준 후 웹 브라우저 자동 오픈"""
    time.sleep(delay)
    try:
        webbrowser.open_new(url)
    except Exception:
        pass

if __name__ == "__main__":
    base_dir = get_base_dir()
    app_path = os.path.join(base_dir, "app.py")

    # core 모듈 import 경로 추가
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    print("=" * 65)
    print(" [AI 수행평가 자동 채점 및 평어 생성 웹 프로그램 - 단독 실행 모드]")
    print("=" * 65)
    print("  * 서버 기동 후 기본 웹 브라우저가 자동으로 실행됩니다.")
    print("  * 접속 주소: http://localhost:8501")
    print("  * 프로그램을 종료하시려면 본 콘솔 창을 닫아주세요.")
    print("=" * 65)
    print("서버 준비 중... 잠시만 기다려 주십시오.\n")

    # 브라우저 자동 실행 백그라운드 스레드
    thread = threading.Thread(target=open_browser, args=("http://localhost:8501", 2.5), daemon=True)
    thread.start()

    from streamlit.web import bootstrap

    flag_options = {
        "global.developmentMode": False,
        "server.port": 8501,
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "server.address": "localhost"
    }

    bootstrap.run(
        main_script_path=app_path,
        is_hello=False,
        args=[],
        flag_options=flag_options
    )
