import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fire import Fire
from tqdm import tqdm

from chains import invoke_chain

load_dotenv()
tqdm.pandas(desc="Processing")

TEMP_DIR = os.getenv("TEMP_DIR", '.tmp')


def format_input_content(row: pd.DataFrame) -> str:
    return "\n".join([
        f"- 공정: {row['공정']}",
        f"- 세부공정: {row['세부공정'] if pd.notna(row['세부공정']) else '누락됨'}",
        f"- 설비: {row['설비'] if pd.notna(row['설비']) else '없음'}",
        f"- 물질: {row['물질'] if pd.notna(row['물질']) else '없음'}",
        f"- 유해위험요인: {row['유해위험요인']}",
        f"- 감소대책: {row['감소대책']}"
    ])


def listpkls() -> list:
    return [f for f in os.listdir(TEMP_DIR) if f.endswith('.pkl')]


def main(**kwargs):
    # Init time
    initial_time = datetime.now()

    # Load the DataFrame
    df = pd.read_excel(r"data\합본_전체_사고분류결과_v5.xlsx")

    # Range filtering
    df = df[kwargs.get("start", 0):kwargs.get("end", len(df))]
    
    # 데이터프레임에서 'input_content' 열이 있는지 확인
    if "sample" in kwargs:
        df = df.sample(kwargs.get("sample"), random_state=42)
    
    max_retries = kwargs.get("trial", 5)
    buffer_size = kwargs.get("buffer", 50)

    # 1. .tmp 폴더 생성
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # 2. .tmp 폴더에 있는 pkl 파일 중 수정 시간이 initial_time 이전인 파일을 찾아서 df에 neo_사고분류 업데이트
    pre_plks = [f for f in listpkls() 
                if os.path.getmtime(os.path.join(TEMP_DIR, f)) < initial_time.timestamp()]
    if pre_plks:
        tmp_df = pd.DataFrame()
        for plk in pre_plks:
            with open(os.path.join(TEMP_DIR, plk), 'rb') as f:
                df_part = pickle.load(f)
                tmp_df = pd.concat([tmp_df, df_part], ignore_index=True)
        # neo_사고분류가 있는 row는 df에 업데이트
        if 'neo_사고분류' in tmp_df.columns:
            for _, row in tmp_df.iterrows():
                cond = (
                    (df['공정'] == row['공정']) &
                    (df['세부공정'] == row['세부공정']) &
                    (df['설비'] == row['설비']) &
                    (df['물질'] == row['물질']) &
                    (df['유해위험요인'] == row['유해위험요인']) &
                    (df['감소대책'] == row['감소대책'])
                )
                df.loc[cond, 'neo_사고분류'] = row['neo_사고분류']

    # 3. mask 재설정: neo_사고분류가 없는 row만 inference
    if 'neo_사고분류' in df.columns:
        mask = df['neo_사고분류'].isna() | (df['neo_사고분류'] == '')
    else:
        mask = pd.Series([True] * len(df))

    buffer = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing with buffer")):
        if not mask.iloc[i]:
            continue
        result = invoke_chain(format_input_content(row), max_retries=max_retries)
        # row_result = row.copy()
        row_result = row.to_dict()
        row_result['neo_사고분류'] = result
        buffer.append(row_result)
        if len(buffer) >= buffer_size:
            buffer_df = pd.DataFrame(buffer)
            with open(os.path.join(TEMP_DIR, '{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))), 'wb') as f:
                pickle.dump(buffer_df, f)
            buffer = []
    if buffer:
        buffer_df = pd.DataFrame(buffer)
        with open(os.path.join(TEMP_DIR, '{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))), 'wb') as f:
            pickle.dump(buffer_df, f)

    # 4. .tmp의 모든 pkl 합쳐서 df에 neo_사고분류 업데이트 (중복 방지)
    neo_df = pd.DataFrame()
    for plk in listpkls():
        with open(os.path.join(TEMP_DIR, plk), 'rb') as f:
            plk_data = pickle.load(f)
            neo_df = pd.concat([neo_df, plk_data], ignore_index=True)

    # 5. 엑셀로 저장
    output_name = kwargs["output"] + "_" if "output" in kwargs else ""
    output_file = output_name + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"{output_file}.xlsx")
    neo_df.to_excel(output_path, index=False)
    logger.info(f"엑셀 파일 저장 완료: {output_path}")

    # 6. 엑셀 저장 성공 시 .tmp 폴더 비우기
    for plk in listpkls():
        try:
            os.remove(os.path.join('.tmp', plk))
            logger.debug(f".tmp 파일 삭제 성공: {plk}")
        except Exception as e:
            logger.warning(f".tmp 파일 삭제 실패: {plk}, {e}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # 파일 핸들러 (DEBUG 이상 기록)
    file_handler = logging.FileHandler(os.path.join('logs', '{}.log'.format(datetime.now().strftime('%Y%m%d'))), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 콘솔 핸들러 (INFO 이상만 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # 루트 로거 핸들러 제거 (중복 방지)
    logging.getLogger().handlers.clear()
    
    Fire(main)