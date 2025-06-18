import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fire import Fire
from tqdm import tqdm

from utils import llm_call


load_dotenv()
tqdm.pandas(desc="Processing")

user_query = """
당신의 목표는 주어진 위험성 평가 보고서를 검토하여, 관련된 사고 유형을 분류하는 것입니다.
아래 제공된 위험성 평가 보고서에서 발생 가능성 있거나 직접·간접적으로 연관된 사고 유형을 모두 골라주세요.

사고 유형 선택 기준은 반드시 아래의 상세 지침을 따르세요:
- '사고 원인'이 아니라, '사람에게 일어난 결과'를 중심으로 사고 유형을 선택합니다.
- '떨어짐'은 사람이 높은 곳에서 낙하한 경우에만 사용하세요.
- 자재 등 사물이 낙하하여 사람에게 충격을 준 경우, '충돌 및 접촉'으로 분류하세요.
- 현장 내 장비, 차량, 지게차, 크레인 등과 충돌한 경우에도 '충돌 및 접촉'으로 분류하세요.
- 협착, 감김 등 신체의 일부가 끼이거나 감긴 경우는 '끼임'으로 분류하세요.
- 신체가 절단, 베임, 찔림을 입은 경우는 '절상(절단,찔림,베임)'으로 분류하세요.
- 폭발로 인한 부상은 '화상'으로 분류하세요.
- 어떠한 사고 유형도 명확히 드러나지 않는 경우, '기타'로 분류합니다.

사고 유형은 반드시 아래 리스트에서만 선택해야 합니다. 
이전 분류 결과와 피드백이 주어진 경우, 그 내용을 참고하여 더 정확하게 개선된 사고 유형 목록을 작성하세요.

- 감전
- 기타
- 깔림
- 끼임
- 넘어짐
- 떨어짐
- 충돌 및 접촉
- 절상(절단,찔림,베임)
- 질병
- 질식
- 화상

해당될 수 있는 모든 사고 유형을 한글로, 세미콜론으로 구분하여 출력하세요.

위험성 평가 보고서:
{}
"""

evaluator_prompt = """
다음 사고 유형 분류 결과를 평가하십시오:

## 평가기준
1. 보고서에서 명시적으로 언급되거나, 합리적으로 추론 가능한 모든 사고 유형이 포함되어 있는가?
   - 직접적으로 언급된 사고 유형은 반드시 포함해야 합니다.
   - 명시적으로 드러나지 않더라도, 보고서의 내용상 논리적으로 충분히 연관성이 인정된다면 포함할 수 있습니다.
   - 근거 없이 포함된 유형은 감점 대상입니다.

2. 사고 유형 선정의 정확성 및 일관성 (아래 지침 준수 여부 확인)
   - '사고 원인'이 아닌 '사람에게 일어난 결과' 중심으로 유형을 선택했는가?
   - '떨어짐'은 반드시 사람이 높은 곳에서 낙하한 경우에만 사용했는가?
   - 자재 등 사물의 낙하는 '충돌 및 접촉'으로 분류했는가?
   - 장비·차량·크레인 등과의 충돌은 '충돌 및 접촉'으로 분류했는가?
   - 협착·감김은 '끼임'으로, 신체 절단·베임·찔림은 '절상(절단,찔림,베임)'으로, 폭발은 '화상'으로 분류했는가?
   - 각 유형의 분류 기준이 명확히 지켜졌는지 확인하세요.

3. 포괄성 및 누락 여부
   - 사고 유형이 지나치게 누락되어, 현장의 실제 위험성이 과소평가된 경우 감점입니다.
   - 중요한 사고 유형이 빠졌다면, 반드시 어떤 내용 때문에 추가되어야 하는지 근거를 명시하세요.

4. 표현 및 포맷
   - 사고 유형은 반드시 한글로, 세미콜론(;)으로 구분되어야 합니다.
   - 리스트에 없는 임의의 유형은 포함시키지 마세요.

## 평가결과 응답예시
- 모든 기준이 충족되었으면 "평가결과 = PASS"를 출력하세요.
- 문제점이 있다면, 구체적으로 어떤 사고 유형이 잘못 포함되었거나 누락되었는지 지적하고, 반드시 개선 방향을 제시하세요.
- 주요 기준을 충족하지 못한 경우 "평가결과 = FAIL"을 출력하고, 반드시 핵심적인 문제점을 설명하세요.

사고 유형 분류 결과:
"""

final_evaluator_prompt = """
최대 시도 횟수에 도달하였습니다.
현재의 평가 결과에 따라 사고 유형 분류 결과가 최종적으로 결정됩니다.
최종 평가 결과가 "평가결과 = FAIL"인 경우, 자동으로 "기타"로 분류됩니다.
{}
""".format(evaluator_prompt)


def loop_workflow(user_query, evaluator_prompt, max_retries=5) -> str:
    """평가자가 생성된 요약을 통과할 때까지 최대 max_retries번 반복."""

    retries = 0
    while retries < max_retries:
        logger.debug(f"📝 사고 유형 분류 프롬프트 (시도 {retries + 1}/{max_retries})\n{user_query}\n")
        
        labels = llm_call(user_query, model="gpt-4.1-mini")
        logger.debug(f"📝 사고 유형 분류 결과 (시도 {retries + 1}/{max_retries})\n사고 유형: {labels}\n")
        
        final_evaluator_prompt = evaluator_prompt + labels
        evaluation_result = llm_call(final_evaluator_prompt, model="gpt-4.1").strip()
        logger.debug(f"🔍 평가 프롬프트 (시도 {retries + 1}/{max_retries})\n{final_evaluator_prompt}\n")
        logger.debug(f"🔍 평가 결과 (시도 {retries + 1}/{max_retries})\n{evaluation_result}\n")

        if "평가결과 = PASS" in evaluation_result:
            logger.debug("✅ 통과! 최종 사고 유형 분류가 승인되었습니다.")
            return labels
        
        retries += 1
        logger.debug(f"🔄 재시도 필요... ({retries}/{max_retries})")

        # If max retries reached, return last attempt
        if retries >= max_retries:
            logger.debug("❌ 최대 재시도 횟수 도달. 마지막 분류를 반환합니다.")
            # 최종 시도에 대해 평가 LLM을 한 번 더 호출
            final_eval = llm_call(final_evaluator_prompt + labels, model="gpt-4.1").strip()
            if "평가결과 = PASS" in final_eval:
                return labels
            else:
                return "기타"

        # Updating the user_query for the next attempt with full history
        user_query += f"\n{retries}차 사고 유형 분류 결과: {labels}\n"
        user_query += f"\n{retries}차 사고 유형 분류 피드백:\n\n{evaluation_result}\n\n"


def invoke_chain(input_content, max_retries):
    final_labels = loop_workflow(user_query.format(input_content), evaluator_prompt, max_retries=max_retries)
    logger.debug(f"💡 최종 결과:\n위험성 평가:\n{input_content}\n사고 유형 분류: {final_labels}\n")
    return final_labels


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
    return [f for f in os.listdir('.tmp') if f.endswith('.pkl')]


# def load_df(check_tmp: bool) -> pd.DataFrame:
#     plks = [f for f in os.listdir(".tmp") if f.endswith(".pkl")]
#     if plks and check_tmp:
#         df = pd.DataFrame()
#         for plk in plks:
#             with open(os.path.join(".tmp", plk), "rb") as f:
#                 df = pd.concat([df, pickle.load(f)])
#     else:
#         df = pd.read_excel(r"data\합본_전체_사고분류결과_v5.xlsx")
#     return df


def main(*args, 
         # check_tmp: bool = True, 
         **kwargs):
    # Init time
    initial_time = datetime.now()

    # Load the DataFrame
    df = pd.read_excel(r"data\합본_전체_사고분류결과_v5.xlsx")

    if "start" in kwargs and "end" in kwargs:
        start = kwargs["start"]
        end = kwargs["end"]
        df = df[start:end]
    elif "start" in kwargs:
        start = kwargs["start"]
        df = df[start:]  # 샘플링을 위해 특정 범위로 슬라이싱
    elif "end" in kwargs:
        end = kwargs["end"]
        df = df[:end]
    
    # 데이터프레임에서 'input_content' 열이 있는지 확인
    if "sample" in kwargs:
        size = kwargs["sample"]
        df = df.sample(size, random_state=42)
    
    max_retries = kwargs["trial"] if "trial" in kwargs else 5
    buffer_size = kwargs["buffer"] if "buffer" in kwargs else 50

    # 1. .tmp 폴더 생성
    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')

    # 2. .tmp에 저장된 pkl 파일이 있으면 모두 불러와서 하나의 DataFrame으로 합침 (initial_time 이후 생성된 파일만)
    # all_plks = [f for f in os.listdir('.tmp') if f.endswith('.pkl')]
    # if "fix" in kwargs:
    #     tmp_df = pd.DataFrame()
    #     for plk in all_plks:
    #         with open(os.path.join('.tmp', plk), 'rb') as f:
    #             df_part = pickle.load(f)
    #             tmp_df = pd.concat([tmp_df, df_part], ignore_index=True)
    #     df = tmp_df
    # 
    pre_plks = [f for f in listpkls() if os.path.getmtime(os.path.join('.tmp', f)) < initial_time.timestamp()]
    if pre_plks:
        tmp_df = pd.DataFrame()
        for plk in pre_plks:
            with open(os.path.join('.tmp', plk), 'rb') as f:
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
            with open(os.path.join('.tmp', '{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))), 'wb') as f:
                pickle.dump(buffer_df, f)
            buffer = []
    if buffer:
        buffer_df = pd.DataFrame(buffer)
        with open(os.path.join('.tmp', '{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))), 'wb') as f:
            pickle.dump(buffer_df, f)

    # 4. .tmp의 모든 pkl 합쳐서 df에 neo_사고분류 업데이트 (중복 방지)
    # post_plks = [f for f in all_plks if os.path.getmtime(os.path.join('.tmp', f)) >= initial_time.timestamp()]
    neo_df = pd.DataFrame()
    for plk in listpkls():
        with open(os.path.join('.tmp', plk), 'rb') as f:
            plk_data = pickle.load(f)
            neo_df = pd.concat([neo_df, plk_data], ignore_index=True)
            # if 'neo_사고분류' in df_part.columns:
            #     for idx, row in df_part.iterrows():
            #         cond = (
            #             (df['공정'] == row['공정']) &
            #             (df['세부공정'] == row['세부공정']) &
            #             (df['설비'] == row['설비']) &
            #             (df['물질'] == row['물질']) &
            #             (df['유해위험요인'] == row['유해위험요인']) &
            #             (df['감소대책'] == row['감소대책'])
            #         )
            #         df.loc[cond, 'neo_사고분류'] = row['neo_사고분류']

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