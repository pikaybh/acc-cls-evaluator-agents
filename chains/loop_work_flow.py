from prompts import user_query_v3, evaluator_prompt_v3, final_prompt
from models import gpt_call, ollama_call

global logger


def loop_workflow_v1(user_query, evaluator_prompt, max_retries=5) -> str:
    """평가자가 생성된 요약을 통과할 때까지 최대 max_retries번 반복."""

    retries = 0
    while retries < max_retries:
        logger.debug(f"📝 사고 유형 분류 프롬프트 (시도 {retries + 1}/{max_retries})\n{user_query}\n")
        
        labels = gpt_call(user_query, model="gpt-4.1-mini")
        logger.debug(f"📝 사고 유형 분류 결과 (시도 {retries + 1}/{max_retries})\n사고 유형: {labels}\n")
        
        final_evaluator_prompt = evaluator_prompt + labels
        evaluation_result = gpt_call(final_evaluator_prompt, model="gpt-4.1").strip()
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
            final_eval = gpt_call(final_evaluator_prompt + labels, model="gpt-4.1").strip()
            if "평가결과 = PASS" in final_eval:
                return labels
            else:
                return "기타"

        # Updating the user_query for the next attempt with full history
        user_query += f"\n{retries}차 사고 유형 분류 결과: {labels}\n"
        user_query += f"\n{retries}차 사고 유형 분류 피드백:\n\n{evaluation_result}\n\n"


def loop_workflow_v2(user_query, evaluator_prompt, max_retries=5) -> str:
    """평가자가 생성된 요약을 통과할 때까지 최대 max_retries번 반복."""

    retries = 0
    while retries < max_retries:
        # Prompting the user query
        logger.debug(f"📝 사고 유형 분류 프롬프트 (시도 {retries + 1}/{max_retries})\n{user_query}\n")
        
        # Call the LLM to classify the accident type
        # labels = ollama_call(user_query, model="exaone3.5:latest")
        labels = gpt_call(user_query, model="gpt-4.1-mini").strip()
        logger.debug(f"📝 사고 유형 분류 결과 (시도 {retries + 1}/{max_retries})\n사고 유형: {labels}\n")
        
        # Call Evaluator LLM to evaluate the classification
        final_evaluator_prompt = evaluator_prompt + labels
        evaluation_result = gpt_call(final_evaluator_prompt, model="gpt-4.1").strip()
        logger.debug(f"🔍 평가 프롬프트 (시도 {retries + 1}/{max_retries})\n{final_evaluator_prompt}\n")
        logger.debug(f"🔍 평가 결과 (시도 {retries + 1}/{max_retries})\n{evaluation_result}\n")

        if "평가결과 = PASS" in evaluation_result:
            logger.debug("✅✅✅ 통과! 최종 사고 유형 분류가 승인되었습니다. ✅✅✅")
            return labels
        
        retries += 1
        logger.debug(f"🔄 재시도 필요... ({retries}/{max_retries})")

        # If max retries reached, return last attempt
        if retries >= max_retries:
            # 최종 시도에 대해 평가 LLM을 한 번 더 호출
            # final_eval = ollama_call(final_evaluator_prompt + labels, model="exaone3.5:latest").strip()
            # final_eval = gpt_call(final_evaluator_prompt + labels, model="gpt-4.1").strip()
            logger.debug("❌❌❌ 최대 재시도 횟수 도달. 마지막 분류를 반환합니다. ❌❌❌")
            return labels
            # if "평가결과 = PASS" in final_eval:
            #     logger.debug("✅✅✅ 통과! 최대 재시도 횟수 도달. 마지막 분류를 반환합니다. ✅✅✅")
            #     return labels
            # else:
            #     logger.debug("❌❌❌ 최대 재시도 횟수 도달. 마지막 분류('기타')를 반환합니다. ❌❌❌")
            #     return "기타"

        # Updating the user_query for the next attempt with full history
        user_query += f"\n{retries}차 사고 유형 분류 결과: {labels}\n"
        user_query += f"\n{retries}차 사고 유형 분류 피드백:\n\n{evaluation_result}\n\n"


def loop_workflow_v3(user_query, evaluator_prompt, max_retries=5) -> str:
    """평가자가 생성된 요약을 통과할 때까지 최대 max_retries번 반복."""

    retries = 0
    while retries < max_retries:
        # Prompting the user query
        logger.debug(f"📝 사고 유형 분류 프롬프트 (시도 {retries + 1}/{max_retries})\n{user_query}\n")
        
        # Call the LLM to classify the accident type
        labels = gpt_call(user_query, model="gpt-4.1-mini").strip()
        logger.debug(f"📝 사고 유형 분류 결과 (시도 {retries + 1}/{max_retries})\n사고 유형: {labels}\n")

        # If max retries reached, return last attempt
        if retries >= max_retries:
            logger.debug("❌❌❌ 최대 재시도 횟수 도달. 마지막 분류를 반환합니다. ❌❌❌")
            return labels
        
        # Call Evaluator LLM to evaluate the classification
        final_evaluator_prompt = evaluator_prompt + labels
        evaluation_result = gpt_call(final_evaluator_prompt, model="gpt-4.1").strip()
        logger.debug(f"🔍 평가 프롬프트 (시도 {retries + 1}/{max_retries})\n{final_evaluator_prompt}\n")
        logger.debug(f"🔍 평가 결과 (시도 {retries + 1}/{max_retries})\n{evaluation_result}\n")

        if "평가결과 = PASS" in evaluation_result:
            logger.debug("✅✅✅ 통과! 최종 사고 유형 분류가 승인되었습니다. ✅✅✅")
            return labels
        
        retries += 1
        logger.debug(f"🔄 재시도 필요... ({retries}/{max_retries})")

        # Updating the user_query for the next attempt with full history
        user_query += f"\n{retries}차 사고 유형 분류 결과: {labels}\n"
        user_query += f"\n{retries}차 사고 유형 분류 피드백:\n\n{evaluation_result}\n\n"


def invoke_chain(input_content, max_retries):
    final_labels = loop_workflow_v3(user_query_v3.format(input_content), evaluator_prompt_v3, max_retries=max_retries)
    logger.debug(f"💡💡💡 최종 결과 💡💡💡\n위험성 평가:\n{input_content}\n사고 유형 분류: {final_labels}\n")
    return final_labels


__all__ = ["invoke_chain"]