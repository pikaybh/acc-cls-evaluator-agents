from .v1 import *
from .v2 import *
from .v3 import *

final_prompt = """
최대 시도 횟수에 도달하였습니다.
현재의 평가 결과에 따라 사고 유형 분류 결과가 최종적으로 결정됩니다.
최종 평가 결과가 "평가결과 = FAIL"인 경우, 자동으로 "기타"로 분류됩니다.
{}
""".format(evaluator_prompt_v3)

__all__ = [
    "user_prompt_v3",
    "evaluator_prompt_v3",
    "final_prompt",
]