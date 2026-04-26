"""
orbital_policy.py — env.py _apply_action() 기반 궤도 순회
===========================================================

액션 구조 (action_space=9, one-hot argmax 방식):
    [0] θ+  [1] θ-  [2] φ+  [3] φ-  [4] ψ+  [5] ψ-
    [6] light-  [7] light 유지  [8] light+

좌표계 (env.py 실제 구현 기준):
    x = ψ * sin(φ) * cos(θ)
    y = ψ * sin(φ) * sin(θ)
    z = ψ * cos(φ)

    φ=0   → 정수직 위 (z축 최대)
    φ=90° → xy평면 (수평)
    
궤도 설계:
    4개 φ 레벨에서 θ를 0→2π 균등 순회
    각 레벨 내 ψ와 φ는 고정, θ만 delta_theta씩 증가
"""

import math
import torch
import numpy as np

