"""Stage 4 대상 메쉬 풀 로딩 — manifest 기반 선별.

`tools/convert_gso_to_usd.py`가 만든 `manifest.json`을 읽어 학습에 쓸 USD 목록을
돌려준다. 변환 자체는 전부 해두고 **선별은 이 지점에서** 한다 — 변환은 싸고,
어떤 물체를 쓸지는 실험 조건이라 바꿔가며 시험해야 하기 때문이다.

납작한 형상을 왜 걸러내는가
---------------------------
NBV는 "어느 시점을 골라야 새 표면이 보이는가"를 푸는 문제다. 평면에 가까운
물체(메인보드, 테이프, 게임팩 등)는 앞뒤 두 시점이면 표면이 거의 다 드러나므로
**시점 선택이라는 문제 자체가 성립하지 않는다**. 그런 물체가 풀에 섞이면 정책이
"아무 데나 두 번 보면 된다"를 배우고, 그게 다른 물체에서는 통하지 않는다.

종횡비(최소변/최대변)로 판정한다. 2026-09-02 GSO 1~20번 실측에서 6개가
0.25 미만이었다(메인보드 3종 0.22~0.24, 게임팩 0.11, 테이프 0.24,
안티슬립 시트 0.19).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_mesh_pool(
    manifest_path: str,
    filter_flat: bool = True,
    min_aspect: float = 0.25,
    require_texture: bool = True,
    limit: int = 0,
    offset: int = 0,
    split: str = "all",
    n_holdout: int = 91,
    split_seed: int = 20260903,
) -> list[str]:
    """manifest에서 조건을 만족하는 USD 경로 목록을 돌려준다.

    Parameters
    ----------
    filter_flat
        종횡비가 `min_aspect` 미만인 납작한 형상을 제외한다(기본 True).
    require_texture
        텍스처가 살아있는 자산만 쓴다. actor 입력이 그레이스케일이라 색보다
        **휘도 변화(표면 무늬)**가 중요한데, 텍스처가 없으면 실루엣만 남아
        표면 관측이라는 과제가 성립하지 않는다.
    limit
        0보다 크면 앞에서 그만큼만. 소규모 시험용. 분할 **이후**에 적용되므로
        limit을 바꿔도 train/holdout 경계는 움직이지 않는다.
    offset
        선택 시작 위치를 이만큼 회전시킨다. `limit`이 env 수로 묶여 있어
        한 번에 풀 전체를 볼 수 없으므로(아래 주의), offset을 옮겨가며
        여러 번 돌리면 풀 전체를 훑을 수 있다. 홀드아웃 91개를 16 env로
        평가하려면 offset 0,16,32,48,64,80으로 6회.
    split
        `"train"` / `"holdout"` / `"all"`. 수조 표적은 정의상 학습에 없던
        물체이므로, 홀드아웃 평가가 곧 배포 리허설이다. 분할은 고정 시드
        셔플로 결정론적이다 — 실행마다 경계가 달라지면 "홀드아웃"이라는 말이
        의미를 잃는다.

    Raises
    ------
    FileNotFoundError
        manifest가 없을 때. 조용히 빈 목록을 돌려주면 단일 메쉬로 학습이
        돌아가 버려서 Stage 4를 한 줄 알고 넘어가게 된다.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(
            f"메쉬 manifest가 없다: {p}\n"
            f"  tools/convert_gso_to_usd.py 로 먼저 USD를 생성할 것."
        )
    entries = json.loads(p.read_text())

    kept, dropped = [], []
    for e in entries:
        if require_texture and not e.get("has_texture", False):
            dropped.append((e["name"], "텍스처 없음"))
            continue
        if filter_flat and e.get("aspect_min_over_max", 1.0) < min_aspect:
            dropped.append((e["name"], f"납작함 {e['aspect_min_over_max']:.2f}"))
            continue
        # manifest에는 변환 당시의 절대경로가 들어 있는데, 컨테이너에서 만들고
        # 호스트나 서버에서 읽으면 경로가 달라진다. 변환기가 항상
        # `<out>/<name>/<name>.usd` 구조로 쓰므로 **manifest 위치 기준으로
        # 다시 해석**한다 — 저장된 절대경로는 참고용으로만 둔다.
        usd = p.parent / e["name"] / f"{e['name']}.usd"
        if not usd.exists():
            dropped.append((e["name"], "USD 파일 없음"))
            continue
        kept.append(str(usd))

    n_filtered = len(kept)

    # ── train / holdout 분할 ────────────────────────────────────────────
    # manifest 순서는 알파벳순이라 그대로 자르면 홀드아웃이 특정 접두사에
    # 몰린다(예: 이름이 z로 시작하는 물체들). 고정 시드로 섞고 나서 자른다.
    if split not in ("all", "train", "holdout"):
        raise ValueError(f"split은 all/train/holdout 중 하나여야 한다: {split!r}")
    if split != "all":
        import random
        order = list(kept)
        random.Random(split_seed).shuffle(order)
        if n_holdout >= len(order):
            raise ValueError(
                f"n_holdout({n_holdout})이 사용 가능한 메쉬 수({len(order)}) 이상이다"
            )
        # holdout은 읽기 좋게 정렬, train은 **셔플된 순서를 유지**한다.
        # 이유: IsaacLab의 `MultiUsdFileCfg`는 씬 생성 시 env i에 pool[i % N]을
        # 배정하고 이후 바뀌지 않는다. 즉 실제로 학습에 등장하는 물체는
        # **min(num_envs, 풀 크기)개**다. 알파벳순으로 두면 앞쪽 N개가
        # "3D_Dollhouse_*"처럼 한 가족에 몰려 다양성이 크게 줄어든다.
        # 셔플 순서를 유지하면 앞에서 N개를 잘라도 계열이 고르게 섞인다.
        hold = sorted(order[-n_holdout:]) if n_holdout > 0 else []
        train = order[:len(order) - n_holdout]
        kept = train if split == "train" else hold
        print(f"[mesh_pool]   분할 {split}: {len(kept)}개 "
              f"(학습 {len(train)} / 홀드아웃 {len(hold)}, seed={split_seed})")

    # ⚠ 주의: IsaacLab은 씬 생성 시 env i에 `pool[i % N]`을 **한 번** 배정하고
    # 이후 바꾸지 않는다(`wrappers.py:114`). 즉 한 실행에서 실제로 등장하는
    # 물체는 min(num_envs, len(kept))개다. offset은 그 창을 옮기는 손잡이다.
    if offset:
        offset %= len(kept)
        kept = kept[offset:] + kept[:offset]
    if limit > 0:
        kept = kept[:limit]

    print(f"[mesh_pool] {len(entries)}개 중 필터 통과 {n_filtered}개"
          f"{' (납작 필터 on)' if filter_flat else ' (납작 필터 off)'}"
          f" → 최종 {len(kept)}개 사용 [{split}"
          f"{f', offset {offset}' if offset else ''}"
          f"{f', limit {limit}' if limit > 0 else ''}]")
    # 제외 목록은 앞 5개만. 1,030개 풀에서는 239개가 걸려 로그를 뒤덮는다.
    for name, why in dropped[:5]:
        print(f"[mesh_pool]   제외: {name[:46]:<46} {why}")
    if len(dropped) > 5:
        from collections import Counter
        reasons = Counter(w.split()[0] for _, w in dropped)
        summary = ", ".join(f"{k} {v}개" for k, v in reasons.items())
        print(f"[mesh_pool]   ... 외 {len(dropped)-5}개 제외 ({summary})")
    if not kept:
        raise RuntimeError("조건을 만족하는 메쉬가 없다 — 필터를 완화하거나 "
                           "변환을 다시 확인할 것")
    return kept
