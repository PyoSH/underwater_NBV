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
        0보다 크면 앞에서 그만큼만. 소규모 시험용.

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

    if limit > 0:
        kept = kept[:limit]

    print(f"[mesh_pool] {len(entries)}개 중 {len(kept)}개 사용"
          f"{' (납작 필터 on)' if filter_flat else ' (납작 필터 off)'}")
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
