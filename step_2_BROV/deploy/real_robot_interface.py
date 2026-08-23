"""
실기체(ArduSub) MAVLink 인터페이스
====================================
`pymavlink` 필요(topside PC에 별도 설치: `pip install pymavlink`) — IsaacLab과
무관한 순수 통신 계층이라 이 파일은 `deploy/` 안에서 유일하게 torch 없이도
동작 가능(다만 반환값은 다른 deploy 모듈과 맞추기 위해 torch 텐서로 감싼다).

SERVOn_FUNCTION 자동 백업/복원
-------------------------------
`enable_passthrough()`/`disable_passthrough()`가 8개 스러스터 채널의
`SERVOn_FUNCTION`을 RCPassThru(=1)로 바꾸고 원복한다 — 메커니즘/안전 주의사항은
`~/Programing/Edo_Project/gazebosim_bluerov2_ardupilot_sitl`의
`gnc_lab/track_b/provided/motor_interface.py`(SITL에서 검증된 구현)를 그대로
따름: `disable_passthrough()`를 안 부르고 끝내면 RCPassThru가 걸린 채로 남아
GUIDED/AUTO 등 ArduSub 내장 제어기가 겉으로는 정상(arm 성공, 에러 없음)인데
스러스터엔 전혀 반영이 안 되는 조용한 실패가 난다(TRACK_A.md §3-6 실측 확인) —
`close()`가 항상 `disable_passthrough()`를 호출하는 이유.

전제 조건 (기체 측, 이 코드 밖에서 확인 — CLAUDE.md 논의 참조)
----------------------------------------------------------------------
1. 비행모드는 MANUAL — 그래야 ArduSub 자체 자세 안정화 루프가 우리 명령에
   개입하지 않는다(Sim2Swim처럼 큰 roll/pitch를 의도적으로 명령하는 태스크에 필수).
2. DVL → ArduSub EKF 융합(EK3_SRC1_VELXY 등)이 이미 구성되어 있고,
   `LOCAL_POSITION_NED`/`EKF_STATUS_REPORT`가 정상 송신되어야 한다(SITL은
   Gazebo 자체 EKF3이 이 역할을 대신하므로 별도 DVL 설정 불필요).

안전장치
--------
- `RC_CHANNELS_OVERRIDE`는 ArduSub 자체에 타임아웃 페일세이프가 있다 — 일정
  시간 메시지가 끊기면 정상 RC 입력으로 자동 복귀. 이 클래스는 그 위에 추가로
  `close()`에서 중립(1500us) 송신 후 명시적으로 release한다.
- `is_ekf_healthy`가 False면 obs_builder가 정책 추론을 멈추도록 설계할 것
  (이 클래스는 상태만 노출, 게이팅 판단은 obs_builder 책임).
"""

from __future__ import annotations

import threading
import time

import torch

_PWM_NEUTRAL_US = 1500
_PWM_RANGE_US   = 400   # [-1,1] → 1500 ± 400 = [1100, 1900]
_RC_RELEASE     = 65535   # MAVLink RC_CHANNELS_OVERRIDE 규약: 채널을 RC로 반환
_RC_NO_CHANGE   = 0
_RC_PASSTHRU_FUNCTION = 1   # ArduSub SERVOn_FUNCTION: RCPassThru

# EKF velocity_variance 임계값 — ArduPilot 자체 ekf_check 관례(0.8=불량, 10Hz*10회
# 연속 불량=1초 지속 시 실패) 참고. 여기서는 매 호출 즉시 판정하는 단순 임계값만 적용
# — 연속 실패 카운트가 필요하면 obs_builder 쪽에서 추가할 것.
_EKF_VEL_VARIANCE_BAD = 0.8


class RealRobotInterface:
    """ArduSub 기체와의 MAVLink 통신 — 텔레메트리 수신(백그라운드 스레드) + PWM 송신."""

    def __init__(
        self,
        connection_string  : str,
        thruster_rc_channels: list[int] | None = None,
        telemetry_rate_hz  : float = 50.0,
    ):
        self._conn_str = connection_string
        self._thr_channels = thruster_rc_channels or list(range(1, 9))   # 기본 RC ch1~8
        assert len(self._thr_channels) == 8, "스러스터는 8개 — RC 채널도 8개 필요"
        self._telem_rate_hz = telemetry_rate_hz

        self._master = None
        self._lock = threading.Lock()
        self._running = False
        self._recv_thread: threading.Thread | None = None

        # 최신 텔레메트리 (락으로 보호) — 아직 수신 전이면 None
        self._att_quat_ned = None   # torch (4,) [w,x,y,z]
        self._body_rates_ned = None  # torch (3,) [p,q,r]
        self._pos_ned = None         # torch (3,) [N,E,D]
        self._vel_ned = None         # torch (3,) [vN,vE,vD]
        self._ekf_vel_variance = None
        self._ekf_flags = None
        self._last_att_time = 0.0
        self._last_pos_time = 0.0
        self._armed = None
        self._param_values: dict[str, float] = {}

        self._original_servo_functions: list[int] | None = None

    # ------------------------------------------------------------------
    def connect(self) -> None:
        from pymavlink import mavutil

        self._master = mavutil.mavlink_connection(self._conn_str)
        self._master.wait_heartbeat()
        print(f"[RealRobotInterface] heartbeat 수신 — system {self._master.target_system}, "
              f"component {self._master.target_component}")

        interval_us = int(1e6 / self._telem_rate_hz)
        for msg_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
        ):
            self._master.mav.command_long_send(
                self._master.target_system, self._master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, interval_us, 0, 0, 0, 0, 0,
            )

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        """이 프로세스에서 `self._master`를 읽는 유일한 지점 — arm()/param 조회 등
        다른 메서드가 별도로 `recv_match()`를 부르면 이 스레드와 소켓을 놓고 경쟁하다가
        서로 메시지를 가로채서 타임아웃나는 버그가 났었다(HEARTBEAT를 이 스레드가
        먼저 가져가버려 arm()이 계속 실패). 그래서 HEARTBEAT/PARAM_VALUE까지 전부
        여기서만 받고 공유 상태에 적재 → 다른 메서드는 그 상태를 폴링만 한다."""
        from pymavlink import mavutil

        while self._running:
            msg = self._master.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            t = msg.get_type()
            now = time.monotonic()
            with self._lock:
                if t == "ATTITUDE_QUATERNION":
                    self._att_quat_ned = torch.tensor(
                        [msg.q1, msg.q2, msg.q3, msg.q4], dtype=torch.float32
                    )
                    self._body_rates_ned = torch.tensor(
                        [msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=torch.float32
                    )
                    self._last_att_time = now
                elif t == "LOCAL_POSITION_NED":
                    self._pos_ned = torch.tensor([msg.x, msg.y, msg.z], dtype=torch.float32)
                    self._vel_ned = torch.tensor([msg.vx, msg.vy, msg.vz], dtype=torch.float32)
                    self._last_pos_time = now
                elif t == "EKF_STATUS_REPORT":
                    self._ekf_vel_variance = float(msg.velocity_variance)
                    self._ekf_flags = int(msg.flags)
                elif t == "HEARTBEAT":
                    self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                elif t == "PARAM_VALUE":
                    param_id = msg.param_id.rstrip("\x00") if isinstance(msg.param_id, str) else msg.param_id
                    self._param_values[param_id] = msg.param_value

    # ------------------------------------------------------------------
    def snapshot(self) -> dict | None:
        """현재까지 수신된 텔레메트리 스냅샷. 아직 뭔가 안 왔으면 None."""
        with self._lock:
            if self._att_quat_ned is None or self._pos_ned is None:
                return None
            return {
                "att_quat_ned"    : self._att_quat_ned.clone(),
                "body_rates_ned"  : self._body_rates_ned.clone(),
                "pos_ned"         : self._pos_ned.clone(),
                "vel_ned"         : self._vel_ned.clone(),
                "ekf_vel_variance": self._ekf_vel_variance,
                "ekf_flags"       : self._ekf_flags,
                "att_age_s"       : time.monotonic() - self._last_att_time,
                "pos_age_s"       : time.monotonic() - self._last_pos_time,
            }

    def is_ekf_healthy(self, snap: dict) -> bool:
        """EKF_STATUS_REPORT 기반 속도 신뢰도 게이팅 (ArduPilot ekf_check 임계값 참고)."""
        if snap["ekf_vel_variance"] is None:
            return False
        return snap["ekf_vel_variance"] < _EKF_VEL_VARIANCE_BAD

    # ------------------------------------------------------------------
    # SERVOn_FUNCTION 백업/RCPassThru 전환 — motor_interface.py(Edo_Project SITL,
    # 검증됨)와 동일 절차. THRUSTER_RC_CHANNELS와 무관하게 항상 SERVO1~8을 다룬다
    # (SERVO 채널 번호 = 스러스터 출력 채널, RC override 채널 번호와는 별개 개념).
    def _get_servo_function(self, servo_num: int, timeout: float = 5.0) -> int | None:
        param_name = f"SERVO{servo_num}_FUNCTION"
        with self._lock:
            self._param_values.pop(param_name, None)   # 이전 값이 남아있으면 새 응답과 헷갈림
        self._master.mav.param_request_read_send(
            self._master.target_system, self._master.target_component,
            param_name.encode("utf-8"), -1,
        )
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if param_name in self._param_values:
                    return int(self._param_values[param_name])
            time.sleep(0.05)
        return None

    def _set_servo_function(self, servo_num: int, function: int) -> None:
        from pymavlink import mavutil

        param_name = f"SERVO{servo_num}_FUNCTION"
        self._master.mav.param_set_send(
            self._master.target_system, self._master.target_component,
            param_name.encode("utf-8"), function,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32,
        )
        time.sleep(0.1)

    def enable_passthrough(self) -> None:
        """SERVO1~8_FUNCTION을 백업 후 RCPassThru로 전환. PWM 송신 전 1회 호출."""
        self._original_servo_functions = [
            self._get_servo_function(i) or 0 for i in range(1, 9)
        ]
        for i in range(1, 9):
            self._set_servo_function(i, _RC_PASSTHRU_FUNCTION)

    def disable_passthrough(self) -> None:
        """백업해둔 SERVO_FUNCTION 원복 — 반드시 close()에서 호출됨."""
        if self._original_servo_functions is None:
            return
        for i, original in enumerate(self._original_servo_functions, start=1):
            self._set_servo_function(i, original)
        self._original_servo_functions = None

    # ------------------------------------------------------------------
    # arm/disarm — motor_interface.py와 동일 재시도 절차. 실기체에서는 보통
    # 조종사가 QGC로 직접 arm하지만, SITL 테스트 편의를 위해 제공.
    def arm(self, timeout: float = 8.0) -> bool:
        from pymavlink import mavutil

        with self._lock:
            if self._armed:
                return True   # 이미 armed 상태(예: 이전 실행이 disarm 실패) — 그대로 진행

        t0 = time.time()
        while time.time() - t0 < timeout:
            self._master.mav.command_long_send(
                self._master.target_system, self._master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 0, 0, 0, 0, 0, 0,
            )
            time.sleep(1.5)
            with self._lock:
                if self._armed:
                    return True
        return False

    def disarm(self) -> None:
        from pymavlink import mavutil

        self._master.mav.command_long_send(
            self._master.target_system, self._master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0,
        )

    # ------------------------------------------------------------------
    def send_pwm(self, pwm: torch.Tensor) -> None:
        """pwm: (8,) [-1,1] → RC_CHANNELS_OVERRIDE(1100~1900us), thruster_rc_channels 순서로 매핑."""
        us = (_PWM_NEUTRAL_US + pwm.clamp(-1.0, 1.0) * _PWM_RANGE_US).round().to(torch.int32)

        chan_us = [_RC_NO_CHANGE] * 18
        for i, ch in enumerate(self._thr_channels):
            chan_us[ch - 1] = int(us[i].item())

        self._master.mav.rc_channels_override_send(
            self._master.target_system, self._master.target_component, *chan_us
        )

    def neutral_stop(self) -> None:
        self.send_pwm(torch.zeros(8))

    def release(self) -> None:
        """override 해제 — 채널을 RC 입력으로 반환."""
        chan_us = [_RC_NO_CHANGE] * 18
        for ch in self._thr_channels:
            chan_us[ch - 1] = _RC_RELEASE
        self._master.mav.rc_channels_override_send(
            self._master.target_system, self._master.target_component, *chan_us
        )

    def close(self) -> None:
        if self._master is not None:
            self.neutral_stop()
            time.sleep(0.1)
            self.release()
            self.disable_passthrough()   # 안 부르면 GUIDED/AUTO가 조용히 무반응 상태로 남음
        self._running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)

    def __enter__(self) -> "RealRobotInterface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
