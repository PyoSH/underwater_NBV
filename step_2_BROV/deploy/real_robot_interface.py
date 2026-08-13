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

RC7/RC8 카메라 입력 격리
------------------------
ArduSub 기본 매핑에서 RC7_OPTION=214(Camera Pan), RC8_OPTION=213(Camera Tilt)다.
8개 스러스터를 RC1~8 RCPassThru로 직접 구동하는 동안 이 옵션을 그대로 두면
스러스터 7/8 명령이 카메라에도 입력된다. 따라서 passthrough 세션 동안 두 옵션을
백업 후 0으로 만들고, 종료 시 원래 값으로 복원한다. 카메라 tilt는 별도의
MAV_CMD_DO_MOUNT_CONTROL 명령으로만 제어한다.

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
_GCS_HEARTBEAT_PERIOD_S = 1.0
_PARAM_REQUEST_PERIOD_S = 1.0
_CAMERA_RC_OPTION_CHANNELS = (7, 8)

# 실기체 ESC/모터 배선 방향 보정. RCPassThru는 MOT_n_DIRECTION을 우회하므로
# ArduSub에서 reversed인 T2/T3/T8을 normalized PWM 최종단에서 반전한다.
_THRUSTER_REVERSAL_SIGN = torch.tensor(
    [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=torch.float32
)

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
        telemetry_rate_hz  : float = 25.0,
        thruster_reversal_sign: list[float] | torch.Tensor | None = None,
    ):
        self._conn_str = connection_string
        self._thr_channels = thruster_rc_channels or list(range(1, 9))   # 기본 RC ch1~8
        assert len(self._thr_channels) == 8, "스러스터는 8개 — RC 채널도 8개 필요"
        self._telem_rate_hz = telemetry_rate_hz
        self._thruster_reversal_sign = torch.as_tensor(
            _THRUSTER_REVERSAL_SIGN if thruster_reversal_sign is None
            else thruster_reversal_sign,
            dtype=torch.float32,
        )
        if self._thruster_reversal_sign.shape != (8,):
            raise ValueError("thruster_reversal_sign은 8개 값이어야 함")
        if not torch.all((self._thruster_reversal_sign == 1) | (self._thruster_reversal_sign == -1)):
            raise ValueError("thruster_reversal_sign 값은 +1 또는 -1이어야 함")

        self._master = None
        self._lock = threading.Lock()
        # pymavlink encoder의 sequence/signing 상태를 여러 송신 호출이 동시에
        # 건드리지 않도록 모든 MAVLink 송신을 직렬화한다.
        self._tx_lock = threading.Lock()
        self._running = False
        self._recv_thread: threading.Thread | None = None
        self._last_gcs_heartbeat_tx = 0.0

        # 최신 텔레메트리 (락으로 보호) — 아직 수신 전이면 None
        self._att_quat_ned = None   # torch (4,) [w,x,y,z]
        self._body_rates_ned = None  # torch (3,) [p,q,r]
        self._pos_ned = None         # torch (3,) [N,E,D]
        self._vel_ned = None         # torch (3,) [vN,vE,vD]
        self._ekf_vel_variance = None
        self._ekf_flags = None
        self._last_att_time = 0.0
        self._last_pos_time = 0.0
        self._last_ekf_time = 0.0
        self._att_seq = 0
        self._pos_seq = 0
        self._servo_output_us = None
        self._last_servo_time = 0.0
        self._armed = None
        self._custom_mode = None
        self._heartbeat_system = None
        self._heartbeat_component = None
        self._param_values: dict[str, float] = {}

        self._original_servo_functions: list[int] | None = None
        self._original_camera_rc_options: dict[int, int] | None = None
        self._camera_mount_mavlink_active = False

    # ------------------------------------------------------------------
    def connect(self) -> None:
        from pymavlink import mavutil

        self._master = mavutil.mavlink_connection(
            self._conn_str,
            source_system=255,
            source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
        )
        deadline = time.monotonic() + 10.0
        heartbeat = None
        while time.monotonic() < deadline:
            # udpout은 BlueOS UDP Server가 이 클라이언트의 주소를 알 수 있도록
            # 반드시 우리가 먼저 패킷을 보내야 한다. udpin에서는 peer가 아직
            # 없어도 pymavlink가 조용히 무시하므로 두 연결 방식 모두 지원한다.
            if time.monotonic() - self._last_gcs_heartbeat_tx >= 0.5:
                self._send_gcs_heartbeat()
            candidate = self._master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if (candidate is not None and candidate.get_srcSystem() != 0
                    and candidate.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1):
                heartbeat = candidate
                break
        if heartbeat is None:
            raise TimeoutError("system!=0, autopilot component=1 heartbeat를 찾지 못함")
        self._master.target_system = int(heartbeat.get_srcSystem())
        self._master.target_component = int(heartbeat.get_srcComponent())
        self._armed = bool(
            heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        self._custom_mode = int(heartbeat.custom_mode)
        self._heartbeat_system = self._master.target_system
        self._heartbeat_component = self._master.target_component
        print(f"[RealRobotInterface] heartbeat 수신 — system {self._master.target_system}, "
              f"component {self._master.target_component}")

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self._request_telemetry_streams()

    def _send_gcs_heartbeat(self) -> None:
        """BlueOS/ArduPilot에 이 연결을 GCS endpoint로 등록하고 유지한다."""
        from pymavlink import mavutil

        with self._tx_lock:
            self._master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
        self._last_gcs_heartbeat_tx = time.monotonic()

    def _request_telemetry_streams(self) -> None:
        """정책에 필요한 MAVLink 메시지만 지정 주기로 요청한다."""
        from pymavlink import mavutil

        interval_us = int(1e6 / self._telem_rate_hz)
        with self._tx_lock:
            for msg_id in (
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION,
                mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
                mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW,
            ):
                self._master.mav.command_long_send(
                    self._master.target_system, self._master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, interval_us, 0, 0, 0, 0, 0,
                )

    def _recv_loop(self) -> None:
        """이 프로세스에서 `self._master`를 읽는 유일한 지점 — arm()/param 조회 등
        다른 메서드가 별도로 `recv_match()`를 부르면 이 스레드와 소켓을 놓고 경쟁하다가
        서로 메시지를 가로채서 타임아웃나는 버그가 났었다(HEARTBEAT를 이 스레드가
        먼저 가져가버려 arm()이 계속 실패). 그래서 HEARTBEAT/PARAM_VALUE까지 전부
        여기서만 받고 공유 상태에 적재 → 다른 메서드는 그 상태를 폴링만 한다."""
        from pymavlink import mavutil

        while self._running:
            if time.monotonic() - self._last_gcs_heartbeat_tx >= _GCS_HEARTBEAT_PERIOD_S:
                self._send_gcs_heartbeat()
            msg = self._master.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            t = msg.get_type()
            now = time.monotonic()
            with self._lock:
                if t == "ATTITUDE_QUATERNION":
                    quat = torch.tensor(
                        [msg.q1, msg.q2, msg.q3, msg.q4], dtype=torch.float32
                    )
                    # q와 -q는 같은 자세다. MAVLink 표현 부호가 바뀌어도 observation과
                    # q_e 적분이 불연속이 되지 않도록 이전 샘플과 같은 hemisphere로 맞춘다.
                    norm = quat.norm()
                    if torch.isfinite(quat).all() and norm > 1e-6:
                        quat = quat / norm
                        if self._att_quat_ned is not None and torch.dot(quat, self._att_quat_ned) < 0:
                            quat = -quat
                    self._att_quat_ned = quat
                    self._body_rates_ned = torch.tensor(
                        [msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=torch.float32
                    )
                    self._last_att_time = now
                    self._att_seq += 1
                elif t == "LOCAL_POSITION_NED":
                    self._pos_ned = torch.tensor([msg.x, msg.y, msg.z], dtype=torch.float32)
                    self._vel_ned = torch.tensor([msg.vx, msg.vy, msg.vz], dtype=torch.float32)
                    self._last_pos_time = now
                    self._pos_seq += 1
                elif t == "EKF_STATUS_REPORT":
                    self._ekf_vel_variance = float(msg.velocity_variance)
                    self._ekf_flags = int(msg.flags)
                    self._last_ekf_time = now
                elif t == "SERVO_OUTPUT_RAW":
                    self._servo_output_us = torch.tensor(
                        [getattr(msg, f"servo{i}_raw") for i in range(1, 9)],
                        dtype=torch.int32,
                    )
                    self._last_servo_time = now
                elif t == "HEARTBEAT":
                    if (msg.get_srcSystem() == self._master.target_system
                            and msg.get_srcComponent() == self._master.target_component):
                        self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        self._custom_mode = int(msg.custom_mode)
                        self._heartbeat_system = int(msg.get_srcSystem())
                        self._heartbeat_component = int(msg.get_srcComponent())
                elif t == "PARAM_VALUE":
                    if isinstance(msg.param_id, bytes):
                        param_id = msg.param_id.decode("utf-8", errors="replace").rstrip("\x00")
                    else:
                        param_id = msg.param_id.rstrip("\x00")
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
                "ekf_age_s"       : (
                    float("inf") if self._last_ekf_time == 0.0
                    else time.monotonic() - self._last_ekf_time
                ),
                "att_rx_time"     : self._last_att_time,
                "pos_rx_time"     : self._last_pos_time,
                "att_seq"         : self._att_seq,
                "pos_seq"         : self._pos_seq,
            }

    def is_ekf_healthy(self, snap: dict) -> bool:
        """EKF_STATUS_REPORT 기반 속도 신뢰도 게이팅 (ArduPilot ekf_check 임계값 참고)."""
        if snap["ekf_vel_variance"] is None:
            return False
        return snap["ekf_vel_variance"] < _EKF_VEL_VARIANCE_BAD

    def control_snapshot(self) -> dict:
        """자세/EKF와 무관하게 arm, mode, 실제 servo output 상태를 반환한다."""
        with self._lock:
            return {
                "armed": self._armed,
                "custom_mode": self._custom_mode,
                "heartbeat_system": self._heartbeat_system,
                "heartbeat_component": self._heartbeat_component,
                "servo_output_us": (
                    None if self._servo_output_us is None else self._servo_output_us.clone()
                ),
                "servo_age_s": (
                    float("inf") if self._last_servo_time == 0.0
                    else time.monotonic() - self._last_servo_time
                ),
            }

    # ------------------------------------------------------------------
    # SERVOn_FUNCTION 백업/RCPassThru 전환 — motor_interface.py(Edo_Project SITL,
    # 검증됨)와 동일 절차. THRUSTER_RC_CHANNELS와 무관하게 항상 SERVO1~8을 다룬다
    # (SERVO 채널 번호 = 스러스터 출력 채널, RC override 채널 번호와는 별개 개념).
    def _get_servo_function(self, servo_num: int, timeout: float = 5.0) -> int | None:
        param_name = f"SERVO{servo_num}_FUNCTION"
        value = self.get_parameter(param_name, timeout=timeout)
        return None if value is None else int(value)

    def get_parameter(self, param_name: str, timeout: float = 5.0) -> float | None:
        """수신 스레드와 경쟁하지 않고 MAVLink parameter 하나를 읽는다."""
        with self._lock:
            self._param_values.pop(param_name, None)
        deadline = time.monotonic() + timeout
        next_request = 0.0
        while time.monotonic() < deadline:
            with self._lock:
                if param_name in self._param_values:
                    return float(self._param_values[param_name])
            now = time.monotonic()
            if now >= next_request:
                with self._tx_lock:
                    self._master.mav.param_request_read_send(
                        self._master.target_system, self._master.target_component,
                        param_name.encode("utf-8"), -1,
                    )
                next_request = now + _PARAM_REQUEST_PERIOD_S
            time.sleep(0.05)
        return None

    def _set_servo_function(self, servo_num: int, function: int) -> None:
        self._set_integer_parameter(f"SERVO{servo_num}_FUNCTION", function)

    def _set_integer_parameter(self, param_name: str, value: int) -> None:
        from pymavlink import mavutil

        with self._tx_lock:
            self._master.mav.param_set_send(
                self._master.target_system, self._master.target_component,
                param_name.encode("utf-8"), value,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32,
            )
        time.sleep(0.1)

    def _set_integer_parameter_verified(
        self, param_name: str, value: int, retries: int = 3
    ) -> bool:
        for attempt in range(1, retries + 1):
            self._set_integer_parameter(param_name, value)
            actual = self.get_parameter(param_name)
            if actual is not None and int(actual) == value:
                return True
            print(
                f"[RealRobotInterface] {param_name} 설정 재시도 "
                f"{attempt}/{retries}: expected={value}, actual={actual}"
            )
        return False

    def _set_servo_function_verified(
        self, servo_num: int, function: int, retries: int = 3
    ) -> bool:
        """PARAM_SET 유실에 대비해 채널 하나를 설정하고 즉시 read-back한다."""
        for attempt in range(1, retries + 1):
            self._set_servo_function(servo_num, function)
            actual = self._get_servo_function(servo_num)
            if actual == function:
                return True
            print(
                f"[RealRobotInterface] SERVO{servo_num}_FUNCTION 설정 재시도 "
                f"{attempt}/{retries}: expected={function}, actual={actual}"
            )
        return False

    def enable_passthrough(self) -> None:
        """카메라 RC 입력을 격리하고 SERVO1~8을 RCPassThru로 전환한다."""
        originals = [self._get_servo_function(i) for i in range(1, 9)]
        missing = [i for i, value in enumerate(originals, start=1) if value is None]
        if missing:
            raise RuntimeError(f"SERVO FUNCTION 읽기 실패: {missing}; passthrough 변경 중단")

        camera_options = {
            channel: self.get_parameter(f"RC{channel}_OPTION")
            for channel in _CAMERA_RC_OPTION_CHANNELS
        }
        missing_options = [channel for channel, value in camera_options.items() if value is None]
        if missing_options:
            raise RuntimeError(
                f"카메라 RC OPTION 읽기 실패: {missing_options}; passthrough 변경 중단"
            )

        self._original_servo_functions = [int(value) for value in originals]
        self._original_camera_rc_options = {
            channel: int(value) for channel, value in camera_options.items()
        }
        try:
            for channel in _CAMERA_RC_OPTION_CHANNELS:
                if not self._set_integer_parameter_verified(f"RC{channel}_OPTION", 0):
                    raise RuntimeError(f"카메라 입력 격리 실패: RC{channel}_OPTION")
            for i in range(1, 9):
                if not self._set_servo_function_verified(i, _RC_PASSTHRU_FUNCTION):
                    raise RuntimeError(f"RCPassThru 설정 검증 실패: SERVO{i}")
        except Exception:
            # 중간 실패 시 이미 바꾼 파라미터도 즉시 원복한다.
            self.disable_passthrough()
            raise

    def disable_passthrough(self) -> None:
        """백업한 SERVO_FUNCTION 및 RC7/RC8 카메라 옵션을 원복한다."""
        if self._original_servo_functions is None and self._original_camera_rc_options is None:
            return

        failed: list[str] = []
        if self._original_servo_functions is not None:
            for i, original in enumerate(self._original_servo_functions, start=1):
                if not self._set_servo_function_verified(i, original):
                    failed.append(f"SERVO{i}_FUNCTION")
        if self._original_camera_rc_options is not None:
            for channel, original in self._original_camera_rc_options.items():
                name = f"RC{channel}_OPTION"
                if not self._set_integer_parameter_verified(name, original):
                    failed.append(name)

        if failed:
            raise RuntimeError(f"제어 파라미터 원복 검증 실패: {failed}")
        self._original_servo_functions = None
        self._original_camera_rc_options = None

    def set_camera_tilt(
        self,
        command: float,
        min_pitch_deg: float = -45.0,
        max_pitch_deg: float = 45.0,
    ) -> float:
        """정규화 tilt 명령 [-1,1]을 MAVLink mount pitch 목표각으로 보낸다.

        반환값은 실제로 명령한 pitch degree다. RC8을 사용하지 않으므로 8번
        thruster override와 충돌하지 않는다.
        """
        from pymavlink import mavutil

        value = float(command)
        if not -1.0 <= value <= 1.0:
            raise ValueError("camera tilt command는 [-1, 1] 범위여야 함")
        if min_pitch_deg >= max_pitch_deg:
            raise ValueError("camera tilt 최소각은 최대각보다 작아야 함")

        pitch_deg = (
            value * max_pitch_deg if value >= 0.0
            else -value * min_pitch_deg
        )
        with self._tx_lock:
            self._master.mav.command_long_send(
                self._master.target_system,
                self._master.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                0,
                pitch_deg, 0.0, 0.0, 0.0, 0.0, 0.0,
                mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING,
            )
        self._camera_mount_mavlink_active = True
        return pitch_deg

    def restore_camera_rc_control(self) -> None:
        """명시적 tilt 제어를 사용했다면 mount를 기존 RC targeting으로 돌린다."""
        if not self._camera_mount_mavlink_active or self._master is None:
            return
        from pymavlink import mavutil

        with self._tx_lock:
            self._master.mav.command_long_send(
                self._master.target_system,
                self._master.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                mavutil.mavlink.MAV_MOUNT_MODE_RC_TARGETING,
            )
        self._camera_mount_mavlink_active = False

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
            with self._tx_lock:
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

        with self._tx_lock:
            self._master.mav.command_long_send(
                self._master.target_system, self._master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                0, 0, 0, 0, 0, 0, 0,
            )

    # ------------------------------------------------------------------
    def send_pwm(self, pwm: torch.Tensor) -> None:
        """논리 pwm(8,)에 배선 반전을 적용한 뒤 RC override µs로 변환한다."""
        physical_pwm = pwm.clamp(-1.0, 1.0) * self._thruster_reversal_sign.to(pwm.device)
        us = (_PWM_NEUTRAL_US + physical_pwm * _PWM_RANGE_US).round().to(torch.int32)

        chan_us = [_RC_NO_CHANGE] * 18
        for i, ch in enumerate(self._thr_channels):
            chan_us[ch - 1] = int(us[i].item())

        # print(f"send_pwm: {chan_us}")

        with self._tx_lock:
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
        with self._tx_lock:
            self._master.mav.rc_channels_override_send(
                self._master.target_system, self._master.target_component, *chan_us
            )

    def close(self, send_stop: bool = True) -> None:
        """연결 종료. 검증 전 거부된 연결은 send_stop=False로 어떤 override도 보내지 않는다."""
        restore_error = None
        try:
            if self._master is not None:
                if send_stop:
                    self.neutral_stop()
                    time.sleep(0.1)
                    self.release()
                try:
                    self.disable_passthrough()  # 설정했다면 원복 및 read-back 검증
                except Exception as exc:
                    restore_error = exc
                try:
                    self.restore_camera_rc_control()
                except Exception as exc:
                    if restore_error is None:
                        restore_error = exc
        finally:
            self._running = False
            if self._recv_thread is not None:
                self._recv_thread.join(timeout=2.0)
        if restore_error is not None:
            raise restore_error

    def __enter__(self) -> "RealRobotInterface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
