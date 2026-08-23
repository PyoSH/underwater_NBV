.PHONY: shell up down ros-build ros-test ros-packages check

shell:
	./docker/brov-shell

up:
	docker compose up -d

down:
	docker compose down

ros-build: up
	docker compose exec brov bash /workspace/deploy/docker/build_ros_workspace.sh

ros-test: ros-build
	docker compose exec brov bash -lc 'source /workspace/deploy/docker/ros_env.sh && cd "$${BROV_ROS_WS}" && colcon test --event-handlers console_direct+ && colcon test-result --verbose && cd /workspace/deploy && python3 -m pytest -q test_guidance_depth_hold.py test_observation_quaternion.py test_model_based_controller.py'

ros-packages: ros-build
	docker compose exec brov bash -lc 'source /workspace/deploy/docker/ros_env.sh && ros2 pkg executables | grep "^brov_"'

check: ros-build
	docker compose exec brov bash -lc 'source /workspace/deploy/docker/ros_env.sh && cd /workspace/deploy && python3 docker/check_environment.py'
