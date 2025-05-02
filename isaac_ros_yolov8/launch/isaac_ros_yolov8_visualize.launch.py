import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('isaac_ros_yolov8')

    # --- Declare launch arguments ---
    ws_port_arg = DeclareLaunchArgument(
        'ws_port',
        default_value='9001',
        description='WebSocket port for the visualizer'
    )
    ns_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top‑level namespace for all nodes'
    )

    # --- LaunchConfigurations ---
    ws_port  = LaunchConfiguration('ws_port')
    namespace   = LaunchConfiguration('namespace')

    return LaunchDescription([
        ws_port_arg,
        ns_arg,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'yolov8_tensor_rt.launch.py')
            )
        ),

        # visualizer under the chosen namespace
        Node(
            package='isaac_ros_yolov8',
            executable='isaac_ros_yolov8_visualizer.py',
            name='yolov8_visualizer',
            namespace=namespace,
            parameters=[
                {'ws_port':  ws_port},
            ]
        ),

        # image_view under the same namespace
        # Node(
        #     package='rqt_image_view',
        #     executable='rqt_image_view',
        #     name='image_view',
        #     namespace=namespace,
        #     arguments=['yolov8_processed_image']
        # )
    ])
