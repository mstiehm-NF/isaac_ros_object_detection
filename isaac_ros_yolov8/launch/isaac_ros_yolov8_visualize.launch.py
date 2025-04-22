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
    http_port_arg = DeclareLaunchArgument(
        'http_port',
        default_value='8080',
        description='HTTP port for the MJPEG stream'
    )
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30.0',
        description='Frame rate for the output stream'
    )
    ns_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top‑level namespace for all nodes'
    )

    # --- LaunchConfigurations ---
    http_port  = LaunchConfiguration('http_port')
    fps         = LaunchConfiguration('fps')
    namespace   = LaunchConfiguration('namespace')

    return LaunchDescription([
        ns_arg,
        http_port_arg,
        fps_arg,
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
                {'http_port':  http_port},
                {'fps':      fps}
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
