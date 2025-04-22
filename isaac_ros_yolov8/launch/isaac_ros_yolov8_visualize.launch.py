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
    output_arg = DeclareLaunchArgument(
        'output_mode',
        default_value='v4l2',
        description='Stream output mode: "v4l2" or "rtsp"'
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='/dev/video9',
        description='v4l2loopback device (only used if output_mode=v4l2)'
    )
    rtsp_url_arg = DeclareLaunchArgument(
        'rtsp_url',
        default_value='rtsp://0.0.0.0:8554/live.sdp',
        description='RTSP listen URL (only used if output_mode=rtsp)'
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
    output_mode = LaunchConfiguration('output_mode')
    device      = LaunchConfiguration('device')
    rtsp_url    = LaunchConfiguration('rtsp_url')
    fps         = LaunchConfiguration('fps')
    namespace   = LaunchConfiguration('namespace')

    return LaunchDescription([
        ns_arg,
        output_arg,
        device_arg,
        rtsp_url_arg,
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
                {'output':   output_mode},
                {'device':   device},
                {'rtsp_url': rtsp_url},
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
