#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

# Only class 0 (“person”) will be visualized
names = {0: 'person'}


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/stream.mjpg':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            while True:
                frame = self.server.latest_frame
                if frame is None:
                    time.sleep(0.05)
                    continue

                # frame is already BGR
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                data = jpeg.tobytes()

                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
                time.sleep(1.0 / self.server.fps)
        except Exception:
            pass


class Yolov8Visualizer(Node):
    QUEUE_SIZE     = 10
    box_color      = (0xFF, 0xE0, 0x42)  # BGR for #FFE042
    bbox_radius    = 16
    bbox_thickness = 4
    fill_alpha     = 0.2

    def __init__(self):
        super().__init__('yolov8_visualizer')
        self.declare_parameter('fps',       30.0)
        self.declare_parameter('http_port', 8080)
        self.fps       = float(self.get_parameter('fps').get_parameter_value().double_value)
        self.http_port = int(self.get_parameter('http_port').get_parameter_value().integer_value)

        self._bridge = cv_bridge.CvBridge()
        self._pub    = self.create_publisher(Image, 'yolov8_processed_image', self.QUEUE_SIZE)

        det_sub = message_filters.Subscriber(self, Detection2DArray, 'detections_output')
        img_sub = message_filters.Subscriber(self, Image, 'resize/image')
        ts      = message_filters.TimeSynchronizer([det_sub, img_sub], self.QUEUE_SIZE)
        ts.registerCallback(self.detections_callback)

        self._latest_frame = None
        self._start_mjpeg_server()

    def _start_mjpeg_server(self):
        server = ThreadingHTTPServer(('', self.http_port), MJPEGHandler)
        server.latest_frame = None
        server.fps = self.fps
        self._http_server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info(
            f'MJPEG HTTP server started on port {self.http_port} at /stream.mjpg'
        )

    def _draw_rounded_box(self, img, pt1, pt2, color, radius, thickness, alpha):
        x1, y1 = pt1
        x2, y2 = pt2
        overlay = img.copy()

        # Semi‑transparent fill
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius,y1+radius), (x2-radius,y1+radius),
                       (x2-radius,y2-radius), (x1+radius,y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        # Inside stroke
        half = thickness//2
        ix1, iy1 = x1+half, y1+half
        ix2, iy2 = x2-half, y2-half
        ir = max(radius-half, 0)
        # edges
        cv2.line(img, (ix1+ir,iy1), (ix2-ir,iy1), color, thickness)
        cv2.line(img, (ix1+ir,iy2), (ix2-ir,iy2), color, thickness)
        cv2.line(img, (ix1,iy1+ir), (ix1,iy2-ir), color, thickness)
        cv2.line(img, (ix2,iy1+ir), (ix2,iy2-ir), color, thickness)
        # corners
        cv2.ellipse(img, (ix1+ir,iy1+ir), (ir,ir), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (ix2-ir,iy1+ir), (ir,ir), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (ix2-ir,iy2-ir), (ir,ir),   0, 0, 90, color, thickness)
        cv2.ellipse(img, (ix1+ir,iy2-ir), (ir,ir),  90, 0, 90, color, thickness)

    def detections_callback(self, detections_msg, img_msg):
        frame = self._bridge.imgmsg_to_cv2(img_msg)

        # Draw rounded boxes for class 0
        for det in detections_msg.detections:
            cid = int(det.results[0].hypothesis.class_id)
            if cid != 0: continue
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w, h   = det.bbox.size_x, det.bbox.size_y
            pt1 = (int(cx-w/2), int(cy-h/2))
            pt2 = (int(cx+w/2), int(cy+h/2))
            self._draw_rounded_box(
                frame, pt1, pt2,
                color=self.box_color,
                radius=self.bbox_radius,
                thickness=self.bbox_thickness,
                alpha=self.fill_alpha,
            )

        # Republish to ROS
        out = self._bridge.cv2_to_imgmsg(frame, encoding=img_msg.encoding)
        out.header = img_msg.header
        self._pub.publish(out)

        # Convert RGB→BGR if needed for MJPEG
        frame_for_mjpeg = frame
        if img_msg.encoding.lower().startswith('rgb'):
            frame_for_mjpeg = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Update MJPEG frame
        self._latest_frame = frame_for_mjpeg
        if hasattr(self, '_http_server'):
            self._http_server.latest_frame = frame_for_mjpeg

    def destroy_node(self):
        if hasattr(self, '_http_server'):
            self._http_server.shutdown()
        super().destroy_node()


def main():
    rclpy.init()
    node = Yolov8Visualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
