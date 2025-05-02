#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import threading
import asyncio
import cv2
import cv_bridge
import message_filters
import rclpy
import websockets
import traceback
import time # Import time for potential future throttling
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

# Only class 0 (“person”) will be visualized
names = {0: 'person'}

class Yolov8WsVisualizer(Node):
    QUEUE_SIZE      = 10
    box_color       = (0x42, 0xE0, 0xFF)  # BGR for #FFE042
    bbox_radius     = 16
    bbox_thickness  = 4
    fill_alpha      = 0.2
    min_box_size_px = 40   # filter out small boxes
    jpeg_quality    = 75   # Lowered JPEG quality (Optimization #2)

    def __init__(self):
        super().__init__('yolov8_ws_visualizer')
        self.declare_parameter('ws_port', 9001)

        self.ws_port = int(self.get_parameter('ws_port')
                             .get_parameter_value().integer_value)

        self._bridge = cv_bridge.CvBridge()
        self._pub    = self.create_publisher(
            Image, 'yolov8_processed_image', self.QUEUE_SIZE)

        # sync detections + raw image
        det_sub = message_filters.Subscriber(
            self, Detection2DArray, 'detections_output')
        img_sub = message_filters.Subscriber(

            self, Image, 'resize/image')
        ts = message_filters.TimeSynchronizer(

            [det_sub, img_sub], self.QUEUE_SIZE)
        ts.registerCallback(self.detections_callback)

        # websocket clients + event loop
        self._ws_clients = set()
        self._ws_loop = None
        self._ws_thread = None
        self._stop_event = None # Initialize stop event

        # start WS server thread
        self._start_ws_server()

    def _start_ws_server(self):
        """Starts the WebSocket server in a separate daemon thread."""
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

    async def _main_ws(self):
        """The main async function to run the WebSocket server."""
        host = '127.0.0.1'
        self._stop_event = asyncio.Future(loop=self._ws_loop)

        async with websockets.serve(self._ws_handler, host, self.ws_port) as server:
            actual_addr = server.sockets[0].getsockname() if server.sockets else 'unknown'
            self.get_logger().info(f'WebSocket server started successfully, listening on {actual_addr}')
            await self._stop_event
            self.get_logger().info('WebSocket server stopping...')

    def _run_ws_server(self):
        """Target function for the WebSocket server thread."""
        loop = None
        try:
            self.get_logger().info("WebSocket server thread started.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ws_loop = loop
            self.get_logger().info("Asyncio event loop created and set for this thread.")
            loop.run_until_complete(self._main_ws())
        except OSError as e:
            self.get_logger().error(f"WebSocket server OS Error: {e}. Is port {self.ws_port} already in use?")
        except asyncio.CancelledError:
            self.get_logger().info("WebSocket server task cancelled.")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"WebSocket server thread encountered an unhandled exception: {e}\n{tb_str}")
        finally:
            self.get_logger().info("WebSocket server event loop finishing.")
            if loop and not loop.is_closed():
                 self.get_logger().info("Closing WebSocket event loop.")
                 loop.close()
            self.get_logger().info("WebSocket server thread exiting.")

    async def _ws_handler(self, ws, path=None):
        """Handles individual client connections."""
        client_addr = ws.remote_address
        self.get_logger().info(f"WebSocket client connected: {client_addr} (Path: {path})")
        self._ws_clients.add(ws)
        try:
            await ws.wait_closed()
        except websockets.exceptions.ConnectionClosedOK:
            self.get_logger().info(f"WebSocket client connection closed normally: {client_addr}")
        except websockets.exceptions.ConnectionClosedError as e:
            self.get_logger().warning(f"WebSocket client connection closed with error: {client_addr} - {e}")
        finally:
            # Use discard() instead of remove() to avoid KeyError if already removed
            self._ws_clients.discard(ws)
            self.get_logger().info(f"WebSocket client disconnected: {client_addr}")

    def _draw_rounded_box(self, img, pt1, pt2, color, radius, thickness, alpha):
        """Helper function to draw a rounded rectangle."""
        # --- Kept original drawing function as requested ---
        x1, y1 = pt1; x2, y2 = pt2
        overlay = img.copy()

        # semi-transparent fill
        cv2.rectangle(overlay, (x1+radius, y1),   (x2-radius, y2),   color, -1)
        cv2.rectangle(overlay, (x1,      y1+radius), (x2,      y2-radius), color, -1)
        for cx, cy in [
            (x1+radius, y1+radius),
            (x2-radius, y1+radius),
            (x2-radius, y2-radius),
            (x1+radius, y2-radius),
        ]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        # inside stroke
        half = thickness // 2
        ix1, iy1 = x1 + half, y1 + half
        ix2, iy2 = x2 - half, y2 - half
        ir = max(radius - half, 0)

        # edges
        cv2.line(img, (ix1+ir, iy1), (ix2-ir, iy1), color, thickness)
        cv2.line(img, (ix1+ir, iy2), (ix2-ir, iy2), color, thickness)
        cv2.line(img, (ix1, iy1+ir), (ix1, iy2-ir), color, thickness)
        cv2.line(img, (ix2, iy1+ir), (ix2, iy2-ir), color, thickness)
        # corners
        cv2.ellipse(img, (ix1+ir, iy1+ir), (ir,ir), 180,   0,  90, color, thickness)
        cv2.ellipse(img, (ix2-ir, iy1+ir), (ir,ir), 270,   0,  90, color, thickness)
        cv2.ellipse(img, (ix2-ir, iy2-ir), (ir,ir),   0,   0,  90, color, thickness)
        cv2.ellipse(img, (ix1+ir, iy2-ir), (ir,ir),  90,   0,  90, color, thickness)
        # --- End of original drawing function ---

    def detections_callback(self, det_msg: Detection2DArray, img_msg: Image):
        """Callback for synchronized detection and image messages."""
        has_ws_clients = bool(self._ws_clients)

        try:
            # Convert ROS msg to OpenCV frame. cv_bridge might convert based on encoding.
            # Let's assume it gives us a frame matching img_msg.encoding for now.
            frame_initial = self._bridge.imgmsg_to_cv2(img_msg) # Removed desired_encoding

            # --- Explicitly ensure frame is BGR for drawing ---
            if 'rgb' in img_msg.encoding.lower():
                # If the input encoding was RGB, convert the frame to BGR
                frame = cv2.cvtColor(frame_initial, cv2.COLOR_RGB2BGR)
            elif 'bgr' in img_msg.encoding.lower():
                # If the input encoding was already BGR, use it directly
                frame = frame_initial
            elif 'mono' in img_msg.encoding.lower():
                 # If mono, convert to BGR for color drawing (boxes will be color)
                 frame = cv2.cvtColor(frame_initial, cv2.COLOR_GRAY2BGR)
            else:
                 # Fallback or handle other encodings if necessary
                 self.get_logger().warn(f"Unsupported input encoding {img_msg.encoding}, attempting BGR conversion.", throttle_duration_sec=5)
                 # Try a generic conversion, might fail or be incorrect
                 try:
                      frame = cv2.cvtColor(frame_initial, cv2.COLOR_YUV2BGR_YUYV) # Example for YUYV
                 except cv2.error:
                      self.get_logger().error(f"Could not convert encoding {img_msg.encoding} to BGR. Skipping frame.")
                      return

        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError converting image: {e}')
            return
        except Exception as e: # Catch potential cvtColor errors too
            self.get_logger().error(f'Error during color conversion: {e}')
            return


        # --- Drawing happens on the BGR frame ---
        for det in det_msg.detections:
            cid = int(det.results[0].hypothesis.class_id)
            if cid != 0: continue
            w, h = det.bbox.size_x, det.bbox.size_y
            if w < self.min_box_size_px or h < self.min_box_size_px: continue
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            pt1 = (int(cx - w/2), int(cy - h/2))
            pt2 = (int(cx + w/2), int(cy + h/2))
            self._draw_rounded_box(frame, pt1, pt2, self.box_color, self.bbox_radius, self.bbox_thickness, self.fill_alpha)

        # --- Flip the image vertically ---
        frame = cv2.flip(frame, 0)

        # Republish to ROS - Frame is now guaranteed to be BGR
        try:
            # Explicitly set encoding to bgr8 as we ensured the frame is BGR
            output_encoding = 'bgr8'
            out_msg = self._bridge.cv2_to_imgmsg(frame, encoding=output_encoding)
            out_msg.header = img_msg.header
            self._pub.publish(out_msg)
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError on republish: {e}')

        # --- Optimization: Only encode and send if clients are connected ---
        if has_ws_clients and self._ws_loop and self._ws_loop.is_running():
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            # Encode the BGR frame
            ret, jpg = cv2.imencode('.jpg', frame, params)
            if not ret:
                self.get_logger().error("cv2.imencode failed")
                return
            blob = jpg.tobytes()

            current_clients = list(self._ws_clients)
            tasks = [ws.send(blob) for ws in current_clients]
            for task in tasks:
                 if self._ws_loop.is_running():
                      future = asyncio.run_coroutine_threadsafe(task, self._ws_loop)
                 else:
                      self.get_logger().warn("WS loop stopped during send scheduling.", throttle_duration_sec=5)
                      break
        # ---

    def destroy_node(self):
        """Cleanly shuts down the node and WebSocket server."""
        self.get_logger().info("Destroying node...")
        ws_thread = self._ws_thread
        ws_loop = self._ws_loop

        if ws_loop and ws_loop.is_running() and self._stop_event:
             self.get_logger().info("Requesting WebSocket server to stop...")
             ws_loop.call_soon_threadsafe(self._stop_event.set_result, None)

        if ws_thread and ws_thread.is_alive():
             self.get_logger().info("Waiting for WebSocket thread to join...")
             ws_thread.join(timeout=2.0)
             if ws_thread.is_alive():
                  self.get_logger().warn("WebSocket thread did not exit cleanly after timeout.")

        try:
            super().destroy_node()
            self.get_logger().info("Base node destroyed.")
        except Exception as e:
             self.get_logger().error(f"Error during super().destroy_node(): {e}\n{traceback.format_exc()}")

        self.get_logger().info("Node destruction complete.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Yolov8WsVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception in main spin: {e}\n{traceback.format_exc()}")
        else:
            print(f"Unhandled exception before node init: {e}\n{traceback.format_exc()}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()