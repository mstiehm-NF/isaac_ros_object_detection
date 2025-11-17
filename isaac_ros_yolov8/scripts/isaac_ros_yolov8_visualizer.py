#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import threading
import asyncio
import cv2
import cv_bridge
# import message_filters # No longer needed
import rclpy
import websockets
import traceback
import time 
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
    jpeg_quality    = 75   # Lowered JPEG quality

    def __init__(self):
        super().__init__('yolov8_ws_visualizer')
        self.declare_parameter('ws_port', 9001)

        self.ws_port = int(self.get_parameter('ws_port')
                             .get_parameter_value().integer_value)

        self._bridge = cv_bridge.CvBridge()
        self._pub    = self.create_publisher(
            Image, 'yolov8_processed_image', self.QUEUE_SIZE)

        # Store latest messages with automatic cleanup
        self._latest_detection_msg = None
        self._detection_lock = threading.Lock()
        self._frame_counter = 0  # For periodic cleanup

        # Create independent subscribers
        self.detection_subscriber = self.create_subscription(
            Detection2DArray,
            'detections_output',
            self._new_detection_callback,
            self.QUEUE_SIZE)

        self.image_subscriber = self.create_subscription(
            Image,
            'resize/image',  # Assuming this is the input topic for raw/resized images
            self._new_image_callback,
            self.QUEUE_SIZE)

        # websocket clients + event loop
        self._ws_clients = set()
        self._ws_loop = None
        self._ws_thread = None
        self._stop_event = None 

        # start WS server thread
        self._start_ws_server()

    def _start_ws_server(self):
        """Starts the WebSocket server in a separate daemon thread."""
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

    async def _main_ws(self):
        """The main async function to run the WebSocket server."""
        host = '127.0.0.1' # Consider making this configurable or '0.0.0.0'
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
            self._ws_clients.discard(ws)
            self.get_logger().info(f"WebSocket client disconnected: {client_addr}")

    def _draw_rounded_box(self, img, pt1, pt2, color, radius, thickness, alpha):
        """Helper function to draw a rounded rectangle."""
        x1, y1 = pt1; x2, y2 = pt2
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+radius, y1),   (x2-radius, y2),   color, -1)
        cv2.rectangle(overlay, (x1,      y1+radius), (x2,      y2-radius), color, -1)
        for cx, cy in [
            (x1+radius, y1+radius), (x2-radius, y1+radius),
            (x2-radius, y2-radius), (x1+radius, y2-radius),
        ]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        half = thickness // 2
        ix1, iy1 = x1 + half, y1 + half
        ix2, iy2 = x2 - half, y2 - half
        ir = max(radius - half, 0)
        cv2.line(img, (ix1+ir, iy1), (ix2-ir, iy1), color, thickness)
        cv2.line(img, (ix1+ir, iy2), (ix2-ir, iy2), color, thickness)
        cv2.line(img, (ix1, iy1+ir), (ix1, iy2-ir), color, thickness)
        cv2.line(img, (ix2, iy1+ir), (ix2, iy2-ir), color, thickness)
        cv2.ellipse(img, (ix1+ir, iy1+ir), (ir,ir), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (ix2-ir, iy1+ir), (ir,ir), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (ix2-ir, iy2-ir), (ir,ir),   0, 0, 90, color, thickness)
        cv2.ellipse(img, (ix1+ir, iy2-ir), (ir,ir),  90, 0, 90, color, thickness)

    def _new_detection_callback(self, det_msg: Detection2DArray):
        """Stores the latest detection message."""
        with self._detection_lock:
            self._latest_detection_msg = det_msg

    def _new_image_callback(self, img_msg: Image):
        """Callback for new images. Processes image with latest detections."""
        has_ws_clients = bool(self._ws_clients)
        
        # Periodic cleanup to prevent memory accumulation
        self._frame_counter += 1
        if self._frame_counter % 100 == 0:  # Every 100 frames
            import gc
            gc.collect()
        
        # Get the latest detection message in a thread-safe way
        current_detection_msg = None
        with self._detection_lock:
            if self._latest_detection_msg:
                current_detection_msg = self._latest_detection_msg

        try:
            frame_initial = self._bridge.imgmsg_to_cv2(img_msg)
            if 'rgb' in img_msg.encoding.lower():
                frame = cv2.cvtColor(frame_initial, cv2.COLOR_RGB2BGR)
            elif 'bgr' in img_msg.encoding.lower():
                frame = frame_initial
            elif 'mono' in img_msg.encoding.lower():
                 frame = cv2.cvtColor(frame_initial, cv2.COLOR_GRAY2BGR)
            else:
                 self.get_logger().warn(f"Unsupported input encoding {img_msg.encoding}, attempting BGR conversion.", throttle_duration_sec=5)
                 try:
                      frame = cv2.cvtColor(frame_initial, cv2.COLOR_YUV2BGR_YUYV) 
                 except cv2.error:
                      self.get_logger().error(f"Could not convert encoding {img_msg.encoding} to BGR. Skipping frame.")
                      return
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError converting image: {e}')
            return
        except Exception as e:
            self.get_logger().error(f'Error during color conversion: {e}')
            return

        # --- Drawing happens on the BGR frame ---
        if current_detection_msg: # Only draw if we have detections
            for det in current_detection_msg.detections:
                cid = int(det.results[0].hypothesis.class_id)
                if cid != 0 or names.get(cid) != 'person': continue # Ensure it's a person
                
                w, h = det.bbox.size_x, det.bbox.size_y
                if w < self.min_box_size_px or h < self.min_box_size_px: continue
                
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                pt1 = (int(cx - w/2), int(cy - h/2))
                pt2 = (int(cx + w/2), int(cy + h/2))
                self._draw_rounded_box(frame, pt1, pt2, self.box_color, self.bbox_radius, self.bbox_thickness, self.fill_alpha)

        # --- Flip the image vertically ---
        frame = cv2.flip(frame, 0)

        # Republish to ROS (Optional, uncomment if needed)
        try:
            output_encoding = 'bgr8'
            out_msg = self._bridge.cv2_to_imgmsg(frame, encoding=output_encoding)
            out_msg.header = img_msg.header # Use current image's header
            self._pub.publish(out_msg)
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError on republish: {e}')

        # --- Send to WebSocket clients ---
        if has_ws_clients and self._ws_loop and self._ws_loop.is_running():
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, jpg = cv2.imencode('.jpg', frame, params)
            if not ret:
                self.get_logger().error("cv2.imencode failed")
                return
            
            blob = jpg.tobytes()
            current_clients = list(self._ws_clients) # Copy to avoid modification during iteration
            
            # Send to clients and handle failed connections
            for ws in current_clients:
                if self._ws_loop.is_running():
                    try:
                        task = asyncio.run_coroutine_threadsafe(ws.send(blob), self._ws_loop)
                        # Don't wait for completion to avoid blocking, but add timeout
                        task.result(timeout=0.01)  # Very short timeout to detect immediate failures
                    except (asyncio.TimeoutError, asyncio.InvalidStateError):
                        # Task still running or failed immediately, continue to next client
                        pass
                    except Exception as e:
                        # Client likely disconnected, remove from list
                        self.get_logger().debug(f"Removing failed WebSocket client: {e}")
                        self._ws_clients.discard(ws)
                else:
                    self.get_logger().warn("WS loop stopped during send scheduling.", throttle_duration_sec=5)
                    break
            
            # Explicit cleanup to help garbage collection
            del blob, jpg
        # ---

    def destroy_node(self):
        """Cleanly shuts down the node and WebSocket server."""
        self.get_logger().info("Destroying node...")
        ws_thread = self._ws_thread
        ws_loop = self._ws_loop

        if ws_loop and ws_loop.is_running() and self._stop_event and not self._stop_event.done():
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
        if node:
            node.get_logger().info("KeyboardInterrupt, shutting down.")
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
        print("rclpy shutdown complete.")


if __name__ == '__main__':
    main()