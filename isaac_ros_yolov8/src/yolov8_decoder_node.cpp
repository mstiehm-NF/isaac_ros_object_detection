// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV8DecoderNode::InputCallback, this,
      std::placeholders::_1))},
  
  // Publisher for output Detection2DArray messages
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  target_width_{declare_parameter<int>("target_width", 640)},
  target_height_{declare_parameter<int>("target_height", 480)},
  num_classes_{declare_parameter<int>("num_classes", 3)}
  
{}

YoloV8DecoderNode::~YoloV8DecoderNode() = default;

void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView& msg)
{
  long int num_classes = num_classes_;

  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  std::vector<float> results_vector(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  int out_dim = 8400;
  float* results_data = results_vector.data();

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  bboxes.reserve(out_dim);
  scores.reserve(out_dim);
  classes.reserve(out_dim);

  for (int i = 0; i < out_dim; i++) {
    float x = results_data[i];
    float y = results_data[out_dim + i];
    float w = results_data[2 * out_dim + i];
    float h = results_data[3 * out_dim + i];

    float max_conf = 0.0f;
    int max_index = 0;
    for (int j = 0; j < num_classes; j++) {
      float conf = results_data[(4 + j) * out_dim + i];
      if (conf > max_conf) {
        max_conf = conf;
        max_index = j;
      }
    }

    bboxes.emplace_back(x, y, w, h);
    scores.push_back(max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  final_detections_arr.header.frame_id = msg.GetFrameId();

  for (const auto& ind : indices) {
    vision_msgs::msg::Detection2D detection;
    vision_msgs::msg::BoundingBox2D bbox;

    float aspect_ratio = static_cast<float>(target_width_) / target_height_;

    float x_center, y_center, w, h;

    // if (aspect_ratio > 1.0) {
    //   float width = 640.0;  // Resized width
    //   float height = 640.0 / aspect_ratio;  // Resized height (maintain aspect ratio)
    //   float scale = static_cast<float>(target_width_) / width;  // Scaling factor
    //   float vertical_padding = (640.0 - height) / 2.0;  // Padding added to height

    //   // Scale bounding box coordinates
    //   float x1_scaled = (bboxes[ind].x - vertical_padding) * scale;
    //   float y1_scaled = bboxes[ind].y * scale;

    //   w = bboxes[ind].width * scale;
    //   h = bboxes[ind].height * scale;

    //   x_center = x1_scaled;
    //   y_center = y1_scaled;

    // } else if (aspect_ratio < 1.0) {
    //   float width = 640.0 * aspect_ratio;  // Resized width (maintain aspect ratio)
    //   float height = 640.0;  // Resized height
    //   float scale = static_cast<float>(target_height_) / height;  // Scaling factor
    //   float horizontal_padding = (640.0 - width) / 2.0;  // Padding added to width

    //   // Scale bounding box coordinates
    //   float x1_scaled = bboxes[ind].x * scale;
    //   float y1_scaled = (bboxes[ind].y - horizontal_padding) * scale;

    //   w = bboxes[ind].width * scale;
    //   h = bboxes[ind].height * scale;

    //   x_center = x1_scaled;
    //   y_center = y1_scaled;

    // } else {
    //   float width = 640.0;
    //   float height = 640.0;
    //   float scale = static_cast<float>(target_width_) / width;  // Scaling factor

    //   // Scale bounding box coordinates
    //   float x1_scaled = bboxes[ind].x * scale;
    //   float y1_scaled = bboxes[ind].y * scale;

    //   w = bboxes[ind].width * scale;
    //   h = bboxes[ind].height * scale;

    //   x_center = x1_scaled;
    //   y_center = y1_scaled;
    // }

    float scale;
    float pad_x, pad_y;

    if (aspect_ratio > 1.0) {  
        scale = 640.0 / target_width_;  
        float new_height = target_height_ * scale; 
        pad_y = (640.0 - new_height) / 2.0;  
        pad_x = 0.0; 
    } else if (aspect_ratio < 1.0) {  
        scale = 640.0 / target_height_;  
        float new_width = target_width_ * scale;  
        pad_x = (640.0 - new_width) / 2.0;  
        pad_y = 0.0;  
    } else {  
        scale = 640.0 / target_width_;
        pad_x = 0.0;
        pad_y = 0.0;
    }

   
    x_center = (bboxes[ind].x - pad_x) / scale;
    y_center = (bboxes[ind].y - pad_y) / scale;
    w = bboxes[ind].width / scale;
    h = bboxes[ind].height / scale;

    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes[ind]);
    hyp.hypothesis.score = scores[ind];
    detection.results.push_back(hyp);

    detection.header.stamp = final_detections_arr.header.stamp;
    detection.header.frame_id = final_detections_arr.header.frame_id;
    final_detections_arr.detections.push_back(detection);
  }

  pub_->publish(final_detections_arr);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)
