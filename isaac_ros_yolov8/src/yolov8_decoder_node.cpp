// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  target_width_{declare_parameter<int>("target_width", 1280)},
  num_classes_{declare_parameter<int64_t>("num_classes", 3)},       // moved before target_height_
  target_height_{declare_parameter<long int>("target_height", 720)} // after num_classes_
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

    if (aspect_ratio > 1.0) {
      float width = 640.0;
      float height = 360.0;
      float width_scale = static_cast<float>(target_width_) / width;
      float height_scale = static_cast<float>(target_height_) / height;

      float x1_scaled = bboxes[ind].x * width_scale;
      float y1_scaled = bboxes[ind].y * height_scale;
      w = bboxes[ind].width * width_scale;
      h = bboxes[ind].height * height_scale;
      y_center = y1_scaled + (640 - height);
      x_center = x1_scaled;

    } else if (aspect_ratio < 1.0) {
      float width = 640 / aspect_ratio;
      float height = 640;
      float width_scale = static_cast<float>(target_width_) / width;
      float height_scale = static_cast<float>(target_height_) / height;

      float x1_scaled = bboxes[ind].x * width_scale;
      float y1_scaled = bboxes[ind].y * height_scale;
      w = bboxes[ind].width * width_scale;
      h = bboxes[ind].height * height_scale;
      y_center = y1_scaled;
      x_center = x1_scaled - (640 - width);

    } else {
      float width = 640;
      float height = 640;
      float width_scale = static_cast<float>(target_width_) / width;
      float height_scale = static_cast<float>(target_height_) / height;

      float x1_scaled = bboxes[ind].x * width_scale;
      float y1_scaled = bboxes[ind].y * height_scale;
      w = bboxes[ind].width * width_scale;
      h = bboxes[ind].height * height_scale;
      y_center = y1_scaled;
      x_center = x1_scaled;
    }

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
