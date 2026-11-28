#include "postprocess.h"
#include "utils.h"
#include "config.hpp"


cv::Rect get_rect(cv::Mat& img, float bbox[4] ,int kInputW,int kInputH) {
  float l, r, t, b;
  float r_w = kInputW / (img.cols * 1.0);
  float r_h = kInputH / (img.rows * 1.0);
  if (r_h > r_w) {
    l = bbox[0] - bbox[2] / 2.f;
    r = bbox[0] + bbox[2] / 2.f;
    t = bbox[1] - bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
    b = bbox[1] + bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
    l = l / r_w;
    r = r / r_w;
    t = t / r_w;
    b = b / r_w;
  } else {
    l = bbox[0] - bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
    r = bbox[0] + bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
    t = bbox[1] - bbox[3] / 2.f;
    b = bbox[1] + bbox[3] / 2.f;
    l = l / r_h;
    r = r / r_h;
    t = t / r_h;
    b = b / r_h;
  }
  return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

static float iou(float lbox[4], float rbox[4]) {
  float interBox[] = {
    (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
    (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
    (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
    (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection& a, const Detection& b) {
  return a.conf > b.conf;
}

//YOLOv5 NMS
//输出：最终保留的所有检测框
//插件推理输出缓冲区 [有效框数, Detection1, Detection2...]
//置信度阈值，低于这个直接扔掉
//NMS交并比阈值，重叠超过这个就删掉重复框
//Detection bbox信息，x,y,w,h,conf,class_id,mask[32]
void v5_nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
  int det_size = sizeof(Detection) / sizeof(float);
  std::map<float, std::vector<Detection>> m;
  int total = (int)output[0];
  int pass = 0;
  // 遍历所有框，过滤低置信度框，然后在按照class_id进行分组存储
  for (int i = 0; i < total && i < 1000; i++) {
    float conf = output[1 + det_size * i + 4];
    if (conf <= conf_thresh) continue;
    pass++;
    Detection det;
    memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
    if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
    m[det.class_id].push_back(det);
  }
  
  //遍历m.second(即为一个vector<Detection>),按照conf进行排序
  // 遍历所有类别（map的key是类别id，value是该类所有框）
  for (auto it = m.begin(); it != m.end(); it++) {
      auto& dets = it->second; // dets = 当前这一类的全部检测框 vector<Detection>
      std::sort(dets.begin(), dets.end(), cmp); // 按置信度从高到低排序

      // 逐个取置信度最高的框作为基准框
      for (size_t m = 0; m < dets.size(); ++m) {
          auto& item = dets[m];
          res.push_back(item); // 基准框直接保留，存入最终结果

          // 基准框 和 它后面所有低分框 逐一对比重叠程度
          for (size_t n = m + 1; n < dets.size(); ++n) {
              // 计算两个框的IOU交并比
              if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                  // 重叠超过阈值，判定为重复框，直接删除
                  dets.erase(dets.begin() + n);
                  --n; // 删除后数组元素前移，下标-1防止跳过下一个框
              }
          }
      }
  }
}


// ONNX 输出的 NMS（处理 [1, num_dets, 5+nc] 格式：cx, cy, w, h, obj_conf, class_confs...）
// 对于不同格式的 engine head, 需要根据实际情况调整
void nms_onnx(std::vector<Detection>& res, float* output, int num_dets, int nc,
              float conf_thresh, float nms_thresh) {
    int det_dim = 5 + nc;
    std::map<int, std::vector<Detection>> m;  // 按类别分组

    // 解析engine head 输出，过滤低置信度
    for (int i = 0; i < num_dets; i++) {
        float* row = output + i * det_dim;
        float obj_conf = row[4];

        // 找到最大类置信度及其索引
        float max_cls_conf = 0.0f;
        int best_class = 0;
        for (int c = 0; c < nc; c++) {
            if (row[5 + c] > max_cls_conf) {
                max_cls_conf = row[5 + c];
                best_class = c;
            }
        }

        //置信度=框置信度*类别置信度
        float conf = obj_conf * max_cls_conf;
        if (conf <= conf_thresh) continue;

        Detection det;
        det.bbox[0]  = row[0];       // cx
        det.bbox[1]  = row[1];       // cy
        det.bbox[2]  = row[2];       // w
        det.bbox[3]  = row[3];       // h
        det.conf     = conf;
        det.class_id = best_class;
        memset(det.mask, 0, sizeof(det.mask));

        m[best_class].push_back(det);
    }

    // 每类独立 NMS
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(),
                  [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

        for (size_t i = 0; i < dets.size(); i++) {
            res.push_back(dets[i]);
            float l_i = dets[i].bbox[0] - dets[i].bbox[2] / 2.0f;
            float r_i = dets[i].bbox[0] + dets[i].bbox[2] / 2.0f;
            float t_i = dets[i].bbox[1] - dets[i].bbox[3] / 2.0f;
            float b_i = dets[i].bbox[1] + dets[i].bbox[3] / 2.0f;

            for (size_t j = i + 1; j < dets.size(); j++) {
                float l_j = dets[j].bbox[0] - dets[j].bbox[2] / 2.0f;
                float r_j = dets[j].bbox[0] + dets[j].bbox[2] / 2.0f;
                float t_j = dets[j].bbox[1] - dets[j].bbox[3] / 2.0f;
                float b_j = dets[j].bbox[1] + dets[j].bbox[3] / 2.0f;

                float inter_l = std::max(l_i, l_j);
                float inter_r = std::min(r_i, r_j);
                float inter_t = std::max(t_i, t_j);
                float inter_b = std::min(b_i, b_j);
                if (inter_l >= inter_r || inter_t >= inter_b) continue;

                float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
                float area_i = (r_i - l_i) * (b_i - t_i);
                float area_j = (r_j - l_j) * (b_j - t_j);
                float iou = inter_area / (area_i + area_j - inter_area);

                if (iou > nms_thresh) {
                    dets.erase(dets.begin() + j);
                    j--;
                }
            }
        }
    }
}

// output: engine 原始输出 [1, 4+kNumClass, kMaxNumOutputBbox]，channel-major
//         元素 (c, n) 的地址 = c * num_anchors + n
void v8_nms(std::vector<Detection>& res, float* output, float ClsNum , float MaxNumOutputBbox ,  float conf_thresh, float nms_thresh) {
    const int num_classes = ClsNum;          // 80
    const int num_anchors = MaxNumOutputBbox;  // 8400 = 80*80 + 40*40 + 20*20
    std::map<float, std::vector<Detection>> m;

    // 解码：每个候选点取 80 个类里分数最高的那个（分数在导出时已 sigmoid 过）
    for (int n = 0; n < num_anchors; n++) {
        int   best_id = 0;
        float best_p  = 0.f;
        for (int c = 0; c < num_classes; c++) {
            float p = output[(4 + c) * num_anchors + n];
            if (p > best_p) { best_p = p; best_id = c; }
        }
        if (isnan(best_p) || best_p <= conf_thresh) continue;

        Detection det{};                             // 零初始化，mask 保持 0
        det.bbox[0] = output[0 * num_anchors + n];   // cx，已是 640 像素单位
        det.bbox[1] = output[1 * num_anchors + n];   // cy
        det.bbox[2] = output[2 * num_anchors + n];   // w
        det.bbox[3] = output[3 * num_anchors + n];   // h
        det.conf     = best_p;
        det.class_id = (float)best_id;

        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch, int kInputW, int kInputH) {
  for (size_t i = 0; i < img_batch.size(); i++) {
    auto& res = res_batch[i];
    cv::Mat img = img_batch[i];
    for (size_t j = 0; j < res.size(); j++) {
      cv::Rect r = get_rect(img, res[j].bbox,kInputW,kInputH);
      cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
  }
}

static cv::Rect get_downscale_rect(float bbox[4], float scale,int kInputW,int kInputH) {
  float left = bbox[0] - bbox[2] / 2;
  float top = bbox[1] - bbox[3] / 2;
  float right = bbox[0] + bbox[2] / 2;
  float bottom = bbox[1] + bbox[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets, int kInputW, int kInputH) {
  std::vector<cv::Mat> masks;
  for (size_t i = 0; i < dets.size(); i++) {
    cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
    auto r = get_downscale_rect(dets[i].bbox, 4,kInputW,kInputH);
    for (int x = r.x; x < r.x + r.width; x++) {
      for (int y = r.y; y < r.y + r.height; y++) {
        float e = 0.0f;
        for (int j = 0; j < 32; j++) {
          e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
        }
        e = 1.0f / (1.0f + expf(-e));
        mask_mat.at<float>(y, x) = e;
      }
    }
    cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
    masks.push_back(mask_mat);
  }
  return masks;
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img,int kInputW,int kInputH) {
  int x, y, w, h;
  float r_w = kInputW / (img.cols * 1.0);
  float r_h = kInputH / (img.rows * 1.0);
  if (r_h > r_w) {
    w = kInputW;
    h = r_w * img.rows;
    x = 0;
    y = (kInputH - h) / 2;
  } else {
    w = r_h * img.cols;
    h = kInputH;
    x = (kInputW - w) / 2;
    y = 0;
  }
  cv::Rect r(x, y, w, h);
  cv::Mat res;
  cv::resize(mask(r), res, img.size());
  return res;
}

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks, std::unordered_map<int, std::string>& labels_map, int kInputW, int kInputH) {
  static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A,
                                         0x92CC17, 0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF,
                                         0x344593, 0x6473FF, 0x0018EC, 0x8438FF, 0x520085, 0xCB38FF,
                                         0xFF95C8, 0xFF37C7};
  for (size_t i = 0; i < dets.size(); i++) {
    cv::Mat img_mask = scale_mask(masks[i], img, kInputW,kInputH);
    auto color = colors[(int)dets[i].class_id % colors.size()];
    auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);
    
    cv::Rect r = get_rect(img, dets[i].bbox,kInputW,kInputH);
    for (int x = r.x; x < r.x + r.width; x++) {
      for (int y = r.y; y < r.y + r.height; y++) {
        float val = img_mask.at<float>(y, x);
        if (val <= 0.5) continue;
        img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + bgr[0] / 2;
        img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + bgr[1] / 2;
        img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + bgr[2] / 2;
      }
    }

    cv::rectangle(img, r, bgr, 2);
    
    // Get the size of the text
    cv::Size textSize = cv::getTextSize(labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf), cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
    // Set the top left corner of the rectangle
    cv::Point topLeft(r.x, r.y - textSize.height);

    // Set the bottom right corner of the rectangle
    cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

    // Set the thickness of the rectangle lines
    int lineThickness = 2;

    // Draw the rectangle on the image
    cv::rectangle(img, topLeft, bottomRight, bgr, -1);

    cv::putText(img, labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf), cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);

  }
}

