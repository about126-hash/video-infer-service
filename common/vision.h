// #pragma once
// #include<iostream>
// #include<opencv2/opencv.hpp>
// #include"detector.hpp"
// #include"types.h"

// enum class DETECTMODE
// {
//     NONE,PERSON,TEAPATH
// }



// //不管理摄像头资源，只处理cv::Mat
// class Vision
// {
// private:
//     //    
//     Detector<std::vector<Detection>>* person_detector ;
//     Detector<cv::Mat> *path_detector ;
//     DETECTMODE VisionMode = DETECTMODE::None;
// public:
//     Vision(/* args */);

//     void Run(cv::Mat input_image)
//     {
//         switch (VisionMode)
//         {
//         case DETECTMODE::NONE:
//             break;
//         case DETECTMODE::PERSON:
//             break;
//         case DETECTMODE::TEAPATH:
//             break;
        
//         default:
//             break;
//         }
//     }



//     ~Vision()
//     {
//         delete []person_detector;
//         delete []path_detector;
//     };
// };

// Vision::Vision(/* args */)
// {
// }

// Vision::~Vision()
// {
// }
