#pragma once
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "config.hpp"
using namespace std;

extern int worker_count;
extern int port;
extern EngineParam EngineV8;


void readParameters(std::string config_file);
