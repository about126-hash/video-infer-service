# models 目录

模型文件不进仓库（见根目录 `.gitignore`）。

| 文件 | 用途 |
|---|---|
| `yolov8n_fp32.engine` | 默认检测引擎（YOLOv8n，FP32） |

引擎路径在 `common/config.h` 的 `EngineV8` 中配置，换机器时修改该路径即可。
