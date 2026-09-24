# seed-api-test

最小化的火山方舟 `API_KEY` 连通性与 Seedance 2.0 风格迁移测试目录。

## 功能

1. 测试 `GET /models` 是否可用
2. 使用 `doubao-seed-2-0-mini-260428` 做一次简单文本问答
3. 使用 `inputs/ref-01.jpg` 对 Seedream 5.0 Pro 做图编测试
4. 对 Seed-2.1 Pro / Lite 与 Seed-Evolving 做轻量文本 + 图文理解冒烟测试
5. **Seedance 2.0 风格迁移流水线**（首帧风格编辑 → 编辑视频，固定 **720p / 4s**）

## 运行

```bash
conda activate base
cd D:\Knowin\tmp\seed-api-test
pip install -r requirements.txt

# 基础连通性
python check_api_key.py
python check_seedream_edit.py
python check_seed_vlm.py

# Seedance 2.0 风格迁移（推荐一键）
python run_seedance2_pipeline.py

# 或分步
python extract_first_frame.py
python run_seedream_style_first_frame.py
python run_seedance2_style_transfer.py
```

## Seedance 2.0 风格迁移说明

- 输入视频：`inputs/seedance2_test/26-08-19_Origin_4s.mp4`
- 首图编辑：Seedream 5.0 Pro，风格「科技夜晚」
- 视频生成：Seedance 2.0 **编辑视频**模式（`reference_image` + `reference_video` + 提示词）
- **硬性输出**：`resolution=720p`，`duration=4`，`ratio=adaptive`，`generate_audio=false`，`watermark=false`
- 产物目录：`output/seedance2_test/`

注意：

- `first_frame` 与 `reference_video` 官方互斥，本流程不能把编辑图设为 `first_frame`
- Seedance 的 `reference_video` **必须是公网 HTTPS URL**（不支持本地路径 / base64 / 未开通的 `asset://`）
- 脚本会把本地图/视频上传到 TOS 私有桶，再生成 **预签名 HTTPS URL** 交给 Seedance

## 配置

同目录 `.env`：

```bash
API_KEY=你的ark_key

# TOS（Seedance 参考视频/图片公网拉取）
ACCESS_KEY_ID=...
ACCESS_KEY_SECRET=...   # 按控制台给出的字符串原样填写，不要再 base64 解码
TOS_ENDPOINT=tos-cn-beijing.volces.com
TOS_REGION=cn-beijing
TOS_BUCKET=knowin-oss
TOS_PRESIGN_EXPIRE=7200
```

脚本不会打印完整密钥。
