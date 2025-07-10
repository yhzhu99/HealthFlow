# extract_task/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# 建议使用环境变量存储API密钥，而不是硬编码
# 你需要在项目根目录下创建一个 .env 文件，内容为：
# OPENAI_API_KEY="sk-..."
DEEPSEEK_API_KEY="sk-36844880409740a59f3785d104cd8f1e"

# API Configuration
# 使用 DeepSeek 或 OpenAI 的 API endpoint
# 对于 DeepSeek，其 API 与 OpenAI 兼容
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.deepseek.com/v1" # DeepSeek的URL
# 如果用 OpenAI，可以注释掉上面一行，或者设为 "https://api.openai.com/v1"

MODEL_NAME = "deepseek-chat" # 或 "gpt-4-turbo", "gpt-3.5-turbo"

# File Paths
PDF_DIR = "./pdfs"
OUTPUT_DIR = "./outputs/tasks"