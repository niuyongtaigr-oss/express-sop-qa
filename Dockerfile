# 快递 SOP 智能问答系统 — 生产镜像
# 构建: docker build -t express-sop-qa .
# 运行: docker compose up -d  (推荐, 见 docker-compose.yml)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖 (利用层缓存: 依赖不变时不会重复装)
# 先升级 pip: 基座自带 pip 25.0.1 解析器在部分依赖组合下会误报 ResolutionImpossible
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# 拷贝源码与默认知识库/评测集
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# 非 root 运行
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 配置全部走环境变量 (前缀 SOP_QA_, 见 .env.example)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
