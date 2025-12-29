from .vector_store_chroma import vector_index

sample_docs = [
    {
        "id": "orders-api-001",
        "title": "订单查询接口",
        "text": """
GET /api/orders/{order_id}

参数：
- order_id: 订单ID

功能：
根据订单ID返回订单详情，包括状态、价格、物流信息。
""",
    },
    {
        "id": "orders-api-002",
        "title": "订单列表查询接口",
        "text": """
GET /api/orders?user_id={uid}

参数：
- user_id: 用户ID

功能：
返回用户最近 50 条订单，支持状态筛选、时间范围筛选。
""",
    },
]

vector_index.add_documents(sample_docs)
