from app import run_query


def run_case(name, query):
    print("\n===============")
    print(f"测试用例：{name}")
    print("输入：", query)

    result = run_query(query)

    trace = result["execution_trace"]

    # 找到 critic 相关步骤
    critic_steps = [t for t in trace if t["step"] == "critic_node"]

    for c in critic_steps:
        print("\n--- Critic Round ---")
        print("status:", c["output"]["critic_result"]["status"])
        print("action:", c["output"]["critic_result"].get("action"))
        print("rewrite_query:", c["output"]["critic_result"].get("rewrite_query"))

    print("\n最终答案：")
    print(result["answer"])


def test_cases():
    run_case("模糊接口查询", "请帮我查一下系统接口信息")
    run_case("退款 UI", "帮我设计退款流程 UI 布局")
    run_case("长句订单查询", "请问订单系统查询接口是否支持根据用户ID筛选订单")


if __name__ == "__main__":
    test_cases()
