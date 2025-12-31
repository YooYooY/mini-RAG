from core.graph_builder import build_graph
from core.memory import init_task_memory, create_init_state, memory_store
from resume.resume_runner import resume_from_checkpoint

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def run_query(query: str):
    app = build_graph()

    task_id = init_task_memory()
    init_state = create_init_state(task_id, query)
    result = app.invoke(
        init_state,
        config= {
            "llm": llm
        }
    )

    # print("\nresult=>", result)

    # resume
    # result = resume_from_checkpoint(app, "2cb60d12-57be-4015-8174-fd0dc54f30a6")

    trace = memory_store[result["task_id"]]["execution_trace"]
    # # 找到 critic 相关步骤
    critic_steps = [t for t in trace if t["step"] == "critic_node"]

    for c in critic_steps:
        print("\n--- Critic Round ---")
        print("status:", c["output"]["critic_result"]["status"])
        print("action:", c["output"]["critic_result"].get("action"))
        print("rewrite_query:", c["output"]["critic_result"].get("rewrite_query"))

    print("\n=== 最终答案 ===")
    print(result["answer"])


if __name__ == "__main__":

    run_query("请帮我查询订单查询 API 的接口说明")

    # print("===模糊接口查询===")
    # run_query("请帮我查一下系统接口信息")

    # print("===退款 UI===")
    # run_query("帮我设计退款流程 UI 布局")

    # print("===长句订单查询===")
    # run_query("请问订单系统查询接口是否支持根据用户ID筛选订单")
