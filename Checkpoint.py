import json
import os

CHECKPOINT_DIR = "./checkpoints"


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path(task_id: str):
    return os.path.join(CHECKPOINT_DIR, f"{task_id}.json")


# ===============================
# 写入 Checkpoint
# ===============================
def save_checkpoint(task_id: str, memoryStore, state, step: str):
    ensure_checkpoint_dir()

    payload = {
        "task_id": task_id,
        "last_step": step,
        "state": state,
        "memory": memoryStore,
    }

    with open(checkpoint_path(task_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ===============================
# 读取 Checkpoint
# ===============================
def load_checkpoint(task_id: str):
    path = checkpoint_path(task_id)

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================
# 判断是否已有 checkpoint
# ===============================
def has_checkpoint(task_id: str) -> bool:
    return os.path.exists(checkpoint_path(task_id))


# ===============================
# 恢复 Checkpoint
# ===============================
def resume_from_checkpoint(app, task_id: str, memory_store):
    ckpt = load_checkpoint(task_id)

    if not ckpt:
        print("⚠️ 没有 checkpoint，执行完整流程")
        return None

    last_step = ckpt["last_step"]
    step = ckpt["state"]

    print(f"🔁 从 checkpoint 恢复: {last_step}")

    memory_store[task_id] = ckpt["memory"]

    return app.invoke(step)
