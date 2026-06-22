"""
进程组外协调器 —— 信号 / 屏障机制。

提供两种实现:
  - FileCoordinator(共享文件系统版,需要 4 个 pod 共享 /workspace/...)
  - TCPCoordinator(独立 TCPStore 版,不依赖共享存储,只需 pod 间网络互通)

两者接口完全一致(barrier / signal / wait_signal / cleanup_barrier /
clear_signal / cleanup_all / stop_server),可以无缝替换。

TCP 版用法::
    coord = TCPCoordinator(
        master_addr="10.244.x.x",   # rank 0 的 IP
        master_port=29502,           # 与 NCCL 端口 (29500) 错开
        rank=my_rank,
        world_size=4,
    )
    coord.barrier("pg_destroyed", my_rank)
    coord.signal("ckpt_loaded")
    coord.wait_signal("ckpt_loaded")
    coord.cleanup_barrier("pg_destroyed")
    coord.stop_server()              # 程序结束时调用(rank 0 关闭 server)
"""
import os
import time
from datetime import timedelta
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# FileCoordinator: 基于共享文件系统的协调器(原实现,需要真共享存储)
# ═════════════════════════════════════════════════════════════════════════════

class FileCoordinator:
    def __init__(self, base_dir: str, world_size: int, **kwargs):
        # **kwargs 忽略 TCP 版的额外参数(rank, master_addr, master_port),
        # 使得两种 coordinator 可以用完全相同的构造调用。
        self.base = Path(base_dir)
        self.W    = world_size
        self.base.mkdir(parents=True, exist_ok=True)

    # ── 屏障 ──────────────────────────────────────────────────────────────────

    def barrier(self, name: str, rank: int,
                timeout: float = 300.0, poll: float = 0.2) -> float:
        """到达屏障并等待全组。返回等待时间(秒)。"""
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / f"bar_{name}_{rank}").write_text(str(rank))
        t0 = time.monotonic()
        while True:
            count = sum(1 for r in range(self.W)
                        if (self.base / f"bar_{name}_{r}").exists())
            if count >= self.W:
                return time.monotonic() - t0
            if time.monotonic() - t0 > timeout:
                have = [r for r in range(self.W)
                        if (self.base / f"bar_{name}_{r}").exists()]
                raise TimeoutError(
                    f"FileBarrier '{name}' timeout {timeout:.0f}s; "
                    f"rank {rank}; present={have}/{self.W}")
            time.sleep(poll)

    def cleanup_barrier(self, name: str):
        for r in range(self.W):
            try:
                (self.base / f"bar_{name}_{r}").unlink(missing_ok=True)
            except Exception:
                pass

    # ── 单向信号 ──────────────────────────────────────────────────────────────

    def signal(self, name: str, value: str = "1"):
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / f"sig_{name}").write_text(value)

    def wait_signal(self, name: str,
                    timeout: float = 300.0, poll: float = 0.1) -> str:
        sig = self.base / f"sig_{name}"
        t0  = time.monotonic()
        while not sig.exists():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Signal '{name}' timeout {timeout:.0f}s")
            time.sleep(poll)
        return sig.read_text().strip()

    def clear_signal(self, name: str):
        (self.base / f"sig_{name}").unlink(missing_ok=True)

    def cleanup_all(self):
        import shutil
        try:
            shutil.rmtree(self.base)
        except Exception:
            pass
        self.base.mkdir(parents=True, exist_ok=True)

    def stop_server(self):
        """文件版无 TCP server,此方法为空(兼容 TCPCoordinator 接口)。"""
        pass


# ═════════════════════════════════════════════════════════════════════════════
# TCPCoordinator: 基于独立 TCPStore 的协调器(不依赖共享存储)
# ═════════════════════════════════════════════════════════════════════════════
#
# 设计要点:
#   1. TCPStore 是 torch.distributed 提供的 KV server,独立于 NCCL PG 生命周期。
#      即使 dist.destroy_process_group() 之后,TCPStore 仍可正常 set/get/wait,
#      因为它跟 NCCL 用的是两条独立的网络连接。
#
#   2. rank 0 起 server (is_master=True);其他 rank 起 client (is_master=False)。
#      client 构造时会阻塞重试,所以即使 server 晚启动也能等到。
#
#   3. TCPStore 内置 wait(keys, timeout) API,原生支持"等若干 key 出现"语义,
#      不需要轮询。但 timeout 是固定值,我们仍包一层 try/except 转成 TimeoutError。
#
#   4. server 端口必须与 NCCL 端口 (29500) 和 FileStore 端口错开。
#      默认用 29502。
#
#   5. cleanup 不能真删 key (TCPStore 没有 delete API,只能覆写)。
#      我们用"递增 epoch" 的方式让旧 key 自然过期(barrier name 加 epoch 前缀)。
#      简化起见,本实现直接用覆写(set 同名 key 为空字符串)处理 cleanup_barrier,
#      并在 wait 时检查值是否非空。
# ═════════════════════════════════════════════════════════════════════════════

class TCPCoordinator:
    def __init__(self, master_addr: str, master_port: int,
                 rank: int, world_size: int, **kwargs):
        import torch.distributed as dist
        self.rank = rank
        self.W    = world_size
        self.is_master = (rank == 0)
        # 给 store 内部 socket 一个长 timeout,wait_signal/barrier 各自有自己的 timeout
        self.store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=self.is_master,
            timeout=timedelta(seconds=600),
            wait_for_workers=True,   # rank 0 等所有 client 连上才返回
        )
        # 记下信息便于调试
        self.master_addr = master_addr
        self.master_port = master_port

    # ── 屏障 ──────────────────────────────────────────────────────────────────
    #
    # 每个 rank 写 "bar_{name}_{rank}" = "1",然后等所有 W 个 key 都出现。
    # TCPStore.wait(keys, timeout) 原生阻塞,不轮询。

    def barrier(self, name: str, rank: int,
                timeout: float = 300.0, poll: float = 0.2) -> float:
        """到达屏障并等待全组。返回等待时间(秒)。poll 参数仅为接口兼容,实际不轮询。"""
        keys = [f"bar_{name}_{r}" for r in range(self.W)]
        self.store.set(keys[rank], "1")
        t0 = time.monotonic()
        try:
            self.store.wait(keys, timedelta(seconds=timeout))
        except Exception as e:
            # TCPStore 超时抛 RuntimeError("Socket Timeout"),转成 TimeoutError 跟 FileBarrier 对齐
            # 查一下哪些已经到了,便于调试
            arrived = []
            for r, k in enumerate(keys):
                try:
                    self.store.get(k)
                    arrived.append(r)
                except Exception:
                    pass
            raise TimeoutError(
                f"TCPBarrier '{name}' timeout {timeout:.0f}s; "
                f"rank {rank}; present={arrived}/{self.W}; "
                f"underlying: {type(e).__name__}: {e}"
            )
        return time.monotonic() - t0

    def cleanup_barrier(self, name: str):
        # TCPStore 没有 delete,用一个 epoch 计数让后续 barrier 用新 key。
        # 但为了不改动 barrier 的 key schema,这里做"软清理":把所有 W 个 key 覆写为 "0"。
        # 后续如果同名 barrier 再次启动,会被新的 set("1") 覆盖,wait 仍能正确返回。
        # 注意:wait 不检查值,只要 key 存在就返回 —— 所以"覆写为 0"不能让 wait 重新阻塞。
        # 因此 cleanup_barrier 在 TCP 版下是 no-op,实际不影响正确性(每个 barrier name
        # 在一个 trial 内只用一次,cleanup 仅为节省内存)。
        pass

    # ── 单向信号 ──────────────────────────────────────────────────────────────

    def signal(self, name: str, value: str = "1"):
        # 空字符串语义上等价于"已清除",这里保证非空
        if value == "":
            value = "1"
        self.store.set(f"sig_{name}", value)

    def wait_signal(self, name: str,
                    timeout: float = 300.0, poll: float = 0.1) -> str:
        """阻塞等待 signal。poll 参数仅为接口兼容,TCPStore 不轮询。"""
        key = f"sig_{name}"
        try:
            self.store.wait([key], timedelta(seconds=timeout))
        except Exception as e:
            raise TimeoutError(
                f"TCPSignal '{name}' timeout {timeout:.0f}s; "
                f"underlying: {type(e).__name__}: {e}"
            )
        v = self.store.get(key)
        # store.get 返回 bytes,需要 decode
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

    def clear_signal(self, name: str):
        # 同样地,TCPStore 没有 delete。覆写为空再 set 会让 wait 直接返回(因 key 存在),
        # 所以 clear 在 TCP 版下也是 no-op。每个 signal name 在一个 trial 内只用一次。
        pass

    def cleanup_all(self):
        # 整个 store 没有 wipe API。trial 间复用 store 是安全的,因为每个 trial
        # 用唯一的 inject_step 后缀(crashed_20 等),不会冲突。
        pass

    def stop_server(self):
        """rank 0 持有 server,Python 进程退出时 socket 会自动关。这里显式删除引用。"""
        # 在某些 PyTorch 版本里显式 del 才能释放端口,以便同一进程下次重启时复用
        try:
            del self.store
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# 工厂函数: 根据 config 自动选用合适的 coordinator
# ═════════════════════════════════════════════════════════════════════════════

def make_coordinator(cfg_injection: dict, *, rank: int, world_size: int,
                     coord_dir: str, master_addr: str):
    """
    根据 cfg["injection"]["coordinator"] 字段选用 coordinator 实现。

      coordinator: "file"  → FileCoordinator(需要真共享 coord_dir)
      coordinator: "tcp"   → TCPCoordinator(只需网络互通,coord_dir 仅用作 ckpt 路径)
      未设置 / 其他       → "file"(向后兼容)
    """
    kind = str(cfg_injection.get("coordinator", "file")).lower()
    if kind == "tcp":
        port = int(cfg_injection.get("coord_tcp_port", 29502))
        return TCPCoordinator(
            master_addr=master_addr,
            master_port=port,
            rank=rank,
            world_size=world_size,
        )
    return FileCoordinator(coord_dir, world_size)
