# 可选的双 ubatch prefill

`--prefill-ubatches 1|2` 控制模型层同时执行的物理 ubatch 数量，默认 `1`。
物理批大小仍由 `--ubatch-size` 控制。Dense Qwen3、Llama、Granite 和混合 Qwen3.5
共用同一调度器；默认单批执行。

```sh
./bin/open_rknpu2_lm_infer --model MODEL --ubatch-size 128 --prefill-ubatches 2 bench -p 4096 -n 0
```

开启后，每次有两个完整物理批时启动两条执行 lane。前一批完成整层后，后一批才能进入
同一层；两条 lane 可以同时执行不同层。每层 KV、GDN 递归状态和卷积历史的更新顺序
保持不变。最后不足两个完整物理批的输入，以及逐 token decode，按单批执行。
逐行 NLL 评分和 speculative verification 仍按单批计算。

两条 lane 共用一份层对象、只读权重、设备 context/domain 以及每层已提交状态；各自持有
激活缓冲、算子 scratch 与执行 workspace。GDN canonical state 不复制到 lane 中。
因此开启 `2` 会增加 scratch 内存占用。关闭时不创建第二条 lane。
调优策略由第一条 lane 准备后复制给第二条，第二条不单独搜索。
第二条 lane 只准备完整 ubatch 的 workspace；decode、尾批和 verification 的形状留在
第一条 lane。Qwen3.5 仅交换最后一批的 hidden 结果，GDN scratch 固定在各自 lane。
独立执行 workspace 覆盖量化 Linear、Attention 和各 lane 的 GDN scratch；
FloatBackend 继续沿用自身的互斥 workspace，未验证浮点 decoder 的并发收益。

该选项不启用实验性的 CPU 阶段锁、按提交交接 CPU、三 lane 或算子内部新流水。
这些实验代码没有并入本次集成。

Qwen3.5 的多模态输入沿用各 token 的三轴位置和注入特征。双批不跨越前缀 checkpoint
边界，保存/恢复前会排空流水；MTP catch-up、特征提交和 DFlash context 更新继续按
原来的物理批顺序执行。
第二批的特征 pending 缓冲在双批组之间持久复用，prepared 位置切片直接借用。

实现分工：`model/pipeline.rs` 管理层存储、线程和依赖顺序；`model/prefill.rs` 管理可合并的
批次及 checkpoint 边界；`npu/lane.rs` 在作用域内为算子调用选择独立 workspace。
模型适配层仅提供原有的层计算和批次提交，不复制 Layer，也不各自维护握手协议。

统计字段 `prefill_ubatches` 给出配置值，`dual_prefill_groups` 给出自 reset 后实际执行的
双批组数。`linear_ms`、`attention_ms`、`cpu_ms` 是各 lane 阶段耗时之和，双 lane 时有重叠，
不能相加当作端到端时间；端到端 prefill 应使用调用者的 wall time。

回归测试包括参数默认值/范围、共享权重的独立执行 workspace，以及
两种模型在完整批、尾批、已有状态、后续 decode、reset/restore 下的完整词表 logits 对照，
以及 Qwen3.5 的 canonical state、卷积历史、特征、MTP 和 checkpoint 回归。
