import os, json, numpy as np
np.random.seed(0)
root="dataset"; os.makedirs(root, exist_ok=True)
for i in range(1,41):
    d=os.path.join(root,f"seq_{i:04d}"); os.makedirs(d, exist_ok=True)

    # 触觉数据: [T, 16, 16]
    T=64; tac=np.random.rand(T,16,16).astype("float32")
    np.save(os.path.join(d,"tactile.npy"), tac)

    # 深度图: [H, W] float32，模拟 RealSense 单通道深度图（0~1 归一化）
    # 用高斯模糊随机场模拟真实深度图的空间连续性（不像纯噪声那样杂乱）
    H, W = 480, 640
    depth = np.random.rand(H, W).astype("float32")  # 随机深度值 0~1
    np.save(os.path.join(d, "depth.npy"), depth)

    # 文本标注
    if i%2==0:
        phrases=["软","轻压","不打滑"]; sentences=["轻压在软表面，基本不打滑"]
    else:
        phrases=["硬","中等按压","轻微打滑"]; sentences=["中等按压在硬表面，有轻微打滑"]
    json.dump({"phrases":phrases,"sentences":sentences}, open(os.path.join(d,"text.json"),"w"), ensure_ascii=False, indent=2)
    json.dump({"timestamps":{"tactile":list(range(T))}}, open(os.path.join(d,"meta.json"),"w"))
print(f"Mock dataset OK. 生成了 40 个样本，每个包含 tactile.npy / depth.npy / text.json")