#!/bin/zsh
#SBATCH --job-name=nccl_allreduce
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=0:30:00

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# NCCL configuration
export NODE_RANK=${JOB_COMPLETION_INDEX}

# ============================================
# 1. 动态节点发现 (Pod IP Discovery)
# ============================================
echo "Discovering nodes dynamically..."

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    GROUP_PREFIX=$(echo "$SLURM_JOB_NODELIST" | sed 's/-[0-9]\..*//')
else
    echo "ERROR: SLURM_JOB_NODELIST not set"
    exit 1
fi

max_retries=30
retry_count=0
expected_pods=2 
# 注意：如果你申请了2个节点，这里期望2个Pod IP

while [ $retry_count -lt $max_retries ]; do
    pod_output=$(kubectl get pods -o wide | grep $GROUP_PREFIX | sort)
    # Zsh array parsing
    ips=($(echo "$pod_output" | awk 'NF==9 && $6!~/<none>/ && $6!~/^$/ {print $6}'))
    valid_pods=${#ips[@]}
    
    echo "Attempt $((retry_count + 1)): Found $valid_pods valid IPs, expected $expected_pods"
    if [ $valid_pods -eq $expected_pods ]; then
        echo "All pods have valid IPs!"
        break
    fi
    sleep 1
    retry_count=$((retry_count + 1))
done

if [ $valid_pods -ne $expected_pods ]; then
    echo "ERROR: Failed to discover all pods"
    exit 1
fi

echo "Discovered IPs: ${ips[@]}"

# ============================================
# 2. 启动 SSHD
# ============================================
/home/jovyan/.pixi/bin/sshd \
    -h ~/.ssh/ssh_host_rsa_key \
    -h ~/.ssh/ssh_host_ecdsa_key \
    -h ~/.ssh/ssh_host_ed25519_key \
    -p 2222
sleep 2

# ============================================
# 3. 运行 NCCL Test (仅在 Rank 0 触发)
# ============================================
if [ "${NODE_RANK}" = "0" ]; then
    echo "Master node: Launching mpirun..."
    
    # --- 关键修改 1: 生成支持 8 卡并行的 Hostfile ---
    rm -f /tmp/hostfile
    for ip in "${ips[@]}"; do
        echo "$ip slots=8" >> /tmp/hostfile
    done
    
    echo "Hostfile content:"
    cat /tmp/hostfile
    
    # 清理 Slurm 变量防止 MPI 冲突
    unset SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS

    # --- 关键修改 2: mpirun 参数调整 ---
    # -np 16: 总共 16 个进程 (2节点 * 8卡)
    # 移除了 NCCL_P2P_DISABLE 和 NCCL_SHM_DISABLE
    # 修复了 --allow-run-as-root 后面的空格语法错误
    
    pixi run --frozen --no-install \
       mpirun \
        --allow-run-as-root \
        --mca plm_rsh_agent "/home/jovyan/.pixi/bin/ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222" \
        --mca plm ^slurm \
        --mca ras ^slurm \
        --hostfile /tmp/hostfile \
        -np 16 \
        --map-by slot \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_QPS_PER_CONNECTION=16 \
        -x NCCL_IB_SPLIT_DATA_ON_QPS=0 \
        -x NCCL_ALGO=ring \
        -x NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
        /home/jovyan/test-nccl/.pixi/envs/default/bin/sendrecv_perf \
            -b 1G \
            -e 8G \
            -f 2 \
            -t 1 \
            -c 0 \
            -n 20 \
            -w 10
else
    echo "Worker node (rank ${NODE_RANK}) - waiting..."
    while true; do sleep 60; done
fi