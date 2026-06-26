import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    ):
    """
    forward pass kernel 2 (inner loop, over K,V) # TODO: verify
    """

    # stage-wise handling
    if STAGE == 1:
        # causal attention (mask out)
        low, high = 0, block_index_q * BLOCK_SIZE_Q
    if STAGE == 2:
        # for the block where there is transition between non-masked and masked keys
        low, high = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        low - tl.multiple_of(low, BLOCK_SIZE_Q) # ???
    else:
        # non-causal attention (no masking out)
        low, high = 0, SEQ_LEN

    # point to relevant K and V blocks respectively
    K_block_ptr = tl.advance(K_block_ptr, (0, low))
    V_block_ptr = tl.advance(V_block_ptr, (low, 0))

    # loop over K, V and update accumulator
    for start_kv in range(low, high, BLOCK_SIZE_KV):
        # letting the compiler know that start_n is a multiple of BLOCK_N, so that the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # compute q @ k
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2: # for diagonal elements in q @ k
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :]) # ???
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6) # ???
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # compute the maximum value of q @ k or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        # compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # this is the Correction Factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)

        # apply the Correction Factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        # load the V_block to SRAM from HBM
        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        # this computes: O_new = P * V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        # update the m_i
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


@triton.autotune(
        [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [64, 128] # try, select best
            for BLOCK_SIZE_KV in [32, 64] # try, select best
            for num_stages in [[3, 4, 7]] # ???
            for num_warps in [2, 4] # try, select best # TODO: verify
        ],
        key=["SEQ_LEN", "HEAD_DIM"], # run across each pair of SEQ_LEN, HEAD_DIM, select 'max-throughput-in-least-time' config       
)


@triton.jit 
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE = tl.constexpr,
    ):
    """
    forward pass kernel 1
    """

    tl.static_assert(BLOCK_SIZE_KV <= NUM_HEADS)

    # indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # indicates which head and batch to process
    index_batch_head = tl.program_id(1)

    # indicates which batch this program is associated with
    index_batch = index_batch_head // NUM_HEADS

    # indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by indexing it by the batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # block pointer indexing for Q, K, V, O
    # makes pointer indexing easier by defining block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, # pointer moves from starting of Q to Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, # pointer moves from starting of V to V[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, # pointer moves from starting of K to K[index_batch, index_head, :, :]
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq), # invert the stride wrt Q (to transpose the matrix)
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset, # pointer moves from starting of O to O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    # offs_q: offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offs_kv: offsets for the tokens in the K, V sequences to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running max, we have one for each query
    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) - float('inf') 

    # l_i: the running sum, we have one for each query (as we sum attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # acc: the accumulator for the output block, which is a group of rows of the O block
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    if STAGE == 1 or STAGE == 3: # TODO: check again
        # non-causal attention 
        O_block, l_i, m_i = _attn_fwd_inner( # forward pass kernel 2 called here
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3: # TODO: check again
        # causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # needed for computing logsumexp (for the backward pass)
    m_i += tl.math.log(l_i)

    # normalize the block at the end, after computing all normalization factors for all rows for the current output block
    O_block = O_block / l_i[:, None]

    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q #???

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_type)) # ??? 
