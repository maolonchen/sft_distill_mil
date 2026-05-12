"""
Block Expansion + Local Distillation for Qwen

在原始模型的指定 Transformer 层后插入恒等块，
通过局部 KL 散度（恒等块前后 hidden states 经 lm_head 投射后的分布差异）
+ MSE 特征蒸馏来缓解灾难性遗忘。

无需额外的 Teacher 模型，显存开销减半。

支持任意 Qwen3 模型（0.6B / 1.7B / 4B / 8B / 32B 等）
和多种插入策略。
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 插入策略
# ---------------------------------------------------------------------------

def get_insert_positions(
    strategy: str,
    num_layers: int,
    **kwargs,
) -> list[int]:
    """根据策略生成需要在哪些原始层之后插入恒等块。

    Args:
        strategy: 插入策略名称
            - "second_half": 后半部分每层后插入（默认）
            - "every_layer": 每层后都插入
            - "every_n": 每隔 N 层后插入（需传 n=N）
            - "first_half": 前半部分每层后插入
            - "custom": 自定义位置（需传 positions=[...]）
        num_layers: 原始模型的总层数
        **kwargs: 策略参数

    Returns:
        排序后的原始层索引列表，表示在这些层之后插入恒等块

    Examples:
        >>> get_insert_positions("second_half", 28)
        [14, 15, 16, ..., 27]

        >>> get_insert_positions("every_layer", 4)
        [0, 1, 2, 3]

        >>> get_insert_positions("every_n", 8, n=2)
        [1, 3, 5, 7]

        >>> get_insert_positions("custom", 28, positions=[0, 13, 27])
        [0, 13, 27]
    """
    if strategy == "second_half":
        start = num_layers // 2
        return list(range(start, num_layers))

    if strategy == "every_layer":
        return list(range(num_layers))

    if strategy == "first_half":
        return list(range(0, num_layers // 2))

    if strategy == "every_n":
        n = kwargs.get("n", 2)
        return list(range(n - 1, num_layers, n))

    if strategy == "custom":
        positions = kwargs.get("positions", [])
        return sorted(set(positions))

    raise ValueError(
        f"Unknown strategy: {strategy!r}. "
        f"Choose from: second_half, every_layer, every_n, first_half, custom"
    )


# ---------------------------------------------------------------------------
# 层映射
# ---------------------------------------------------------------------------

def _build_layer_mapping(
    num_original: int,
    insert_positions: list[int],
) -> tuple[list[int], list[int]]:
    """构建原始层到扩展层的索引映射和恒等块位置。

    核心公式：对于原始层 i，
        expanded_idx = i + count(p < i for p in insert_positions)
    即前面插了多少个恒等块，就往后偏移多少。

    Args:
        num_original: 原始模型层数
        insert_positions: 排序后的插入位置列表

    Returns:
        (layer_mapping, identity_indices)
        - layer_mapping[original_idx] = expanded_idx
        - identity_indices: 恒等块在扩展模型中的层索引列表
    """
    insert_set = set(insert_positions)

    # 原始层 → 扩展层映射
    layer_mapping = []
    offset = 0
    insert_iter = iter(sorted(insert_positions))
    next_insert = next(insert_iter, None)

    for i in range(num_original):
        while next_insert is not None and next_insert < i:
            offset += 1
            next_insert = next(insert_iter, None)
        layer_mapping.append(i + offset)

    # 恒等块在扩展模型中的索引 = mapping[p] + 1
    identity_indices = []
    for p in sorted(insert_positions):
        student_idx = layer_mapping[p] + 1
        identity_indices.append(student_idx)

    return layer_mapping, identity_indices


# ---------------------------------------------------------------------------
# 权重复制
# ---------------------------------------------------------------------------

def _copy_decoder_layer(src: nn.Module, dst: nn.Module):
    """逐个子模块复制 DecoderLayer 权重。

    适用于所有 Qwen3 模型（0.6B ~ 32B），因为架构相同。
    """
    # self_attn
    dst.self_attn.q_proj.weight.data.copy_(src.self_attn.q_proj.weight.data)
    dst.self_attn.k_proj.weight.data.copy_(src.self_attn.k_proj.weight.data)
    dst.self_attn.v_proj.weight.data.copy_(src.self_attn.v_proj.weight.data)
    dst.self_attn.o_proj.weight.data.copy_(src.self_attn.o_proj.weight.data)
    dst.self_attn.q_norm.weight.data.copy_(src.self_attn.q_norm.weight.data)
    dst.self_attn.k_norm.weight.data.copy_(src.self_attn.k_norm.weight.data)

    # mlp
    dst.mlp.gate_proj.weight.data.copy_(src.mlp.gate_proj.weight.data)
    dst.mlp.up_proj.weight.data.copy_(src.mlp.up_proj.weight.data)
    dst.mlp.down_proj.weight.data.copy_(src.mlp.down_proj.weight.data)

    # layer norms
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(
        src.post_attention_layernorm.weight.data
    )


# ---------------------------------------------------------------------------
# 模型创建
# ---------------------------------------------------------------------------

def create_expanded_model(
    model_path: str,
    insert_positions: list[int],
    source_model: Optional[AutoModelForCausalLM] = None,
) -> AutoModelForCausalLM:
    """创建含恒等块的扩展模型。

    在 insert_positions 指定的每个原始层后插入一个恒等块。
    恒等块通过将 o_proj 和 down_proj 权重置零来实现输入=输出。

    Args:
        model_path: 原始模型路径（支持任意 Qwen3 模型）
        insert_positions: 排序后的原始层索引列表，在这些层之后插入恒等块
        source_model: 可选的已加载模型，用于复制权重（避免重复加载）

    Returns:
        扩展后的模型
    """
    insert_positions = sorted(insert_positions)

    # 复用已有模型或从磁盘加载
    if source_model is not None:
        original_config = source_model.config
        original_model = source_model
        owns_model = False
    else:
        original_config = AutoConfig.from_pretrained(model_path)
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        )
        owns_model = True
    num_original = original_config.num_hidden_layers

    # 构建层映射
    layer_mapping, identity_indices = _build_layer_mapping(
        num_original, insert_positions
    )
    num_total = num_original + len(insert_positions)

    # 构建扩展后的 config
    new_config = copy.deepcopy(original_config)
    new_config.num_hidden_layers = num_total

    # 扩展 layer_types
    old_layer_types = list(original_config.layer_types)
    insert_set = set(insert_positions)
    new_layer_types = []
    for i in range(num_original):
        new_layer_types.append(old_layer_types[i])
        if i in insert_set:
            new_layer_types.append("full_attention")
    new_config.layer_types = new_layer_types

    # 用新 config 创建模型（随机初始化）
    new_model = AutoModelForCausalLM.from_config(new_config)
    new_model = new_model.to(torch.bfloat16)

    # --- 权重复制 ---
    new_model.model.embed_tokens.weight.data.copy_(
        original_model.model.embed_tokens.weight.data
    )
    new_model.model.norm.weight.data.copy_(original_model.model.norm.weight.data)
    new_model.lm_head.weight.data.copy_(original_model.lm_head.weight.data)

    for orig_idx, expanded_idx in enumerate(layer_mapping):
        _copy_decoder_layer(
            original_model.model.layers[orig_idx],
            new_model.model.layers[expanded_idx],
        )

    # --- 恒等块初始化 ---
    for identity_idx in identity_indices:
        layer = new_model.model.layers[identity_idx]
        nn.init.zeros_(layer.self_attn.o_proj.weight)
        nn.init.zeros_(layer.mlp.down_proj.weight)

    if owns_model:
        del original_model
    return new_model


# ---------------------------------------------------------------------------
# 局部蒸馏训练封装
# ---------------------------------------------------------------------------

class BlockExpansionWrapper(nn.Module):
    """封装含恒等块的扩展模型，通过局部蒸馏约束防遗忘。

    无需 Teacher 模型。三个损失：
    1. task_loss: SFT 交叉熵
    2. local_kl_loss: 恒等块前后 hidden states 经 lm_head 投射后的 KL 散度
    3. feat_loss: 恒等块前后 hidden states 的 MSE
    """

    def __init__(
        self,
        model_path: str,
        strategy: str = "second_half",
        strategy_kwargs: Optional[dict] = None,
        temperature: float = 2.0,
        lambda_kl: float = 0.5,
        lambda_feat: float = 0.1,
    ):
        """
        Args:
            model_path: 原始模型路径
            strategy: 插入策略（见 get_insert_positions）
            strategy_kwargs: 策略参数，如 {"n": 2} 或 {"positions": [0, 13, 27]}
            temperature: 局部 KL 蒸馏温度
            lambda_kl: 局部 KL 损失权重
            lambda_feat: 特征蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.lambda_feat = lambda_feat

        # 加载原始模型（仅用于创建扩展模型）
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        )

        # 计算插入位置
        num_layers = original_model.config.num_hidden_layers
        strategy_kwargs = strategy_kwargs or {}
        self.insert_positions = get_insert_positions(strategy, num_layers, **strategy_kwargs)

        # 构建层映射
        self.layer_mapping, self.identity_indices = _build_layer_mapping(
            num_layers, self.insert_positions
        )
        self.identity_indices_set = set(self.identity_indices)

        # 创建扩展模型（复用原始模型权重）
        self.model = create_expanded_model(
            model_path, self.insert_positions, source_model=original_model
        )

        # 释放原始模型
        del original_model

        # 冻结原始层，只保留恒等块可训练
        self._freeze_original_layers()

    def _freeze_original_layers(self):
        """冻结所有原始层及 embed_tokens/norm/lm_head，只保留恒等块可训练。"""
        for idx, layer in enumerate(self.model.model.layers):
            if idx not in self.identity_indices_set:
                for param in layer.parameters():
                    param.requires_grad = False

        # embed_tokens, norm, lm_head 也冻结
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model.model.norm.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = False

    def get_trainable_params(self) -> list[nn.Parameter]:
        """返回所有可训练参数（仅恒等块）。"""
        params = []
        for idx in self.identity_indices:
            params.extend(self.model.model.layers[idx].parameters())
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """前向传播，计算三部分损失。

        Returns:
            dict with keys: total_loss, task_loss, kl_loss, feat_loss
        """
        # 扩展模型前向传播，获取所有层的 hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # 1. Task loss（交叉熵）
        task_loss = outputs.loss

        hidden_states = outputs.hidden_states

        # 2 & 3. 局部 KL 散度损失 + 特征蒸馏损失
        kl_loss = torch.tensor(0.0, device=input_ids.device)
        feat_loss = torch.tensor(0.0, device=input_ids.device)

        for idx in self.identity_indices:
            # hidden_states 索引: [0]=embedding, [i+1]=layer i 的输出
            # identity block 在 model.layers[idx]
            # 其输入 = hidden_states[idx]（上一层输出）
            # 其输出 = hidden_states[idx + 1]
            before_hidden = hidden_states[idx].detach()  # 恒等块输入（切断梯度，不训练前面的层）
            after_hidden = hidden_states[idx + 1]        # 恒等块输出（保留梯度）

            # feat_loss: MSE 约束恒等块前后 hidden state 接近
            feat_loss = feat_loss + F.mse_loss(after_hidden.float(), before_hidden.float())

            # local_kl: 恒等块前后 hidden state 经 lm_head 投射后的 KL 散度
            with torch.no_grad():
                before_logits = self.model.lm_head(before_hidden).float()
                before_probs = F.softmax(before_logits / self.temperature, dim=-1)

            after_logits = self.model.lm_head(after_hidden).float()
            after_log_probs = F.log_softmax(after_logits / self.temperature, dim=-1)

            kl = F.kl_div(
                after_log_probs, before_probs, reduction="batchmean"
            ) * (self.temperature ** 2)
            kl_loss = kl_loss + kl.clamp(min=0.0)

        num_identities = len(self.identity_indices)
        if num_identities > 0:
            kl_loss = kl_loss / num_identities
            feat_loss = feat_loss / num_identities

        # 总损失
        total_loss = task_loss + self.lambda_kl * kl_loss + self.lambda_feat * feat_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "kl_loss": kl_loss,
            "feat_loss": feat_loss,
        }
