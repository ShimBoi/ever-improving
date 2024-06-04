# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR-1 model."""
import torch
import torch.nn as nn
import transformers
from flamingo_pytorch import PerceiverResampler
from models.transformer_utils import get_2d_sincos_pos_embed
from models.vision_transformer import Block
from omegaconf import OmegaConf as OC
from transformers import GPT2Model
from util.loss import masked_loss


import torch.nn.functional as F


class GR1(nn.Module):
    def __init__(
        self,
        model_clip,
        model_mae,
        state_dim,
        act_dim,
        hidden_size,
        sequence_length,
        chunk_size,
        training_target,
        img_feat_dim,
        patch_feat_dim,
        lang_feat_dim,
        resampler_params,
        without_norm_pixel_loss=False,
        use_hand_rgb=True,
        **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size

        # GPT
        self.hidden_size = hidden_size
        self.init_gpt2_model(hidden_size, **kwargs)

        # Perceiver
        self.init_perceiver_resampler(patch_feat_dim, resampler_params)

        # CLIP
        self.init_clip_model(model_clip)
        # MAE
        self.init_mae_model(model_mae)

        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224  # TODO: make this a parameter
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = "act_pred" in training_target
        self.fwd_pred = "fwd_pred" in training_target
        self.fwd_pred_hand = "fwd_pred_hand" in training_target

        self.without_norm_pixel_loss = without_norm_pixel_loss

        self.init_embedding_functions()
        self.init_queries()
        self.init_action_prediction()
        self.init_forward_prediction()

    @classmethod
    def from_hydra(cls, model_clip, model_mae, cn):
        resampler_params = {
            "depth": cn.perciever.resampler_depth,
            "dim_head": cn.perciever.resampler_dim_head,
            "heads": cn.perciever.resampler_heads,
            "num_latents": cn.perciever.resampler_num_latents,
            "num_media_embeds": cn.perciever.resampler_num_media_embeds,
        }

        return cls(
            model_clip,
            model_mae,
            state_dim=cn.state_dim,
            act_dim=cn.act_dim,
            hidden_size=cn.embed_dim,
            sequence_length=cn.seq_len,
            chunk_size=cn.chunk_size,
            training_target=cn.training_target,
            img_feat_dim=cn.img_feat_dim,
            patch_feat_dim=cn.patch_feat_dim,
            lang_feat_dim=cn.lang_feat_dim,
            resampler_params=resampler_params,
            without_norm_pixel_loss=cn.without_norm_pixel_loss,
            use_hand_rgb=cn.use_hand_rgb,
            **OC.to_container(cn.gpt_kwargs, resolve=True)
        )

    def init_gpt2_model(self, hidden_size, **kwargs):
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

    def init_perceiver_resampler(self, patch_feat_dim, resampler_params):
        self.n_patch_latents = resampler_params["num_latents"]
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params["depth"],
            dim_head=resampler_params["dim_head"],
            heads=resampler_params["heads"],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params["num_media_embeds"],
        )

    def init_clip_model(self, model_clip):
        """Initialize CLIP model and freeze its parameters."""
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False

    def init_mae_model(self, model_mae):
        """Initialize MAE model and freeze its parameters."""
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False

    def init_embedding_functions(self):

        self.embed_arm_state = nn.Linear(self.state_dim - 1, self.hidden_size)
        # one-hot gripper state
        self.embed_gripper_state = nn.Linear(2, self.hidden_size)
        self.embed_state = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # Relative timestep embedding
        self.embed_timestep = nn.Embedding(self.sequence_length, self.hidden_size)
        # Embedding function for languages
        self.embed_lang = nn.Linear(self.lang_feat_dim, self.hidden_size)

        # Embedding functions for images
        self.embed_hand_img = nn.Linear(self.img_feat_dim, self.hidden_size)
        self.embed_img = nn.Linear(self.img_feat_dim, self.hidden_size)
        self.embed_hand_patch = nn.Linear(self.patch_feat_dim, self.hidden_size)
        self.embed_patch = nn.Linear(self.patch_feat_dim, self.hidden_size)

        # Layer norm for embeddings
        self.embed_ln = nn.LayerNorm(self.hidden_size)

    def init_queries(self):

        # Action query token
        self.action_queries = nn.Embedding(1, self.hidden_size)
        self.action_chunk_queries = nn.Embedding(self.chunk_size, self.hidden_size)
        self.action_chunk_queries.weight.data.fill_(0)  # finetune it from zero weight

        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

    def init_action_prediction(self):

        hid2 = self.hidden_size // 2
        self.pred_act_mlps = nn.ModuleList(
            [nn.Linear(self.hidden_size, hid2), nn.Linear(hid2, hid2)]
        )
        self.pred_arm_act = nn.Linear(hid2, self.act_dim - 1)  # arm action
        self.pred_gripper_act = nn.Linear(hid2, 1)  # gripper action (binary)

    def init_forward_prediction(self):
        self.decoder_embed = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))

        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.hidden_size,
                    16,
                    4,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(self.hidden_size)
        self.decoder_pred = nn.Linear(
            self.hidden_size, self.patch_size**2 * 3, bias=True
        )  # decoder to patch
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, (self.image_size // self.patch_size) ** 2, self.hidden_size),
            requires_grad=False,
        )  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], (self.image_size // self.patch_size)
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

    def embed_state_info(self, state, batch_size, sequence_length):
        arm_state = state["arm"]
        gripper_state = state["gripper"]
        arm_state_embeddings = self.embed_arm_state(
            arm_state.view(batch_size, sequence_length, self.state_dim - 1)
        )
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        state_embeddings = torch.cat(
            (arm_state_embeddings, gripper_state_embeddings), dim=2
        )
        return self.embed_state(state_embeddings)

    def embed_language_info(self, language):
        lang_embeddings = self.model_clip.encode_text(language)
        lang_embeddings = lang_embeddings / (
            lang_embeddings.norm(dim=1, keepdim=True) + 1e-6
        )
        return self.embed_lang(lang_embeddings.float())

    def get_mae_features(self, rgb, hand_rgb, batch_size, sequence_length, c, h, w):
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size * sequence_length, c, h, w)
        )
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)
        hand_obs_embeddings, hand_patch_embeddings = None, None
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size * sequence_length, c, h, w)
            )
            hand_obs_embeddings = hand_obs_embeddings.view(
                batch_size, sequence_length, -1
            )
        return (
            obs_embeddings,
            patch_embeddings,
            hand_obs_embeddings,
            hand_patch_embeddings,
        )

    def prepare_forward_prediction(
        self, rgb, hand_rgb, batch_size, sequence_length, h, w
    ):
        obs_targets, obs_hand_targets = None, None
        if self.fwd_pred:
            p = self.patch_size
            h_p, w_p = h // p, w // p
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
            obs_targets = self.normalize_targets(
                rgb.permute(0, 1, 3, 5, 4, 6, 2).reshape(
                    shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3)
                )
            )
            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(
                    shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)
                )
                obs_hand_targets = self.normalize_targets(
                    hand_rgb.permute(0, 1, 3, 5, 4, 6, 2).reshape(
                        shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3)
                    )
                )
        return obs_targets, obs_hand_targets

    def normalize_targets(self, targets):
        if not self.without_norm_pixel_loss:
            targets = (targets - targets.mean(dim=-1, keepdim=True)) / (
                targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
            )
        return targets

    def process_patch_embeddings(
        self, patch_embeddings, batch_size, sequence_length, is_hand=False
    ):
        if is_hand and patch_embeddings is None:
            return None
        patch_embeddings = patch_embeddings.unsqueeze(1)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)
        patch_embeddings = patch_embeddings.squeeze(1)
        return patch_embeddings.view(
            batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim
        )

    def add_timestep_embeddings(
        self,
        state_embeddings,
        lang_embeddings,
        patch_embeddings,
        obs_embeddings,
        hand_obs_embeddings,
        hand_patch_embeddings,
        time_embeddings,
        batch_size,
    ):
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        patch_embeddings = patch_embeddings + time_embeddings.view(
            self.sequence_length, 1, self.hidden_size
        )
        obs_embeddings = obs_embeddings + time_embeddings
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(
                self.sequence_length, 1, self.hidden_size
            )
        return (
            state_embeddings,
            lang_embeddings,
            patch_embeddings,
            obs_embeddings,
            hand_obs_embeddings,
            hand_patch_embeddings,
        )

    def format_sequence(
        self,
        lang_embeddings,
        state_embeddings,
        patch_embeddings,
        obs_embeddings,
        hand_patch_embeddings,
        hand_obs_embeddings,
        batch_size,
        sequence_length,
    ):
        lang_embeddings = lang_embeddings.view(
            batch_size, sequence_length, 1, self.hidden_size
        )
        state_embeddings = state_embeddings.view(
            batch_size, sequence_length, 1, self.hidden_size
        )
        obs_embeddings = obs_embeddings.view(
            batch_size, sequence_length, 1, self.hidden_size
        )
        stacked_inputs = torch.cat(
            (lang_embeddings, state_embeddings, patch_embeddings, obs_embeddings), dim=2
        )
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(
                batch_size, sequence_length, 1, self.hidden_size
            )
            stacked_inputs = torch.cat(
                (stacked_inputs, hand_patch_embeddings, hand_obs_embeddings), dim=2
            )
        return self.add_action_obs_queries(stacked_inputs, batch_size, sequence_length)

    def add_action_obs_queries(self, stacked_inputs, batch_size, sequence_length):
        if self.act_pred:
            action_queries = self.action_queries.weight
            action_chunk_queries = self.action_chunk_queries.weight + action_queries
            action_chunk_queries = action_chunk_queries.view(
                1, 1, self.chunk_size, self.hidden_size
            ).repeat(batch_size, sequence_length, 1, 1)
            stacked_inputs = torch.cat((stacked_inputs, action_chunk_queries), dim=2)
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight
            obs_queries = obs_queries.view(
                1, 1, self.n_patch_latents + 1, self.hidden_size
            ).repeat(batch_size, sequence_length, 1, 1)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight
                obs_hand_queries = obs_hand_queries.view(
                    1, 1, self.n_patch_latents + 1, self.hidden_size
                ).repeat(batch_size, sequence_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
        return stacked_inputs

    def create_attention_mask(
        self, attention_mask, batch_size, sequence_length, stacked_inputs
    ):
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        if self.use_hand_rgb:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, self.get_total_tokens()
            )
        else:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, self.get_base_tokens()
            )
        return self.add_query_attention_masks(
            stacked_attention_mask, batch_size, sequence_length, stacked_inputs
        )

    def get_total_tokens(self):
        return 1 + 1 + self.n_patch_latents + 1 + self.n_patch_latents + 1

    def get_base_tokens(self):
        return 1 + 1 + self.n_patch_latents + 1

    def add_query_attention_masks(
        self, stacked_attention_mask, batch_size, sequence_length, stacked_inputs
    ):
        if self.act_pred:
            act_query_attention_mask = torch.zeros(
                (batch_size, sequence_length, self.chunk_size),
                dtype=torch.long,
                device=stacked_inputs.device,
            )
            stacked_attention_mask = torch.cat(
                (stacked_attention_mask, act_query_attention_mask), dim=2
            )
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros(
                (batch_size, sequence_length, self.n_patch_latents + 1),
                dtype=torch.long,
                device=stacked_inputs.device,
            )
            stacked_attention_mask = torch.cat(
                (stacked_attention_mask, obs_query_attention_mask), dim=2
            )
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros(
                    (batch_size, sequence_length, self.n_patch_latents + 1),
                    dtype=torch.long,
                    device=stacked_inputs.device,
                )
                stacked_attention_mask = torch.cat(
                    (stacked_attention_mask, obs_hand_query_attention_mask), dim=2
                )
        return stacked_attention_mask.reshape(batch_size, -1)

    def perform_transformer_forward(
        self, stacked_inputs, stacked_attention_mask, batch_size, sequence_length
    ):
        stacked_inputs = stacked_inputs.reshape(batch_size, -1, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask
        )
        return transformer_outputs["last_hidden_state"].reshape(
            batch_size, sequence_length, -1, self.hidden_size
        )

    def predict_actions(self, x, batch_size, sequence_length):
        arm_action_preds, gripper_action_preds = None, None
        if self.act_pred:
            action_embedding = x[:, :, -self.chunk_size :]
            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)
            gripper_action_preds = self.pred_gripper_act(action_embedding)
        return arm_action_preds, gripper_action_preds

    def predict_forward(self, x, batch_size, sequence_length):
        obs_preds, obs_hand_preds = None, None
        if self.fwd_pred:
            mask_tokens = self.mask_token.repeat(
                batch_size,
                sequence_length,
                (self.image_size // self.patch_size) ** 2,
                1,
            ) + self.decoder_pos_embed.unsqueeze(0).repeat(
                batch_size, sequence_length, 1, 1
            )
            obs_preds = self.decode_predictions(
                x, mask_tokens, batch_size, sequence_length, is_hand=False
            )
            if self.fwd_pred_hand:
                obs_hand_preds = self.decode_predictions(
                    x, mask_tokens, batch_size, sequence_length, is_hand=True
                )
        return obs_preds, obs_hand_preds

    def decode_predictions(
        self, x, mask_tokens, batch_size, sequence_length, is_hand=False
    ):
        start_idx = -self.n_patch_latents - 1 if is_hand else -self.chunk_size
        obs_pred = self.decoder_embed(x[:, :, start_idx:])
        obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2).reshape(
            -1, obs_pred.shape[-2], obs_pred.shape[-1]
        )
        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_).reshape(
            batch_size, sequence_length, -1, obs_pred_.shape[-1]
        )
        return obs_preds[:, :, -self.n_patch_latents :]

    def create_prediction_dict(
        self,
        obs_preds,
        obs_targets,
        obs_hand_preds,
        obs_hand_targets,
        arm_action_preds,
        gripper_action_preds,
    ):
        return {
            "obs_preds": obs_preds,
            "obs_targets": obs_targets,
            "obs_hand_preds": obs_hand_preds,
            "obs_hand_targets": obs_hand_targets,
            "arm_action_preds": arm_action_preds,
            "gripper_action_preds": gripper_action_preds,
        }

    def forward(self, rgb, hand_rgb, state, language, attention_mask):
        batch_size, sequence_length, c, h, w = rgb.shape

        state_embeddings = self.embed_state_info(state, batch_size, sequence_length)
        lang_embeddings = self.embed_language_info(language)

        obs_embeddings, patch_embeddings, hand_obs_embeddings, hand_patch_embeddings = (
            self.get_mae_features(rgb, hand_rgb, batch_size, sequence_length, c, h, w)
        )
        obs_targets, obs_hand_targets = self.prepare_forward_prediction(
            rgb, hand_rgb, batch_size, sequence_length, h, w
        )

        patch_embeddings = self.process_patch_embeddings(
            patch_embeddings, batch_size, sequence_length
        )
        hand_patch_embeddings = self.process_patch_embeddings(
            hand_patch_embeddings, batch_size, sequence_length, is_hand=True
        )

        obs_embeddings = self.embed_img(obs_embeddings.float())
        patch_embeddings = self.embed_patch(patch_embeddings.float())
        hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())
        hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())

        time_embeddings = self.embed_timestep.weight
        (
            state_embeddings,
            lang_embeddings,
            patch_embeddings,
            obs_embeddings,
            hand_obs_embeddings,
            hand_patch_embeddings,
        ) = self.add_timestep_embeddings(
            state_embeddings,
            lang_embeddings,
            patch_embeddings,
            obs_embeddings,
            hand_obs_embeddings,
            hand_patch_embeddings,
            time_embeddings,
            batch_size,
        )

        stacked_inputs = self.format_sequence(
            lang_embeddings,
            state_embeddings,
            patch_embeddings,
            obs_embeddings,
            hand_patch_embeddings,
            hand_obs_embeddings,
            batch_size,
            sequence_length,
        )
        stacked_attention_mask = self.create_attention_mask(
            attention_mask, batch_size, sequence_length, stacked_inputs
        )

        x = self.perform_transformer_forward(
            stacked_inputs, stacked_attention_mask, batch_size, sequence_length
        )

        arm_action_preds, gripper_action_preds = self.predict_actions(
            x, batch_size, sequence_length
        )
        obs_preds, obs_hand_preds = self.predict_forward(x, batch_size, sequence_length)

        return self.create_prediction_dict(
            obs_preds,
            obs_targets,
            obs_hand_preds,
            obs_hand_targets,
            arm_action_preds,
            gripper_action_preds,
        )

    def loss(self, pred, batch, cfg):
        obs_mask = batch["mask"][..., 0]

        loss = {}

        _masked_loss = lambda x, y: masked_loss(
            x, y, obs_mask, cfg.skip_frame, F.mse_loss
        )
        loss["rgb_static"] = _masked_loss(pred["obs_preds"], pred["obs_targets"])
        loss["rgb_gripper"] = _masked_loss(
            pred["obs_hand_preds"], pred["obs_hand_targets"]
        )

        _masked_loss = lambda x, y: masked_loss(
            x, y, batch['mask'], 0, F.smooth_l1_loss
        )
        loss["action_arm"] = _masked_loss(
            pred["arm_action_preds"], batch["actions"][..., :6]
        )
        loss["action_gripper"] = _masked_loss(
            pred["gripper_action_preds"], batch["actions"][..., -1:]
        )

        loss['total'] = (
            loss["rgb_static"]
            + loss["rgb_gripper"]
            + cfg.arm_loss_ratio * loss["action_arm"]
            + loss["action_gripper"]
        )
        return loss