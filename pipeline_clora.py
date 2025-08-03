# pipeline_clora_xl.py

import inspect
import sys
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from pytorch_metric_learning import losses

# Import utility classes from a 'utils.py' file as in the original CLoRA repo
# For this self-contained script, we assume these are available.
# (AttentionStore, register_attention_control)
# If you don't have a utils.py, you would define these helper classes here.
from utils import AttentionStore, register_attention_control 

# Helper functions for masking, which are part of the core CLoRA logic
def erode(image, kernel_size=3, stride=1, padding=1):
    """Erodes a binary image tensor."""
    return 1 - F.max_pool2d(1 - image, kernel_size, stride, padding)

def dilate(image, kernel_size=3, stride=1, padding=1):
    """Dilates a binary image tensor."""
    return F.max_pool2d(image, kernel_size, stride, padding)


class CloraXLPipeline(StableDiffusionXLPipeline):
    r"""
    The definitive CLoRA pipeline for Stable Diffusion XL.

    This pipeline synthesizes the best practices from multiple approaches:
    - It INHERITS from the official `StableDiffusionXLPipeline` for maximum robustness,
      maintainability, and access to all of SDXL's features.
    - It implements the core CLoRA logic (per-LoRA forward passes, attention-based
      loss, latent updates, and mask-based noise blending) in a clean and
      readable `__call__` method.
    - It requires no custom `__init__`, leveraging the parent class's `from_pretrained`
      for simplicity and reliability.
    """

    @torch.no_grad()
    def attention_map_to_mask(
        self,
        attention_maps: List[torch.Tensor],
        mask_indices: List[List[int]],
        size: Tuple[int, int],
        mask_threshold_alpha: float = 0.3,
        mask_erode: bool = False,
        mask_dilate: bool = False,
        mask_opening: bool = False,
        mask_closing: bool = False,
    ):
        """Converts attention maps to binary masks for noise blending."""
        masks = []
        for attention_map, mask_indice_list in zip(attention_maps, mask_indices):
            if not mask_indice_list:
                # If no indices are provided for a LoRA, its mask is zero everywhere.
                mask = torch.zeros_like(attention_map[:, :, 0])
            else:
                # Aggregate attention for all specified token indices
                multi_token_maps = [attention_map[:, :, idx] for idx in mask_indice_list]
                attn_map = torch.stack(multi_token_maps).sum(dim=0)
                
                # Binarize the mask based on the threshold
                multi_token_map = torch.where(attn_map > (attn_map.max() * mask_threshold_alpha), 1.0, 0.0)
                mask = multi_token_map

            mask = mask.unsqueeze(0).unsqueeze(0)
            # Apply morphological operations
            if mask_opening:
                mask = erode(dilate(mask, 3, 1, 1), 3, 1, 1)
            if mask_closing:
                mask = dilate(erode(mask, 3, 1, 1), 3, 1, 1)
            if mask_dilate:
                mask = dilate(mask, 3, 1, 1)
            if mask_erode:
                mask = erode(mask, 3, 1, 1)

            masks.append(mask.squeeze(0).squeeze(0))

        masks = torch.stack(masks)
        # Ensure the background (first LoRA) covers any empty areas
        masks[0] = torch.where(masks.sum(dim=0) == 0, 1.0, masks[0])

        return F.interpolate(masks.unsqueeze(1), size=size, mode="bilinear", align_corners=False)

    @torch.enable_grad()
    def loss_fn(
        self,
        attention_maps: List[torch.Tensor],
        important_token_indices: List[List[List[int]]],
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """Calculates the contrastive loss between attention maps of different concepts."""
        classes, embeddings = [], []
        for class_id, concept_group in enumerate(important_token_indices):
            for prompt_id, token_indices_per_prompt in enumerate(concept_group):
                for token_idx in token_indices_per_prompt:
                    # Flatten and normalize the attention map for the specific token
                    embedding = attention_maps[prompt_id][:, :, token_idx].view(-1)
                    embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-6)
                    embeddings.append(embedding)
                    classes.append(class_id)
        
        if not embeddings:
            return torch.tensor(0.0, device=attention_maps[0].device)

        classes = torch.tensor(classes, device=attention_maps[0].device)
        embeddings = torch.stack(embeddings, dim=0)

        loss_func = losses.NTXentLoss(temperature=temperature)
        return loss_func(embeddings, classes)

    @torch.enable_grad()
    def update_latents(self, latent: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Updates latents by backpropagating the contrastive loss."""
        grads = torch.autograd.grad(loss, [latent], retain_graph=True)[0]
        return latent - step_size * grads

    @torch.no_grad()
    def __call__(
        self,
        # ---- CLoRA-specific arguments ----
        prompt_list: List[str],
        lora_list: List[str],
        negative_prompt_list: List[str],
        important_token_indices: List[List[List[int]]],
        mask_indices: List[List[int]],
        style_lora: str = "",
        style_lora_weight: float = 1.0,
        latent_update: bool = True,
        max_iter_to_alter: int = 25,
        step_size: float = 0.02,
        mask_threshold_alpha: float = 0.3,
        mask_erode: bool = False,
        mask_dilate: bool = False,
        mask_opening: bool = False,
        mask_closing: bool = False,
        # ---- Standard SDXL arguments ----
        prompt: Optional[Union[str, List[str]]] = None, # Will be ignored
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None, # Will be ignored
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        # 1. Setup
        device = self._execution_device
        attn_res = (height // 32, width // 32)
        attention_store = AttentionStore(attn_res)
        register_attention_control(self.unet, attention_store)

        # 2. Prepare Timesteps and Latents
        timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps, device)
        latents = self.prepare_latents(num_images_per_prompt, self.unet.config.in_channels, height, width, self.text_encoder.dtype, device, generator, latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 3. Pre-compute embeddings for each LoRA
        prompt_embeds_list = []
        add_text_embeds_list = []
        add_time_ids_list = []

        for i, lora_name in enumerate(lora_list):
            lora_names = [lora_name] if lora_name else []
            if style_lora:
                lora_names.append(style_lora)
            
            lora_weights = [1.0] * len(lora_names)
            if style_lora:
                lora_weights[-1] = style_lora_weight

            self.set_adapters(lora_names, lora_weights)

            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=prompt_list[i],
                prompt_2=prompt_list[i], # Use same prompt for both encoders
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt_list[i],
                negative_prompt_2=negative_prompt_list[i],
                clip_skip=clip_skip,
            )

            # For classifier-free guidance, we concatenate the embeddings
            current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            current_add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            current_add_time_ids = self._get_add_time_ids((height, width), (0,0), (height, width), self.text_encoder.dtype).to(device)
            current_add_time_ids = torch.cat([current_add_time_ids, current_add_time_ids], dim=0)

            prompt_embeds_list.append(current_prompt_embeds)
            add_text_embeds_list.append(current_add_text_embeds)
            add_time_ids_list.append(current_add_time_ids)

        # 4. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ---- CLoRA Core Logic ----
                with torch.enable_grad():
                    latents.requires_grad_(True)
                    
                    noise_preds = []
                    attention_maps_list = []
                    
                    # A. Forward pass for each LoRA to get noise predictions and attention maps
                    for j, lora_name in enumerate(lora_list):
                        lora_names = [lora_name] if lora_name else []
                        if style_lora:
                            lora_names.append(style_lora)
                        
                        lora_weights = [1.0] * len(lora_names)
                        if style_lora:
                            lora_weights[-1] = style_lora_weight
                        
                        self.set_adapters(lora_names, lora_weights)
                        
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        
                        added_cond_kwargs = {"text_embeds": add_text_embeds_list[j], "time_ids": add_time_ids_list[j]}
                        
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds_list[j],
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        
                        noise_preds.append(noise_pred)
                        attention_maps_list.append(attention_store.aggregate_attention(["down", "mid", "up"]))
                        attention_store.reset()

                    # B. Latent update via contrastive loss
                    if latent_update and i < max_iter_to_alter:
                        loss = self.loss_fn(attention_maps_list, important_token_indices)
                        latents = self.update_latents(latents, loss, step_size)
                        
                # Detach tensors for the blending part
                noise_preds = [p.detach() for p in noise_preds]
                noise_preds = torch.stack(noise_preds)
                noise_pred_uncond, noise_pred_text = noise_preds.chunk(2, dim=1)

                # C. Masked Blending
                masks = self.attention_map_to_mask(
                    attention_maps_list, mask_indices, size=(latents.shape[2], latents.shape[3]),
                    mask_threshold_alpha=mask_threshold_alpha,
                    mask_erode=mask_erode,
                    mask_dilate=mask_dilate,
                    mask_opening=mask_opening,
                    mask_closing=mask_closing,
                ).to(device=device, dtype=latents.dtype)
                
                # Blend the noise predictions using the masks
                noise_pred_uncond = (noise_pred_uncond * masks).sum(dim=0) / masks.sum(dim=0)
                noise_pred_text = (noise_pred_text * masks).sum(dim=0) / masks.sum(dim=0)

                # 5. Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0.0:
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # 6. Denoise one step
                latents = self.scheduler.step(noise_pred, t, latents.detach(), **extra_step_kwargs, return_dict=False)[0]

                progress_bar.update()

        # 7. Post-processing
        image = self.decode_latents(latents)
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, attention_maps_list, masks)

        return StableDiffusionXLPipelineOutput(images=image)
